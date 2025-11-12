import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
import numpy as np
from omegaconf import DictConfig
import logging
import mlflow
from typing import List, Tuple

from src.models.utils_transformer import AbsolutePositionalEncoding, RelativePositionalEncoding, TransformerMultiInputBlock
from src.models.basic_blocks import OutcomeHead_GRNN

logger = logging.getLogger(__name__)


class G_transformer(LightningModule):
    """
    Pytorch Lightning Module for the G-transformer https://github.com/konstantinhess/G_transformer
    The CAPO is estimated using the G-computation formula with a regression model for the potential outcomes.
    The regression model is a LSTM with variational dropout.
    We use \mu_t(h_t, a) =  E[Y_{t+m}^(do(a))|H_t=h_t] to represent the nuisance parameter of the potential outcome model
    For Plug-in learner, the CATE TE(h_t, a, b) is estimated as \mu_t(h_t, a) - \mu_t(h_t, b).
    """
    def __init__(self, args:DictConfig, treatment_sequence: np.ndarray = None):
        super().__init__()
        self.args = args
        self.model_type = 'plug-in G-comp'
        dataset_params = args.dataset
        self.n_treatments = dataset_params['n_treatments']
        self.n_treatments_disc = dataset_params.get('n_treatments_disc', 0)
        self.n_treatments_cont = dataset_params.get('n_treatments_cont', 0)
        assert self.n_treatments == self.n_treatments_disc + self.n_treatments_cont
        self.n_x = dataset_params['n_x']
        self.n_static = dataset_params.get('n_static', 0)
        self.n_periods = dataset_params['n_periods']
        self.input_size = self.n_treatments + self.n_x + self.n_static + 1
        self.sequence_length = dataset_params['sequence_length']
        self.projection_horizon = self.n_periods - 1
        self.dim_vitals = self.n_x
        self.dim_treatments = self.n_treatments
        self.dim_static_features = self.n_static
        self.dim_outcome = 1
        self.has_vitals = True if self.dim_vitals > 0 else False
        self.treatment_sequence = torch.tensor(args.model.treatment_sequence)[: self.projection_horizon+1, :] \
            if treatment_sequence is None else torch.from_numpy(treatment_sequence).to(self.device)

        self.save_hyperparameters(args)
        self._initialize_model(args)

    def _initialize_model(self, args: DictConfig):
        self.max_seq_length = self.sequence_length
        self.hr_size = args.model.hr_size
        self.seq_hidden_units = args.model.hidden_size
        self.fc_hidden_units = args.model.fc_hidden_size
        self.dropout_rate = args.model.dropout_rate
        self.num_layer = args.model.num_layer
        self.num_heads = args.model.num_heads
        self.pe_trainable = args.model.positional_encoding.trainable
        self.relative_position = args.model.max_relative_position

        self.head_size = self.seq_hidden_units // self.num_heads
        
        self.self_positional_encoding = RelativePositionalEncoding(self.relative_position, self.head_size, trainable = True)
        self.self_positional_encoding_k = RelativePositionalEncoding(self.relative_position, self.head_size, trainable = True)
        self.self_positional_encoding_v = RelativePositionalEncoding(self.relative_position, self.head_size, trainable = True)

        self.treatments_input_transformation = nn.Linear(self.dim_treatments, self.seq_hidden_units)
        self.vitals_input_transformation = nn.Linear(self.dim_vitals, self.seq_hidden_units)
        self.output_input_transformation = nn.Linear(1, self.seq_hidden_units)
        self.static_input_transformation = nn.Linear(self.dim_static_features, self.seq_hidden_units)

        self.n_inputs = 3 if self.has_vitals else 2
        self.transformer_blocks = nn.ModuleList([
            TransformerMultiInputBlock(self.seq_hidden_units, 
                                       self.num_heads, 
                                       self.head_size, 
                                       self.seq_hidden_units * 4,
                                       self.dropout_rate,
                                       self.dropout_rate,
                                       self_positional_encoding_k=self.self_positional_encoding_k,
                                       self_positional_encoding_v=self.self_positional_encoding_v,
                                       n_inputs=self.n_inputs,
                                       disable_cross_attention=False,
                                       isolate_subnetwork='_')
        ])
        self.hr_output_transformation = nn.Linear(self.seq_hidden_units, self.hr_size)
        self.output_dropout = nn.Dropout(self.dropout_rate)

        # G-computation heads: nested expectations
        self.G_comp_heads = nn.ModuleList(
            [OutcomeHead_GRNN(self.seq_hidden_units, self.hr_size, self.fc_hidden_units, self.dim_treatments, 1)
             for _ in range(self.projection_horizon + 1)]
        )


    def build_hr(self, prev_treatments, vitals, prev_outputs, static_features, active_entries):

        active_entries_treat_outcomes = torch.clone(active_entries)
        active_entries_vitals = torch.clone(active_entries)
        T = prev_treatments.size(1)

        x_t = self.treatments_input_transformation(prev_treatments)
        x_o = self.output_input_transformation(prev_outputs.unsqueeze(-1))
        x_v = self.vitals_input_transformation(vitals) if self.has_vitals else None
        x_s = self.static_input_transformation(static_features.unsqueeze(1)).expand(-1, T, -1)

        for block in self.transformer_blocks:
            if self.self_positional_encoding is not None:
                x_t = x_t + self.self_positional_encoding(x_t)
                x_o = x_o + self.self_positional_encoding(x_o)
                x_v = x_v + self.self_positional_encoding(x_v) if self.has_vitals else None 
            
            x_t, x_o, x_v = block((x_t, x_o, x_v), x_s, active_entries_treat_outcomes, active_entries_vitals)

        x = (x_t + x_o + x_v) / 3 if self.has_vitals else (x_t + x_o) / 2
        x = self.output_dropout(x)
        hr = nn.ELU()(self.hr_output_transformation(x))

        return hr
    
    def forward(self, batch, mode = 'test'):
        
        prev_outputs = batch['prev_outputs']
        b, L = prev_outputs.size(0), prev_outputs.size(1)
        prev_treatments_disc = batch['prev_treatments_disc'] if self.n_treatments_disc > 0 else torch.zeros((b, L, 0), device=self.device)
        prev_treatments_cont = batch['prev_treatments_cont'] if self.n_treatments_cont > 0 else torch.zeros((b, L, 0), device=self.device)
        prev_treatments = torch.cat([prev_treatments_disc, prev_treatments_cont], dim = -1)
        static_features = batch['static_features'] if self.n_static > 0 else torch.zeros((b, 0), device=self.device)
        vitals = batch['curr_covariates']
        curr_treatments_disc = batch['curr_treatments_disc'] if self.n_treatments_disc > 0 else torch.zeros((b, L, 0), device=self.device)
        curr_treatments_cont = batch['curr_treatments_cont'] if self.n_treatments_cont > 0 else torch.zeros((b, L, 0), device=self.device)
        curr_treatments = torch.cat([curr_treatments_disc, curr_treatments_cont], dim = -1)
        active_entries = batch['active_entries'].clone()

        batch_size = prev_treatments.size(0)
        time_dim = prev_treatments.size(1)

        #If in training or validation mode
        if mode == 'train' or mode == 'val' or self.training:

            # 1) train all hidden states on factual data
            if self.projection_horizon == 0:
                hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, active_entries)
                pred_factuals = self.G_comp_heads[0].build_outcome(hr, curr_treatments)
                pseudo_outcomes = pred_pseudos = None

                return pred_factuals, pred_pseudos, pseudo_outcomes, active_entries


            else:
                # 2) G-computation formula: iterate over all time steps
                # 2a) Initialize
                pseudo_outcomes_all_steps = torch.zeros((batch_size, time_dim-self.projection_horizon-1,
                                                         self.projection_horizon+1, self.dim_outcome), device=self.device)
                pred_pseudos_all_steps = torch.zeros((batch_size, time_dim-self.projection_horizon-1,
                                                      self.projection_horizon+1, self.dim_outcome), device=self.device)
                active_entries_all_steps = torch.zeros((batch_size, time_dim-self.projection_horizon-1, 1), device=self.device)

                for t in range(1, time_dim-self.projection_horizon):
                    current_active_entries = batch['active_entries'].clone().squeeze(-1)
                    current_active_entries[:, int(t + self.projection_horizon):] = 0.0
                    active_entries_all_steps[:, t-1,:] = current_active_entries[:, t+self.projection_horizon-1].unsqueeze(-1)

                    # 2b) Generate pseudo outcomes
                    with torch.no_grad():
                        indexes_cf = (torch.arange(0, time_dim, device=self.device) >= t-1)*(
                                torch.arange(0, time_dim, device=self.device) < t+self.projection_horizon)
                        curr_treatments_cf = curr_treatments.clone()
                        curr_treatments_cf[:,indexes_cf,:] = self.treatment_sequence.to(self.device)
                        prev_treatments_cf = torch.concat((prev_treatments[:, :1, :], curr_treatments_cf[:, :-1, :]), dim=1)

                        hr_cf = self.build_hr(prev_treatments_cf, vitals, prev_outputs, static_features, current_active_entries)
                        pseudo_outcomes = torch.zeros((batch_size, self.projection_horizon+1, self.dim_outcome), device=self.device)

                        for i in range(self.projection_horizon, 0, -1):
                            pseudo_outcome = self.G_comp_heads[i].build_outcome(hr_cf, curr_treatments_cf)[:, t+i-1, :]
                            pseudo_outcomes[:, i-1, :] = pseudo_outcome
                        pseudo_outcomes[:, -1, :] = batch['curr_outputs'][:, t+self.projection_horizon-1].unsqueeze(-1)
                        # Store pseudo outcomes
                        pseudo_outcomes_all_steps[:, t-1, :, :] = pseudo_outcomes

                    # 2c) Predict pseudo outcomes with G-computation heads
                    hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, current_active_entries)
                    pred_pseudos = torch.zeros((batch_size, self.projection_horizon + 1, self.dim_outcome), device=self.device)
                    for i in range(self.projection_horizon, -1, -1):
                        pred_pseudo = self.G_comp_heads[i].build_outcome(hr, curr_treatments)[:, t+i-1, :]
                        pred_pseudos[:, i, :] = pred_pseudo
                    # Store predicted pseudo outcomes
                    pred_pseudos_all_steps[:, t-1, :, :] = pred_pseudos

                return None, pred_pseudos_all_steps, pseudo_outcomes_all_steps, active_entries_all_steps

        # 3) Prediction: only use the first head ("=outermost expectation")
        else:
            sequence_lengths = batch['active_entries'].sum(dim=1)
            fixed_split = sequence_lengths - self.projection_horizon if self.projection_horizon > 0 else batch['sequence_lengths']
            for i in range(len(active_entries)):
                active_entries[i, int(fixed_split[i] + self.projection_horizon):] = 0.0

            hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, active_entries)
            if self.projection_horizon > 0:
                pred_outcomes = self.G_comp_heads[0].build_outcome(hr, curr_treatments) # shape (b, SL, 1)
                index_pred = (torch.arange(0, time_dim, device=self.device) == fixed_split[..., None] - 1)
                pred_outcomes = pred_outcomes[index_pred]
            else:
                pred_outcomes = self.G_comp_heads[0].build_outcome(hr, curr_treatments)

            return pred_outcomes, hr
        
    def training_step(self, batch, batch_ind, optimizer_idx=None):

        for par in self.parameters():
            par.requires_grad = True

        pred_factuals, pred_pseudos, pseudo_outcomes, active_entries_all_steps = self(batch, mode = 'train')

        if self.projection_horizon > 0:

            active_entries_all_steps = active_entries_all_steps.unsqueeze(-2)
            mse_gcomp = F.mse_loss(pred_pseudos, pseudo_outcomes, reduction='none')
            mse_gcomp = (mse_gcomp * active_entries_all_steps).sum(dim=(0,1)) / (active_entries_all_steps.sum(dim=(0,1)) * self.dim_outcome)

            for i in range(mse_gcomp.shape[0]):
                self.log(f'train_mse_'+str(i), mse_gcomp[i].mean(),
                        on_epoch=True, on_step=False, sync_dist=True, prog_bar=False)

            loss = mse_gcomp.mean()

        else:
            mse_factual = F.mse_loss(pred_factuals, batch['curr_outputs'].unsqueeze(-1), reduction='none')
            mse_factual = (mse_factual * batch['active_entries']).sum() / (batch['active_entries'].sum() * self.dim_outcome)
            loss = mse_factual

        self.log(f'train_loss', loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_ind, optimizer_idx=None):

        pred_factuals, pred_pseudos, pseudo_outcomes, active_entries_all_steps = self(batch, mode = 'val')

        if self.projection_horizon > 0:

            active_entries_all_steps = active_entries_all_steps.unsqueeze(-2)
            mse_gcomp = F.mse_loss(pred_pseudos, pseudo_outcomes, reduction='none')
            mse_gcomp = (mse_gcomp * active_entries_all_steps).sum(dim=(0,1)) / (active_entries_all_steps.sum(dim=(0,1)) * self.dim_outcome)

            for i in range(mse_gcomp.shape[0]):
                self.log(f'val_mse_'+str(i), mse_gcomp[i].mean(),
                        on_epoch=True, on_step=False, sync_dist=True, prog_bar=False)

            loss = mse_gcomp.mean()

        else:
            mse_factual = F.mse_loss(pred_factuals, batch['curr_outputs'].unsqueeze(-1), reduction='none')
            mse_factual = (mse_factual * batch['active_entries']).sum() / (batch['active_entries'].sum() * self.dim_outcome)
            loss = mse_factual

        self.log(f'val_loss', loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)

        return loss
    
    def predict_step(self, batch, batch_idx, dataset_idx=None):
        """
        Generates capo predictions for a batch of data.
        """
        outcome_pred, hr = self(batch)
        return outcome_pred.cpu(), hr.cpu()
    
    def predict_capo(self, testloader) -> Tuple[np.ndarray]:
        """
        Predicts the counterfactual outcomes for the given treatment sequence.
        """
        intv = self.treatment_sequence.clone()
        logger.info(f'Predicting capo for treatment sequence: {intv.cpu().numpy()}')
        capo_preds = []
        active_masks = []
        self.eval()
        for i, batch in enumerate(testloader):
            prev_outputs = batch['prev_outputs']
            b, L = prev_outputs.size(0), prev_outputs.size(1)
            prev_treatments_disc = batch['prev_treatments_disc'] if self.n_treatments_disc > 0 else torch.zeros((b, L, 0), device=self.device)
            prev_treatments_cont = batch['prev_treatments_cont'] if self.n_treatments_cont > 0 else torch.zeros((b, L, 0), device=self.device)
            prev_treatments = torch.cat([prev_treatments_disc, prev_treatments_cont], dim = -1)
            static_features = batch['static_features'] if self.n_static > 0 else torch.zeros((b, 0), device=self.device)
            vitals = batch['curr_covariates']
            active_entries = batch['active_entries'].clone().squeeze(-1)

            batch_size = prev_treatments.size(0)
            time_dim = prev_treatments.size(1)

            hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, active_entries)
            assert self.projection_horizon > 0, 'for capo predictions projection horizon should be greater than 0'
            intv_expanded = intv[0, :].unsqueeze(0).unsqueeze(0).expand(batch_size, time_dim, -1)
            pred_outcome = self.G_comp_heads[0].build_outcome(hr, intv_expanded).squeeze(-1) # shape (b, SL)
            capo_preds.append(pred_outcome[:, :time_dim - self.n_periods + 1])
            active_masks.append(active_entries[:, self.n_periods - 1:].to(bool))
        capo_pred = torch.cat(capo_preds, dim=0).detach().cpu().numpy()
        active_mask = torch.cat(active_masks, dim=0).detach().cpu().numpy()
        return capo_pred, active_mask
    
    def get_predictions(self, dataset) -> np.array:
        logger.info(f'Predictions for {dataset.subset_name}.')
        # Creating Dataloader
        data_loader = DataLoader(dataset, batch_size=self.args.exp.batch_size, shuffle=False)
        outcome_pred, _ = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))] # call predict_step(...), which returns predictions and hr
        return outcome_pred.numpy()
    
    def configure_optimizers(self):
        # Select optimizer based on config
        opt_args = self.hparams.model.optimizer
        optimizer_cls = opt_args['optimizer_cls']
        if optimizer_cls.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=opt_args.learning_rate, weight_decay=opt_args.weight_decay)
        elif optimizer_cls.lower() == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=opt_args.learning_rate, weight_decay=opt_args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_cls.lower()}")
        
        scheduler_cls = opt_args["lr_scheduler_cls"]
        if scheduler_cls == "ExponentialLR":
            scheduler = ExponentialLR(optimizer, gamma=opt_args["gamma"])

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'reduce_on_plateau': False,
                'monitor': 'val_loss_epoch',
            }
        }
    

    





