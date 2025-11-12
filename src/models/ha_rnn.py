import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
import numpy as np
from omegaconf import DictConfig
import logging
import mlflow

from src.models.utils_lstm import VariationalLSTM
from src.models.basic_blocks import OutcomeHead


class PluginHistoryAdjustedNetwork(LightningModule):
    """
    Pytorch Lightning Module for the plug-in History Adjustment learner with a RNN backbone.
    The Conditional Average Potential Outcome is directly estimated by the nuisace paramer \delta_(h, a), where a
    is the treatment assignment and h is the history of the patient.
    The Conditional Average Treatment Effect is then estimated by the difference of the Conditional Average Potential
    Outcome -- TE(h, a, b) = CAPY(h, a) - CAPY(h, b) = \delta_(h, a) - \delta_(h, b)
    """
    def __init__(self, args:DictConfig):
        super().__init__()
        self.args = args
        self.model_type = 'plug-in history adjusted network'
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

        self.save_hyperparameters(args)
        self._initialize_model(args)
 
    def _initialize_model(self, args):
        """
        Initialze the LSTM encoder which encodes the history of the patient
        as a embeddding
        """

        self.hidden_size = args.model.hidden_size
        self.hr_size = args.model.hr_size
        self.num_layer = args.model.num_layer
        self.dropout_rate = args.model.dropout_rate
        self.fc_hidden_size = args.model.fc_hidden_size

        self.lstm = VariationalLSTM(self.input_size, self.hidden_size, self.num_layer, self.dropout_rate)
        self.hr_output_transformation = nn.Linear(self.hidden_size, self.hr_size)
        self.output_dropout = nn.Dropout(self.dropout_rate)

        self.capo_comp_head = OutcomeHead(self.hr_size + self.n_treatments * self.n_periods, self.fc_hidden_size, 1)
    
    def build_hr(self, static_features, curr_covariates, prev_treatments, prev_outputs):
        """
        Build the hidden representation of the patient state
        returns: hr: torch.tensor of shape (b, SL, hr_size)
        """
        static_features = static_features.unsqueeze(1).expand(-1, self.sequence_length, -1)   
        x = torch.cat([static_features, curr_covariates, prev_treatments, prev_outputs.unsqueeze(-1)], dim = -1)
        x = self.lstm(x, init_states=None)
        output = self.output_dropout(x)
        hr = nn.ELU()(self.hr_output_transformation(output))
        return hr
    
    def forward(self, batch: dict):
        """
        Forward pass of the model
        Computes the Conditional Average Potential Outcome of the batch
        returns: capo: torch.tensor of shape (b, SL - m + 1)
        """
        prev_outputs = batch['prev_outputs']
        b, L = prev_outputs.size(0), prev_outputs.size(1)
        prev_treatments_disc = batch['prev_treatments_disc'] if self.n_treatments_disc > 0 else torch.zeros((b, L, 0), device=self.device)
        prev_treatments_cont = batch['prev_treatments_cont'] if self.n_treatments_cont > 0 else torch.zeros((b, L, 0), device=self.device)
        prev_treatments = torch.cat([prev_treatments_disc, prev_treatments_cont], dim = -1)
        static_features = batch['static_features'] if self.n_static > 0 else torch.zeros((b, 0), device=self.device)
        curr_covariates = batch['curr_covariates']
        prev_outputs = batch['prev_outputs']
        curr_treatments_disc = batch['curr_treatments_disc'] if self.n_treatments_disc > 0 else torch.zeros((b, L, 0), device=self.device)
        curr_treatments_cont = batch['curr_treatments_cont'] if self.n_treatments_cont > 0 else torch.zeros((b, L, 0), device=self.device)
        curr_treatments = torch.cat([curr_treatments_disc, curr_treatments_cont], dim = -1)
        batch_size = prev_treatments.size(0)

        hr = self.build_hr(static_features, curr_covariates, prev_treatments, prev_outputs)
        #Concatenate hr with current treatment
        #hr_concat = torch.cat([hr, curr_treatments], dim = -1)

        capo = torch.zeros((batch_size, L - self.n_periods + 1), device = self.device)
        for t in range(L - self.n_periods + 1):
            concat_tensors = [hr[:, t, :]]
            for i in range(self.n_periods):
                concat_tensors.append(curr_treatments[:, t + i, :])
            hr_concat = torch.concat(concat_tensors, dim = -1)
            capo[:, t] = self.capo_comp_head.build_outcome(hr_concat).squeeze(-1)
        
        return capo
    
    def training_step(self, batch, batch_idx):
        """
        Compute the mse loss between predicted capo and the observed outcomes, 
        perform batch gradient descent by return the loss
        """
        curr_outputs = batch['curr_outputs']
        active_entries = batch['active_entries']
        if len(active_entries.shape) == 3:
            active_entries = active_entries[:, :, 0]

        pred_capo = self.forward(batch)
        active_y = active_entries[:, self.n_periods - 1:].to(torch.bool)
        loss = F.mse_loss(pred_capo[active_y], curr_outputs[:, self.n_periods - 1:][active_y], reduction = 'mean')
        self.log('train_loss', loss.item(), on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Compute the mse loss between predicted capo and the observed outcomes, 
        perform batch gradient descent by return the loss
        """
        curr_outputs = batch['curr_outputs']
        active_entries = batch['active_entries']
        if len(active_entries.shape) == 3:
            active_entries = active_entries[:, :, 0]

        pred_capo = self.forward(batch)
        active_y = active_entries[:, self.n_periods - 1:].to(torch.bool)
        loss = F.mse_loss(pred_capo[active_y], curr_outputs[:, self.n_periods - 1:][active_y], reduction = 'mean')
        self.log('val_loss', loss.item(), on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        return loss
    
    def predict_setp(self, batch, batch_idx):
        return self.forward(batch)
    
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
    
    def predict_capo(self, dataloader, T_intv_disc:np.ndarray, T_intv_cont:np.ndarray) -> np.ndarray:
        """
        Predict the Conditional Average Potential Outcome of the data (in np array)
        with the given intervention
        Note: The dataloader shouldn't be shuffled
        """
        batch_size = self.args.exp.batch_size
        SL = self.args.dataset.sequence_length
        T_intv = self._combine_disc_cont(T_intv_disc, T_intv_cont).reshape(1, self.n_periods * self.n_treatments) # Np array shape (1, m * n_treatments)
        T_intv_1 = torch.from_numpy(T_intv).float().to(self.device)
        capo_preds = []
        self.eval()
        for i, batch in enumerate(dataloader):
            #load the data as in forward pass except for the current treatment
            prev_outputs = batch['prev_outputs']
            b, L = prev_outputs.size(0), prev_outputs.size(1)
            prev_treatments_disc = batch['prev_treatments_disc'] if self.n_treatments_disc > 0 else torch.zeros((b, L, 0), device=self.device)
            prev_treatments_cont = batch['prev_treatments_cont'] if self.n_treatments_cont > 0 else torch.zeros((b, L, 0), device=self.device)
            prev_treatments = torch.cat([prev_treatments_disc, prev_treatments_cont], dim = -1)
            static_features = batch['static_features'] if self.n_static > 0 else torch.zeros((b, 0), device=self.device)
            curr_covariates = batch['curr_covariates']
            prev_outputs = batch['prev_outputs']
            batch_size = prev_treatments.size(0)

            T_intv = T_intv_1.expand(batch_size, -1)
            hr = self.build_hr(static_features, curr_covariates, prev_treatments, prev_outputs)
            capo = torch.zeros((batch_size, L - self.n_periods + 1), device = self.device)
            for t in range(L - self.n_periods + 1):
                hr_concat = torch.cat([hr[:, t, :], T_intv], dim = -1)
                capo[:, t] = self.capo_comp_head.build_outcome(hr_concat).squeeze(-1)
            capo_preds.append(capo)
        capo_pred = torch.cat(capo_preds, dim = 0)
        return capo_pred.detach().cpu().numpy()

                
            
            



    def _combine_disc_cont(self, T_disc, T_cont):
        """
        Combine discrete and continuous treatments
        """
        if T_disc is None:
            return T_cont
        elif T_cont is None:
            return T_disc
        else:
            assert T_disc.shape[:-1] == T_cont.shape[:-1]
            return np.concatenate([T_disc, T_cont], axis = -1)



        
