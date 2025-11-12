import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
import numpy as np
from omegaconf import DictConfig
from typing import List
import logging
import mlflow

from src.models.utils_lstm import VariationalLSTM
from src.models.basic_blocks import OutcomeHead, PropensityHead

class PropensityNetwork(LightningModule):
    """
    Pytorch Lightning Module for Propensity Network with a RNN backbone.
    The model should learn the propsensity scores in a time-varying setting -- 
    pi(a_t | h_{t}) := P(A_t = a_t | H_{t}), where H_{t} is the patient history, 
    h_t = {x_s}_{s=1}^{t} union {a_s}_{s=1}^{t-1} union {y_s}_{s=1}^{t-1}.
    if a_t^i is discrete, we use a sigmoid activation function to ensure that the output is in [0, 1]
    if a_t^i is continuous, we calculate the conditional density function 's mean and log-variance
    """
    def __init__(self, args:DictConfig):
        super().__init__()
        self.args = args
        self.model_type = 'PropensityNetwork'
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
        self.treatment_types = ['disc'] * self.n_treatments_disc + ['cont'] * self.n_treatments_cont

        self.save_hyperparameters(args)
        self._initialize_model(args)
    
    def _initialize_model(self, args):
        self.hidden_size = args.model.hidden_size
        self.hr_size = args.model.hr_size
        self.num_layer = args.model.num_layer
        self.dropout_rate = args.model.dropout_rate
        self.fc_hidden_size = args.model.fc_hidden_size

        #build rnn backbone
        self.lstm = VariationalLSTM(self.input_size, self.hidden_size, self.num_layer, self.dropout_rate)
        self.hr_output_transformation = nn.Linear(self.hidden_size, self.hr_size)
        self.output_dropout = nn.Dropout(self.dropout_rate)
        #propensity head
        self.propensity_heads = nn.ModuleList([
            PropensityHead(self.hr_size, self.fc_hidden_size, self.treatment_types[i]) for i in range(self.n_treatments)]
        )
    
    def build_hr(self, static_features, curr_covariates, prev_treatments, prev_outputs):
        """
        Build hidden representation (patient clinical state)
        """
        #expand static_features along the time dimension (b, n_static) -> (b, L, n_static)
        static_features = static_features.unsqueeze(1).expand(-1, self.sequence_length, -1)   
        x = torch.cat([static_features, curr_covariates, prev_treatments, prev_outputs.unsqueeze(-1)], dim = -1)
        x = self.lstm(x, init_states=None)
        output = self.output_dropout(x)
        hr = nn.ELU()(self.hr_output_transformation(output))
        return hr
    
    def forward(self, batch, mode = 'train'):
        """
        batch info:
        prev_treatments: torch.tensor shape (b, L, n_treatments)
        prev_outpus: torch.tensor shape (b, L, 1)
        returns:
        propensity_scores: 
        Eval: torch.tensor of shape (b, L, n_t) propensity_disc and propensity_cont concatenated, 
        pi(a_t | h_{t}) := P(A_t = a_t | H_{t})
        in training mode, returns list of sigmoid probas and (mu, log_var) pairs
        """
        prev_outputs = batch['prev_outputs']
        b, L = prev_outputs.size(0), prev_outputs.size(1)
        prev_treatments_disc = batch['prev_treatments_disc'] if self.n_treatments_disc > 0 else torch.zeros((b, L, 0), device=self.device)
        prev_treatments_cont = batch['prev_treatments_cont'] if self.n_treatments_cont > 0 else torch.zeros((b, L, 0), device=self.device)
        prev_treatments = torch.cat([prev_treatments_disc, prev_treatments_cont], dim = -1)
        static_features = batch['static_features'] if self.n_static > 0 else torch.zeros((b, 0), device=self.device)
        curr_covariates = batch['curr_covariates']
        prev_outputs = batch['prev_outputs']

        hr = self.build_hr(static_features, curr_covariates, prev_treatments, prev_outputs)
        outcomes = [self.propensity_heads[i].build_parameter(hr) for i in range(self.n_treatments)]
        if mode == 'train' or mode == 'val':
            return outcomes
        else:
            #'eval mode'
            #for binary curr_treatment, if a = 1 output proba, else 1 - proba
            curr_treatments_disc = batch['curr_treatments_disc'] if self.n_treatments_disc > 0 else torch.zeros((b, L, 0), device=self.device)
            curr_treatments_cont = batch['curr_treatments_cont'] if self.n_treatments_cont > 0 else torch.zeros((b, L, 0), device=self.device)
            
            if self.n_treatments_disc > 0:
                sigmoid_output = torch.cat(outcomes[:self.n_treatments_disc], dim = -1)
                #element-wise if curr_treatments_disc = 1, sigmoid_output, else 1 - sigmoid_output
                propensity_disc = torch.zeros_like(sigmoid_output, device = self.device)
                propensity_disc[curr_treatments_disc == 1] = sigmoid_output[curr_treatments_disc == 1]
                propensity_disc[curr_treatments_disc == 0] = 1 - sigmoid_output[curr_treatments_disc == 0]
            else:
                propensity_disc = torch.zeros((b, L, 0), device = self.device)
            
            if self.n_treatments_cont > 0:
                #For continuous treatment, output the density value of the pdf with mean mu and log-variance log_var
                mu_output = torch.cat([mu_logvar_pair[0] for mu_logvar_pair in outcomes[self.n_treatments_disc:]], dim=-1)
                log_var_output = torch.cat([mu_logvar_pair[1] for mu_logvar_pair in outcomes[self.n_treatments_disc:]], dim = -1)
                assert curr_treatments_cont.shape == mu_output.shape
                #element-wise density values of curr_treatments_cont with mean mu_output and log-variance log_var_output
                variance = torch.exp(log_var_output)
                propensity_cont = torch.exp(-0.5 * (curr_treatments_cont - mu_output) ** 2 / variance) / torch.sqrt(2 * np.pi * variance)
            else:
                propensity_cont = torch.zeros((b, L, 0), device = self.device)
            
            return torch.cat([propensity_disc, propensity_cont], dim = -1)
    
    def training_step(self, batch, batch_idx):
        outcomes = self(batch, mode = 'train')

        if self.n_treatments_disc > 0:
            #apply bce loss to the binary treatment
            curr_treatments_disc = batch['curr_treatments_disc']
            sigmoid_output = torch.cat(outcomes[:self.n_treatments_disc], dim = -1)
            assert sigmoid_output.shape == curr_treatments_disc.shape
            bce_loss = F.binary_cross_entropy(sigmoid_output, curr_treatments_disc, reduction = 'none').mean(dim = (0, 1))
            for i in range(self.n_treatments_disc):
                self.log(f'train_prop_bce[{i}]', bce_loss[i].item(), on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        else:
            bce_loss = torch.zeros(1, device = self.device)

        if self.n_treatments_cont > 0:
            curr_treatments_cont = batch['curr_treatments_cont']
            mu_output = torch.cat([mu_logvar_pair[0] for mu_logvar_pair in outcomes[self.n_treatments_disc:]], dim=-1)
            log_var_output = torch.cat([mu_logvar_pair[1] for mu_logvar_pair in outcomes[self.n_treatments_disc:]], dim = -1)
            #apply the negative log-likelihood loss to the continuous treatment
            nll_loss = self.gaussian_nll_loss(mu_output, log_var_output, curr_treatments_cont).mean(dim = (0, 1))
            for i in range(self.n_treatments_cont):
                self.log(f'train_prop_nll[{i}]', nll_loss[i].item(), on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        else:
            nll_loss = torch.zeros(1, device = self.device)
        
        loss = bce_loss.mean() + nll_loss.mean()
        self.log('train_prop_loss', loss.item(), on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outcomes = self(batch, mode = 'val')

        if self.n_treatments_disc > 0:
            #apply bce loss to the binary treatment
            curr_treatments_disc = batch['curr_treatments_disc']
            sigmoid_output = torch.cat(outcomes[:self.n_treatments_disc], dim = -1)
            assert sigmoid_output.shape == curr_treatments_disc.shape
            bce_loss = F.binary_cross_entropy(sigmoid_output, curr_treatments_disc, reduction = 'none').mean(dim = (0, 1))
            for i in range(self.n_treatments_disc):
                self.log(f'val_prop_bce[{i}]', bce_loss[i].item(), on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        else:
            bce_loss = torch.zeros(1, device = self.device)

        if self.n_treatments_cont > 0:
            curr_treatments_cont = batch['curr_treatments_cont']
            mu_output = torch.cat([mu_logvar_pair[0] for mu_logvar_pair in outcomes[self.n_treatments_disc:]], dim=-1)
            log_var_output = torch.cat([mu_logvar_pair[1] for mu_logvar_pair in outcomes[self.n_treatments_disc:]], dim = -1)
            #apply the negative log-likelihood loss to the continuous treatment
            nll_loss = self.gaussian_nll_loss(mu_output, log_var_output, curr_treatments_cont).mean(dim = (0, 1))
            for i in range(self.n_treatments_cont):
                self.log(f'val_prop_nll[{i}]', nll_loss[i].item(), on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        else:
            nll_loss = torch.zeros(1, device = self.device)
        
        loss = bce_loss.mean() + nll_loss.mean()
        self.log('val_prop_loss', loss.item(), on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        return loss
    

    def gaussian_nll_loss(self, mu, logvar, targets):
        """
        Negative log-likelihood loss for Gaussian distribution
        """
        log_2pi = torch.log(torch.tensor(2 * np.pi))
    
        # Compute the negative log-likelihood
        var = torch.exp(logvar)
        nll = 0.5 * (log_2pi + logvar + ((targets - mu)**2) / var)
    
        return nll
    
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



class InversePropensityWeightedNetwork(LightningModule):
    """
    Pytorch Lightning Module for Inverse Propensity Weighted Network with a RNN backbone.
    The model learns to predict the psuedo-capo of a fixed intervention a = {a_1, a_2, .., a_m} computed from the observed outcome 
    weighted by the inverse propensity scores computed from PropensityNetwork.
    Define pseudo-outcome Y_IPW_{t+m-1}^(a) = (multiply_{l=t}^{t+m-1} frac{Id(A_l=a_l)}{pi(a_l | h_l)} ) * Y_{t+m-1}.
    And then
    Warn: Should be cautious with the fraction and ensure numerical stability.
    Warn: Ensure pseduo-outcome of intervention and base are computed from non-empty sets
    """

    def __init__(self, args:DictConfig, propensity_network:PropensityNetwork = None, treatment_seq:np.ndarray = None):
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
        self.treatment_seq = torch.from_numpy(treatment_seq).to(torch.float32).to(self.device) if \
            treatment_seq is not None else torch.zeros((self.n_periods, self.n_treatments), device=self.device)
        self.propensity_network = propensity_network
        self.valid_entries = 0

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

        self.outcome_head = OutcomeHead(self.hr_size, self.fc_hidden_size, 1)

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
    
    def forward(self, batch):
        """
        get predicted pseudo-outcome of the given intervention : hat_Y_IPW_{t+m-1}^(a)
        Returns:
        pseudo-outcome: torch.tensor of shape (b, L, 1)
        Note: the entries are valid only for 0 <= t < L - m + 1, but we preserve the dimension of time for simplicity
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
        pseudo_outcome = self.outcome_head.build_outcome(hr)
        return pseudo_outcome
    
    def predict_step(self, batch, batch_idx = None):
        return self(batch)
    
    def get_psuedo_outcome(self, batch):
        """
        Get the predicted pseudo-outcome of the given intervention : hat_Y_IPW_{t+m-1}^(a)
        Only select the entries where the pseudo-outcome is valid, which means:
        1. those t+m-1 where A_l = a_l for all l in [t, t + m - 1] (a is treatment_seq with shape (m, n_treatments))
        2. then Y_pseudo_{t+m-1}^(a) = (multiply_{l=t}^{t+m-1} frac{Id(A_l=a_l)}{pi(a_l | h_l)} ) * Y_{t+m-1}.
        returns:
        pseudo_outcome: torch.tensor of shape (b, L - m + 1, 1)  Y_pseudo_{t+m-1}^(a) t = 0, 1, .., L - m
        mask: torch.tensor of shape (b, L, 1) where mask[b, t, 0] = 1 if the entry is valid, 0 otherwise
        """

        true_outcome = batch['curr_outputs']
        b, L = true_outcome.size(0), true_outcome.size(1)
        curr_treatments_disc = batch['curr_treatments_disc'] if self.n_treatments_disc > 0 else torch.zeros((b, L, 0), device=self.device)
        curr_treatments_cont = batch['curr_treatments_cont'] if self.n_treatments_cont > 0 else torch.zeros((b, L, 0), device=self.device)
        curr_treatments = torch.cat([curr_treatments_disc, curr_treatments_cont], dim = -1)
        m = self.n_periods

        with torch.no_grad():
            masks = list()
            propensities = list()
            propensity = self.propensity_network(batch, mode = 'eval')
            for offset in range(m): #offset = l - t
                target_treatment = self.treatment_seq[offset].unsqueeze(0).unsqueeze(0).to(self.device) #shape (1, 1, n_treatments)
                #get the mask
                mask = torch.zeros((b, L, 1), dtype = torch.bool, device=self.device)
                mask[:, m - 1:, :] = (curr_treatments[:, offset:offset + L - m + 1, :] == target_treatment).all(dim = -1, keepdim = True)
                masks.append(mask)
                propensities.append(torch.prod(propensity[:, offset:offset + L - m + 1, :], dim = -1, keepdim = True))
            #multiply the masks and invser of propensities
            mask = torch.prod(torch.stack(masks, dim = 0), dim = 0)
            prop_prod = torch.prod(torch.stack(propensities, dim = 0), dim = 0)
            
            pseudo_outcome = torch.clamp(mask[:, m - 1:, :] / (prop_prod + 1e-4) * true_outcome.unsqueeze(-1)[:, m - 1:, :], -10, 10)
            return pseudo_outcome, mask
    
    def training_step(self, batch, batch_idx):
        """
        Only select the entries where the pseudo-outcome is valid, which means:
        1. those t+m-1 where A_l = a_l for all l in [t, t + m - 1] (a is treatment_seq with shape (m, n_treatments))
        2. active entries given by batch['active_entries']
        if no valid entries, return 0
        record the number of valid entries
        """
        pred_pseudo_outcome = self(batch)
        pseudo_outcome, mask = self.get_psuedo_outcome(batch)
        active_entries = batch['active_entries']
        mask = mask * active_entries.unsqueeze(-1)
        self.valid_entries += mask.sum().item()
        mask = mask.to(torch.bool)
        mse_loss = F.mse_loss(pred_pseudo_outcome[mask], pseudo_outcome[mask[:, self.n_periods - 1:, :]], reduction = 'mean')
        #Note that mask could be all False, which leads to nan loss, in this case we set the loss to 0
        if torch.isnan(mse_loss):
            mse_loss = torch.zeros(1, device=self.device, requires_grad=True)
        self.log('train_po_loss', mse_loss.item(), on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)

        return mse_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Only select the entries where the pseudo-outcome is valid, which means:
        1. those t+m-1 where A_l = a_l for all l in [t, t + m - 1] (a is treatment_seq with shape (m, n_treatments))
        2. active entries given by batch['active_entries']
        if no valid entries, return 0
        record the number of valid entries
        """
        pred_pseudo_outcome = self(batch)
        pseudo_outcome, mask = self.get_psuedo_outcome(batch)
        active_entries = batch['active_entries']
        mask = mask * active_entries.unsqueeze(-1)
        self.valid_entries += mask.sum().item()
        mask = mask.to(torch.bool)
        mse_loss = F.mse_loss(pred_pseudo_outcome[mask], pseudo_outcome[mask[:, self.n_periods - 1:, :]], reduction = 'mean')
        #Note that mask could be all False, which leads to nan loss, in this case we set the loss to 0
        if torch.isnan(mse_loss):
            mse_loss = torch.zeros(1, device=self.device, requires_grad=True)
        self.log('val_po_loss', mse_loss.item(), on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)

        return mse_loss
    
    def on_peoch_end(self):
        """
        Log the sum of total valid entries
        """
        self.log('valid_entries',self.valid_entries, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        self.valid_entries = 0
        return
    
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
        


        