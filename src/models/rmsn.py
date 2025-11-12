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

class RMSN(LightningModule):
    """
    pytorch lightning implementation Recurrent Marginal Structural Networks (RMSNs)
    RMSN is the base class for the four chilren classes:
    1. treatment propensity network
    2. history propensity network
    3. encoder network
    4. decoder network
    Since they share the same structure of LSTM + output head, we can implement them in the same class.
    """

    possible_model_types = [
        'treatment_propensity_network',
        'history_propensity_network',
        'encoder_network',
        'decoder_network'
    ]

    def __init__(self, args:DictConfig):
        super().__init__()
        self.args = args
        #self.model_type = 'RMSN-Base'
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
        self.output_size = None

        assert self.n_treatments_cont == 0, "RMSN does not support continuous treatments yet."
        self.save_hyperparameters(args)
        #self._initialize_model(args)

    def _initialize_model(self, args:DictConfig):
        """
        Initialize the model based on the model type.
        :param args: arguments from the config file
        """
        self.max_seq_length = self.sequence_length
        self.hr_size = args.model.hr_size
        self.seq_hidden_units = args.model.hidden_size
        self.fc_hidden_units = args.model.fc_hidden_size
        self.dropout_rate = args.model.dropout_rate
        self.num_layer = args.model.num_layer

        assert self.model_type in self.possible_model_types, f"Model type {self.model_type} is not supported."
        if self.model_type == "decoder_network":
            self.memory_adapter = nn.Linear(self.encoder_hidden_units, self.seq_hidden_units)
        self.lstm = VariationalLSTM(input_size=self.input_size,
                                    hidden_size=self.seq_hidden_units,
                                    num_layer=self.num_layer,
                                    dropout_rate=self.dropout_rate,
                                    )
        assert self.output_size is not None, "Output size is not set. Please set the output size before initializing the model."
        if self.model_type == 'encoder_network':
            self.output_layer = nn.Linear(self.seq_hidden_units + self.dim_treatments, self.output_size)
        else:
            self.output_layer = nn.Linear(self.seq_hidden_units, self.output_size)
        #self.dropout_layer = nn.Dropout(self.dropout_rate)
    
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
        
    

class RMSNTreatmentPropensityNetwork(RMSN):
    """
    Treatment propensity network
    preidcts P(A_t \mid \overline_{A_{t-1}}) (condtioned on all past treatments)
    """

    def __init__(self, args:DictConfig):

        self.model_type = 'treatment_propensity_network'
        super().__init__(args)
        
        self.input_size = self.n_treatments_disc
        self.output_size = self.n_treatments_disc
        self._initialize_model(args)

    def forward(self, batch):
        """
        Forward pass the logits of the treatment
        """
        prev_treatments = batch['prev_treatments_disc']
        x = self.lstm(prev_treatments, init_states=None)
        #x = self.dropout_layer(x)
        return self.output_layer(x)

    def training_step(self, batch, batch_idx):
        """
        Train the propensity treatment network with bce loss
        """
        treatment_probs = F.sigmoid(self(batch)) # (b, seq_len, n_treatments_disc)
        curr_treatments = batch['curr_treatments_disc']
        active_entries = batch['active_entries']
        if len(active_entries.shape) == 2:
            active_entries = active_entries.unsqueeze(-1)
        loss = F.binary_cross_entropy(treatment_probs, curr_treatments)
        valid_loss = (loss * active_entries).sum() / active_entries.sum()
        self.log('train_loss_t', valid_loss, prog_bar=True, logger=True)

        return valid_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validate the propensity treatment network with bce loss
        """
        treatment_probs = F.sigmoid(self(batch))
        curr_treatments = batch['curr_treatments_disc']
        active_entries = batch['active_entries']
        if len(active_entries.shape) == 2:
            active_entries = active_entries.unsqueeze(-1)
        loss = F.binary_cross_entropy(treatment_probs, curr_treatments)
        valid_loss = (loss * active_entries).sum() / active_entries.sum()
        self.log('val_loss_t', valid_loss, prog_bar=True, logger=True)
        #log accuracy
        pred = (treatment_probs > 0.5).float()
        acc = (pred == curr_treatments).float().mean().item()
        self.log('val_acc_t', acc, prog_bar=True, logger=True)
        return valid_loss
    
    def predict_step(self, batch, batch_idx):
        """
        Predict the treatment probabilities
        """
        treatment_probs = F.sigmoid(self(batch))
        return treatment_probs.cpu()
    


class RMSNHistoryPropensityNetwork(RMSN):
    """
    History propensity network
    preidcts P(A_t \mid \overline_{H_{t}}) (condtioned on all past history)
    """

    def __init__(self, args:DictConfig):
        super().__init__(args)
        self.model_type = 'history_propensity_network'
        self.input_size = self.n_treatments_disc + self.n_x + self.n_static + 1
        self.output_size = self.n_treatments_disc
        self._initialize_model(args)

    def forward(self, batch):
        """
        Forward pass the logits of the treatment
        """
        prev_treatments = batch['prev_treatments_disc']
        vitals = batch['curr_covariates']
        static_features = batch['static_features']
        prev_outputs = batch['prev_outputs']
        T = prev_outputs.shape[1]
    
        x = torch.cat((prev_treatments, vitals, static_features.unsqueeze(1).expand(-1, T, -1),
                       prev_outputs.unsqueeze(-1)), dim=-1)
        x = self.lstm(x, init_states=None)
        #x = self.dropout_layer(x)
        return self.output_layer(x)
    
    def training_step(self, batch, batch_idx):
        """
        Train the propensity history network with bce loss
        """
        treatment_probs = F.sigmoid(self(batch))
        curr_treatments = batch['curr_treatments_disc']
        active_entries = batch['active_entries']
        if len(active_entries.shape) == 2:
            active_entries = active_entries.unsqueeze(-1)
        loss = F.binary_cross_entropy(treatment_probs, curr_treatments)
        valid_loss = (loss * active_entries).sum() / active_entries.sum()
        self.log('train_loss_h', valid_loss.item(), prog_bar=True, logger=True)
        return valid_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validate the propensity history network with bce loss
        """
        treatment_probs = F.sigmoid(self(batch))
        curr_treatments = batch['curr_treatments_disc']
        active_entries = batch['active_entries']
        if len(active_entries.shape) == 2:
            active_entries = active_entries.unsqueeze(-1)
        loss = F.binary_cross_entropy(treatment_probs, curr_treatments)
        valid_loss = (loss * active_entries).sum() / active_entries.sum()
        self.log('val_loss_h', valid_loss.item(), prog_bar=True, logger=True)
        #log accuracy
        pred = (treatment_probs > 0.5).float()
        acc = (pred == curr_treatments).float().mean().item()  
        self.log('val_acc_h', acc, prog_bar=True, logger=True)
        return valid_loss
    
    def predict_step(self, batch, batch_idx):
        """
        Predict the treatment probabilities
        """
        treatment_probs = F.sigmoid(self(batch))
        return treatment_probs.cpu()



def compute_stabilized_weights(dataloader, treatment_propensity, history_propensity):
    """
    Compute the stabilized weights for the treatment and history propensity networks.

    SW(t,\tau) = \prod_{n=t}^{t+\tau} \frac{f(\mathbf{A}_n | \bar{\mathbf{A}}_{n-1})}{f(\mathbf{A}_n | \bar{\mathbf{H}}_n)},

    SW_tilde(t, \tau) is achieved by computing the product of the treatment propensity and history propensity
    
    returns the normalized stabilized weights for each treatment at each time step.
    Shape (N, T, tau + 1)
    """
    b = dataloader.batch_size
    T = treatment_propensity.max_seq_length
    tau = treatment_propensity.projection_horizon
    n_t = treatment_propensity.n_treatments_disc
    prop_h_list = list()
    prop_t_list = list()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            prop_t = treatment_propensity.predict_step(batch, i)
            prop_h = history_propensity.predict_step(batch, i)
            prop_h_list.append(prop_h)
            prop_t_list.append(prop_t)
    
    prop_h = torch.cat(prop_h_list, dim=0) # (N, T, n_treatments_disc)
    prop_t = torch.cat(prop_t_list, dim=0) # (N, T, n_treatments_disc)

    N = prop_h.shape[0]
    #pad prop_h and prop_t to (N, T + tau, n_treatments_disc)
    prop_h = torch.cat((prop_h, torch.ones((N, tau, n_t))), dim = 1)
    prop_t = torch.cat((prop_t, torch.ones((N, tau, n_t))), dim = 1)
    #
    cumprod_t = torch.ones((N, T, n_t, tau + 1))
    cumprod_h = torch.ones((N, T, n_t, tau + 1))
    cumprod_t[:, :, :, 0] = prop_t[:, 0 : T, :]
    cumprod_h[:, :, :, 0] = prop_h[:, 0 : T, :]
    for k in range(1, tau + 1):
        cumprod_t[:, :, :, k] = cumprod_t[:, :, :, k - 1] * prop_t[:, k: T + k, :]
        cumprod_h[:, :, :, k] = cumprod_h[:, :, :, k - 1] * prop_h[:, k: T + k, :]
    
    sw = cumprod_t / (cumprod_h + 1e-8) # (N, T, n_treatments_disc, tau + 1)
    sw = torch.prod(sw, dim = 2) # (N, T, tau + 1)
    # (For simplicity we just use 0.01 and 0.99) (original work truncate at 1st and 99-th perentile)
    sw = torch.clamp(sw, min=0.01, max=0.99)
    #Compute the mean
    sw_mean = sw.mean(dim = (0, 1))
    # divide sw by the mean
    sw_normalized = sw / sw_mean.unsqueeze(0).unsqueeze(0).expand(N, T, -1)
    return sw_normalized






class RMSNEncoderNetwork(RMSN):
    """
    Encoder network
    Encoder is trained to predict the $0$-step ahead outcome with the computed $SW(\cdot,1)$-weighted MSE. 
    where: 
    SW(t,\tau) = \prod_{n=t}^{t+\tau} \frac{f(\mathbf{A}_n | \bar{\mathbf{A}}_{n-1})}{f(\mathbf{A}_n | \bar{\mathbf{H}}_n)},
    """

    def __init__(self, args:DictConfig):
        super().__init__(args)
        self.model_type = 'encoder_network'
        self.input_size = self.n_treatments_disc + self.n_x + self.n_static + 1
        self.output_size = 1
        self._initialize_model(args)

    def forward(self, batch):
        """
        Forward pass the logits of the treatment
        """
        prev_treatments = batch['prev_treatments_disc']
        vitals = batch['curr_covariates']
        static_features = batch['static_features']
        prev_outputs = batch['prev_outputs']
        curr_treatments = batch['curr_treatments_disc']
        T = prev_outputs.shape[1]
    
        x = torch.cat((prev_treatments, vitals, static_features.unsqueeze(1).expand(-1, T, -1), 
                       prev_outputs.unsqueeze(-1)), dim=-1)
        x = self.lstm(x, init_states=None)
        x = torch.cat((x, curr_treatments), dim=-1)

        return self.output_layer(x)
    
    def training_step(self, batch, batch_idx):
        """
        Train the encoder network with SW weighted mse loss
        """
        pred_output = self(batch).squeeze(-1)
        curr_output = batch['curr_outputs']
        active_entries = batch['active_entries']
        assert 'sw_enc' in batch, "SW weights are not provided in the batch."
        sw_enc = batch['sw_enc']
        if len(active_entries.shape) == 3:
            active_entries = active_entries.squeeze(-1)
        
        weighted_mse = F.mse_loss(pred_output, curr_output, reduction='none') * sw_enc
        valid_loss = (weighted_mse * active_entries).sum() / active_entries.sum()
        self.log('train_loss_enc', valid_loss.item(), prog_bar=True, logger=True)
        return valid_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validate the encoder network with SW weighted mse loss
        """
        pred_output = self(batch).squeeze(-1)
        curr_output = batch['curr_outputs']
        active_entries = batch['active_entries']
        assert 'sw_enc' in batch, "SW weights are not provided in the batch."
        sw_enc = batch['sw_enc']
        if len(active_entries.shape) == 3:
            active_entries = active_entries.squeeze(-1)
        
        weighted_mse = F.mse_loss(pred_output, curr_output, reduction='none') * sw_enc
        valid_loss = (weighted_mse * active_entries).sum() / active_entries.sum()
        self.log('val_loss_enc', valid_loss.item(), prog_bar=True, logger=True)
        return valid_loss
    
    def predict_step(self, batch, batch_idx):
        """
        Predict the treatment probabilities
        """
        pred_output = self(batch)
        return pred_output.cpu()
    

class RMSNDecoderNetwork(RMSN):
    """
    Decoder network takes the latent representation from the encoder 
    and then transform the vector into the latent representation fit for the 
    decoder's LSTM via a memory adapter (normally a fully connected linear layer). 
    The decoder then predict the $0$ to $\tau$-step outcome with the $SW(\cdot,\tau)$-weighted MSE.
    """

    def __init__(self, args:DictConfig, encoder:RMSNEncoderNetwork):
        super().__init__(args)
        self.model_type = 'decoder_network'
        self.input_size = self.n_treatments_disc + self.n_static + 1
        self.output_size = 1
        self.encoder = encoder
        self.encoder_hidden_units = encoder.seq_hidden_units
        self._initialize_model(args)

    def forward(self, batch):
        """
        Predict the \tau step ahead outcome with the $SW(\cdot,\tau)$-weighted MSE. For t leq T - \tau, The decoder network
        takes latent representation at time t from the encoder network, and then decode the latent representation 
        gradually into the outcome at time t + \tau
        Returns
        The predicted outcome for t+1,..,t+\tau in shape (b, T - \tau, \tau)
        """
        prev_treatments = batch['prev_treatments_disc']
        vitals = batch['curr_covariates']
        static_features = batch['static_features']
        prev_outputs = batch['prev_outputs']
        curr_treatments = batch['curr_treatments_disc']
        T = prev_outputs.shape[1]

        with torch.no_grad():
            x = torch.cat((prev_treatments, vitals, static_features.unsqueeze(1).expand(-1, T, -1), 
                        prev_outputs.unsqueeze(-1)), dim=-1)
            x = self.encoder.lstm(x, init_states=None) # x encodes information of H_t = (\overline{X_t}, \overline{A}_{t-1}, \overline{Y}_{t-1})
            # predict Y_t with x by encoder
            Yp = self.encoder.output_layer(torch.cat((x, curr_treatments), dim=-1))
        
        
        #transform to decoder hidden state
        hr_decoder = self.memory_adapter(x)

        result = list()
        T, tau = self.max_seq_length, self.projection_horizon
        assert len(Yp.shape) == 3, "Yp should be of shape (b, T, 1)"

        # for t = 1,2,...,T - \tau, we need to predict Y_{t+1},...,Y_{t+\tau} by iteration
        # each iteration takes the previous prediction Y_{t+k} as input to predict Y_{t+k+1}
        for t in range(T - tau):
            h_t = hr_decoder[:, t, :]
            Yp_prev = Yp[:, t, :] # initialize
            predicted_outcomes = list()
            hidden_states = [(h_t, h_t) for _ in range(self.num_layer)]
            for k in range(1, tau + 1): #Need to predict Y_{t+1},...,Y_{t+\tau} by iteration
                A_tk = curr_treatments[:, t + k, :]
                input = torch.cat((Yp_prev, A_tk, static_features), dim=-1)
                hr_tk, hidden_states = self.lstm.single_step(input, hidden_states)
                #h_tk is the latent representation fit for the decoder's LSTM at time t + k
                assert len(hr_tk.shape) == 2, "hr_tk should be of shape (b, hidden_size)"
                Yp_prev = self.output_layer(hr_tk) # Y_{t+k}
                assert Yp_prev.shape == (prev_outputs.shape[0], 1), "Yp_prev should be of shape (b, 1)"
                predicted_outcomes.append(Yp_prev)
            prediction_t = torch.cat(predicted_outcomes, dim=1) # (b, \tau)

            result.append(prediction_t.unsqueeze(1))

        return torch.cat(result, dim=1) # (b, T - \tau, \tau)
    
    def training_step(self, batch, batch_idx):
        """
        Train the decoder network with SW weighted mse loss
        """
        pred_output = self(batch)
        curr_output = batch['curr_outputs']
        active_entries = batch['active_entries'].squeeze(-1)
        assert 'sw_dec' in batch, "SW weights are not provided in the batch."
        sw_dec = batch['sw_dec']
        b, _, tau = pred_output.shape
        T = curr_output.shape[1]
        assert sw_dec.shape == (b, T, tau)
        assert active_entries.shape == (b, T)
        
        # expand the curr-ouput to the same shape as pred_output
        target = torch.zeros_like(pred_output)
        active = torch.zeros_like(pred_output)
        for k in range(tau):
            target[:, :, k] = curr_output[:, k:T - tau + k]
            active[:, :, k] = active_entries[:, k:T - tau + k]

        weighted_mse = F.mse_loss(pred_output, target, reduction='none') * sw_dec[:, :T - tau, :]
        valid_loss = (weighted_mse * active).sum() / active.sum()
        self.log('train_loss_dec', valid_loss.item(), prog_bar=True, logger=True)
        return valid_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validate the decoder network with SW weighted mse loss
        """
        pred_output = self(batch)
        curr_output = batch['curr_outputs']
        active_entries = batch['active_entries'].squeeze(-1)
        assert 'sw_dec' in batch, "SW weights are not provided in the batch."
        sw_dec = batch['sw_dec']
        b, _, tau = pred_output.shape
        T = curr_output.shape[1]
        assert sw_dec.shape == (b, T, tau)
        assert active_entries.shape == (b, T)
        
        # expand the curr-ouput to the same shape as pred_output
        target = torch.zeros_like(pred_output)
        active = torch.zeros_like(pred_output)
        for k in range(tau):
            target[:, :, k] = curr_output[:, k:T - tau + k]
            active[:, :, k] = active_entries[:, k:T - tau + k]

        weighted_mse = F.mse_loss(pred_output, target, reduction='none') * sw_dec[:, :T - tau, :]
        valid_loss = (weighted_mse * active).sum() / active.sum()
        self.log('val_loss_dec', valid_loss.item(), prog_bar=True, logger=True)
        for k in range(tau):
            mse_k = weighted_mse[:, :, k].mean()
            self.log(f'val_loss_dec_{k}', mse_k.item(), prog_bar=True, logger=True)
        return valid_loss
    

    def predict_step(self, batch, batch_idx):
        """
        Predict the treatment probabilities
        only predicts Y_{t+\tau} for t = 1,2,...,T - \tau
        """
        pred_output = self(batch)
        # only take the last time step of the prediction
        return pred_output[:, :, -1].cpu()
    
    def predict_capo(self, data_loader, T_intv:np.ndarray) -> np.ndarray:
        """
        Predict the potential outcome under intervention T_intv
        T_intv np array of shape (tau + 1, n_treatments_disc)
        The process is achieved by replacing observed curr_treatments_disc in forward()
        by the T_intv
        Returns the predicted outcome ONLY for the last time step t + \tau
        capo_pred: ndarray of shape (N, T - tau, 1)
        """
        T_intv = torch.from_numpy(T_intv).float().to(self.device)
        capo_preds = list()
        self.eval()
        for i, batch in enumerate(data_loader):
            prev_treatments = batch['prev_treatments_disc']
            vitals = batch['curr_covariates']
            static_features = batch['static_features']
            prev_outputs = batch['prev_outputs']
            T = prev_outputs.shape[1]
            curr_treatments = T_intv[0].unsqueeze(0).unsqueeze(0).expand(
                            prev_treatments.shape[0], T, -1) # (b, T, n_treatments_disc)    
            #use encoder to encode history H_t
            x = torch.cat((prev_treatments, vitals, static_features.unsqueeze(1).expand(-1, T, -1), 
                        prev_outputs.unsqueeze(-1)), dim=-1)
            x = self.encoder.lstm(x, init_states=None) # x encodes information of H_t = (\overline{X_t}, \overline{A}_{t-1}, \overline{Y}_{t-1})
            # predict Y_t with x by encoder
            Yp = self.encoder.output_layer(torch.cat((x, curr_treatments), dim=-1))

            hr_decoder = self.memory_adapter(x)

            result = list()
            T, tau = self.max_seq_length, self.projection_horizon

            for t in range(T - tau):
                h_t = hr_decoder[:, t, :]
                Yp_prev = Yp[:, t, :] # initialize
                predicted_outcomes = list()
                hidden_states = [(h_t, h_t) for _ in range(self.num_layer)]
                for k in range(1, tau + 1): #Need to predict Y_{t+1},...,Y_{t+\tau} by iteration
                    #set treatments to T_intv[k]
                    A_tk = T_intv[k].unsqueeze(0).expand(Yp_prev.shape[0], -1)
                    input = torch.cat((Yp_prev, A_tk, static_features), dim=-1)
                    hr_tk, hidden_states = self.lstm.single_step(input, hidden_states)
                    #h_tk is the latent representation fit for the decoder's LSTM at time t + k
                    assert len(hr_tk.shape) == 2, "hr_tk should be of shape (b, hidden_size)"
                    Yp_prev = self.output_layer(hr_tk) # Y_{t+k}
                    assert Yp_prev.shape == (prev_outputs.shape[0], 1), "Yp_prev should be of shape (b, 1)"
                    predicted_outcomes.append(Yp_prev)
                prediction_t = torch.cat(predicted_outcomes, dim=1) # (b, \tau)

                result.append(prediction_t.unsqueeze(1))

            capo_pred = torch.cat(result, dim=1)[:, :, -1] # (b, T - \tau, 1)
            capo_preds.append(capo_pred.detach().cpu())
        
        return torch.cat(capo_preds, dim=0).numpy()

        
        
        


