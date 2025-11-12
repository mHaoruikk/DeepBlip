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
from src.models.basic_blocks import OutcomeHead_GNET

logger = logging.getLogger(__name__)

class GNet(LightningModule):
    """
    pytorch lightning implementation for G-Net.
    """

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
        self.treatment_seq = torch.tensor(args.model.treatment_sequence, dtype=torch.float32).to(self.device)

        assert self.n_treatments_cont == 0, "RMSN does not support continuous treatments yet."
        self.save_hyperparameters(args)

        self._initialize_model(args)
        # Initialize lists for residuals, will be populated only during the last validation epoch
        self.validation_outcome_residuals = []
        self.validation_vitals_residuals = []
        self.validation_active_entries_vitals = [] # Added for vitals masking

    def _initialize_model(self, args:DictConfig):

        self.max_seq_length = self.sequence_length
        self.hr_size = args.model.hr_size
        self.seq_hidden_units = args.model.hidden_size
        self.fc_hidden_units = args.model.fc_hidden_size
        self.dropout_rate = args.model.dropout_rate
        self.num_layer = args.model.num_layer

        self.lstm = VariationalLSTM(self.input_size, self.seq_hidden_units, self.num_layer, self.dropout_rate)
        self.hr_output_transformation = nn.Linear(self.seq_hidden_units, self.hr_size)
        
        self.vital_heads = nn.ModuleList()
        for i in range(self.dim_vitals):
            input_size = i + self.hr_size
            self.vital_heads.append(
                OutcomeHead_GNET(input_size, self.fc_hidden_units, 1)
            )
        self.outcome_head = OutcomeHead_GNET(self.hr_size + self.dim_treatments, 
                                             self.fc_hidden_units, self.dim_outcome)
        
    
    def build_hr(self, prev_treatments, vitals, prev_outputs, static_features, active_entries):

        x = torch.cat((prev_treatments, prev_outputs.unsqueeze(-1)), dim=-1)
        x = torch.cat((x, vitals), dim=-1) if self.has_vitals else x
        x = torch.cat((x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        x = self.lstm(x, init_states=None)

        #output = self.output_dropout(x)
        hr = nn.ELU()(self.hr_output_transformation(x))
        return hr

    def forward(self, batch):
        """
        The forward pass predicts the next step vitals and outcome based on the current state
        Note that the covariates are predicted in the iterative manner:
        X_{t+1}^{j} = f_j(R_t, X_{t}^{0,1,..,j-1}),   0\leq j \leq p-2 p is dim_vitals
        The outcome Y_t is predicted from R_t and the intervention a^*_t
        Y_{t} = h(R_t, a^*_t)
        where R_t is the hidden representation of the patient state at time t
        Returns:
            pred_outcome: the predicted outcome at time t, shape (b, L, 1)
            vitals_pred: the predicted vitals at time t, shape (b, L, n_vitals)
        """

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

        #hr is R_t
        hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, active_entries)

        vitals_pred = []
        prev_vitals = np.zeros((b, L, 0), device = self.device, requires_grad = True)
        for i in range(self.dim_vitals):
            vital = self.vital_heads[i].build_outcome(torch.cat((hr, prev_vitals), dim = -1))
            vitals_pred.append(vital)
            prev_vitals = torch.cat((prev_vitals, vital), dim = -1)

        pred_outcome = self.outcome_head.build_outcome(torch.cat((hr, curr_treatments), dim = -1))

        return pred_outcome, torch.cat(vitals_pred, dim = -1)
    
    def training_step(self, batch, batch_idx):
        """
        Train G-Net through minimizing the mse error between the predicted and true outcome / vitals
        """
        pred_outcome, vitals_pred = self(batch)
        curr_outputs = batch['curr_outputs']
        next_vitals = batch['curr_covariates'][:, 1:, :]
        active_entries = batch['active_entries']
        active_entries_vitals = active_entries[:, 1:, :] # Match dimensions for vitals loss

        outcome_mse = F.mse_loss(pred_outcome, curr_outputs, reduction = 'none')
        outcome_mse = (outcome_mse * active_entries).sum() / active_entries.sum()
        vitals_mse = F.mse_loss(vitals_pred[:, :-1, :], next_vitals, reduction = 'none')
        # Ensure active_entries_vitals has the same shape as vitals_mse for broadcasting
        vitals_mse = (vitals_mse * active_entries_vitals).sum() / active_entries_vitals.sum()

        loss = outcome_mse + vitals_mse
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_y_mse', outcome_mse, prog_bar=False, logger=True)
        self.log('train_vitals_mse', vitals_mse, prog_bar=False, logger=True)
        return loss

    def on_validation_epoch_start(self):
        # Clear residuals lists at the start of the *last* validation epoch
        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            self.validation_outcome_residuals = []
            self.validation_vitals_residuals = []
            self.validation_active_entries_vitals = [] # Clear vitals active entries list

    def validation_step(self, batch, batch_idx):
        """
        Validate G-Net through minimizing the mse error between the predicted and true outcome / vitals
        """
        pred_outcome, vitals_pred = self(batch)
        curr_outputs = batch['curr_outputs']
        next_vitals = batch['curr_covariates'][:, 1:, :]
        active_entries = batch['active_entries']
        active_entries_vitals = active_entries[:, 1:, :] # Match dimensions for vitals loss

        outcome_mse = F.mse_loss(pred_outcome, curr_outputs, reduction = 'none')
        outcome_mse = (outcome_mse * active_entries).sum() / active_entries.sum()
        vitals_mse = F.mse_loss(vitals_pred[:, :-1, :], next_vitals, reduction = 'none')
        # Ensure active_entries_vitals has the same shape as vitals_mse for broadcasting
        vitals_mse = (vitals_mse * active_entries_vitals.unsqueeze(-1).expand_as(vitals_mse)).sum() / active_entries_vitals.sum() # Apply mask correctly

        loss = outcome_mse + vitals_mse
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_y_mse', outcome_mse, prog_bar=True, logger=True)
        self.log('val_vitals_mse', vitals_mse, prog_bar=True, logger=True)

        # Store residuals only on the last epoch
        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            # Calculate outcome residuals and filter for active entries before appending
            outcome_res = pred_outcome - curr_outputs
            outcome_res_active = outcome_res[active_entries == 1]
            self.validation_outcome_residuals.append(outcome_res_active.detach().cpu())

            # Calculate and store all vitals residuals and corresponding active entries mask
            vitals_res = vitals_pred[:, :-1, :] - next_vitals
            self.validation_vitals_residuals.append(vitals_res.detach().cpu())
            self.validation_active_entries_vitals.append(active_entries_vitals.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        # Compute and log Gaussian parameters for residuals at the end of the *last* validation epoch
        logger.info("Validation epoch ended. Logging residuals.")
        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            if self.validation_outcome_residuals:
                all_outcome_residuals = torch.cat(self.validation_outcome_residuals)
                outcome_res_mean = torch.mean(all_outcome_residuals)
                outcome_res_std = torch.std(all_outcome_residuals)
                self.log('val_outcome_residual_mean', outcome_res_mean, logger=True)
                self.log('val_outcome_residual_std', outcome_res_std, logger=True)

            # --- Vitals Residuals ---
            if self.validaon_vitals_residuals and self.validation_active_entries_vitals:
                all_vitals_res = torch.cat(self.validation_vitals_residuals, dim=0)
                all_active_entries = torch.cat(self.validation_active_entries_vitals, dim=0)

                all_vitals_res_flat = all_vitals_res.view(-1, self.dim_vitals)
                active_mask_flat = all_active_entries.view(-1).bool()

                filtered_vitals_residuals = all_vitals_res_flat[active_mask_flat]

                vitals_res_std_per_dim = torch.std(filtered_vitals_residuals, dim=0)

                for i in range(self.dim_vitals):
                    self.log(f'val_vitals_residual_std_dim_{i}', vitals_res_std_per_dim[i], logger=True)

                self.validation_vitals_residuals = []
                self.validation_active_entries_vitals = []
        

    def MC_simulate(self, batch, T_intv:torch.tensor):
        """
        Conduct Monte Carlo simulation to predict the outcome of the patient under the intervention.
        The simulation is done in rolling origin manner, conditioning on history H_t for 0 <= t < T - tau,
        and predicts the tau-step ahead potential outcome Y_{t+tau} under intervention T_intv.
        Noise is added based on validation set residual standard deviations.

        Args:
            batch: Input batch dictionary containing observed history.
            T_intv (tau + 1, n_treat).

        Returns:
            final_potential_outcomes: Tensor of shape (b, L - tau) containing the
                                      average potential outcome Y_{t+tau} for each starting time t.
        """
        self.eval() # Set model to evaluation mode

        tau = self.projection_horizon
        n_mc_samples = self.args.model.get('mc_samples', 100) # Get MC samples from config, default 100
        b = batch['prev_outputs'].size(0)
        L = batch['prev_outputs'].size(1) # Sequence length T

        if self.outcome_res_std is None or self.vitals_res_std_per_dim is None:
            raise ValueError("Residual standard deviations not computed. Ensure validation ran on the last epoch.")

        # Move std devs to the correct device
        outcome_std = self.outcome_res_std.to(self.device)
        vitals_std = self.vitals_res_std_per_dim.to(self.device)

        # Extract observed data
        static_features = batch['static_features'] if self.n_static > 0 else torch.zeros((b, 0), device=self.device)
        prev_treatments_disc = batch['prev_treatments_disc'] if self.n_treatments_disc > 0 else torch.zeros((b, L, 0), device=self.device)
        prev_treatments_cont = batch['prev_treatments_cont'] if self.n_treatments_cont > 0 else torch.zeros((b, L, 0), device=self.device)
        prev_treatments = torch.cat([prev_treatments_disc, prev_treatments_cont], dim = -1) # a_{t-1}
        prev_outputs = batch['prev_outputs'] # Y_{t-1}
        curr_covariates = batch['curr_covariates'] # X_t

        # Combine future intervention sequences
        T_intv = T_intv.unsqueeze(0).expand(b, -1, -1)

        # Initialize storage for final results
        final_potential_outcomes = torch.zeros((b, L - tau), device=self.device)

        # --- Loop over starting time points t_start ---
        for t_start in range(L - tau):
            mc_outcomes_at_t_tau = torch.zeros((b, n_mc_samples), device=self.device)

            for mc_idx in range(n_mc_samples):

                # --- Initialize LSTM state at t_start by running on observed history ---
                # Initial hidden states (h_0, c_0) for all layers are zeros
                hx_list = [torch.zeros((b, self.seq_hidden_units), device=self.device) for _ in range(self.num_layer)]
                cx_list = [torch.zeros((b, self.seq_hidden_units), device=self.device) for _ in range(self.num_layer)]

                for k_init in range(t_start):
                    # Input: a_{k-1}, Y_{k-1}, X_k, s
                    input_k = torch.cat((prev_treatments[:, k_init, :],
                                         prev_outputs[:, k_init].unsqueeze(-1),
                                         curr_covariates[:, k_init, :],
                                         static_features), dim=-1)

                    new_hx_list, new_cx_list = [], []
                    layer_input = input_k
                    for layer_idx, cell in enumerate(self.lstm.lstm_layers):
                        hx, cx = cell(layer_input, (hx_list[layer_idx], cx_list[layer_idx]))
                        # No dropout in eval mode
                        new_hx_list.append(hx)
                        new_cx_list.append(cx)
                        layer_input = hx # Output of layer becomes input to next
                    hx_list, cx_list = new_hx_list, new_cx_list
                # hx_list, cx_list now contain hidden states after processing step t_start - 1
                # hx_list[-1] is the representation R_{t_start - 1}
                # --- Initialize Simulation State ---
                sim_hx_list, sim_cx_list = hx_list, cx_list
                sim_vitals_k = curr_covariates[:, t_start, :]      # X_{t_start}
                sim_outcome_k_minus_1 = prev_outputs[:, t_start]   # Y_{t_start - 1}

                # --- Simulation Loop: k from 0 to tau ---
                for k in range(tau + 1):
                    t_current = t_start + k

                    #  Calculate hr_{t_current} 
                    # LSTM Input: a_{t-1}, Y_{t-1}, X_t, s
                    # Use observed prev_treatment at t_current (a_{t_current-1})
                    lstm_input_k = torch.cat((prev_treatments[:, t_current, :],
                                              sim_outcome_k_minus_1.unsqueeze(-1), # Y_{t_current - 1}
                                              sim_vitals_k,                        # X_{t_current}
                                              static_features), dim=-1)

                    new_sim_hx_list, new_sim_cx_list = [], []
                    layer_input = lstm_input_k
                    for layer_idx, cell in enumerate(self.lstm.lstm_layers):
                        hx, cx = cell(layer_input, (sim_hx_list[layer_idx], sim_cx_list[layer_idx]))
                        new_sim_hx_list.append(hx)
                        new_sim_cx_list.append(cx)
                        layer_input = hx
                    sim_hx_list, sim_cx_list = new_sim_hx_list, new_sim_cx_list # State after processing t_current

                    # Calculate representation R_t
                    hr_k = nn.ELU()(self.hr_output_transformation(sim_hx_list[-1]))

                    # Predict & Sample Outcome Y_{t_current} 
                    intervention_k = T_intv[:, k, :] # Use intervention a*_{t_current}
                    outcome_head_input = torch.cat((hr_k, intervention_k), dim=-1)
                    outcome_mean_k = self.outcome_head.build_outcome(outcome_head_input)
                    outcome_noise = torch.randn_like(outcome_mean_k) * outcome_std
                    sim_outcome_k = outcome_mean_k + outcome_noise # Y_{t_current}

                    # Store Final Outcome 
                    if k == tau:
                        mc_outcomes_at_t_tau[:, mc_idx] = sim_outcome_k.squeeze(-1) # Store Y_{t_start + tau}
                        break # Exit simulation loop for this MC sample

                    # Predict & Sample Vitals X_{t_current+1} 
                    # Prediction uses hr_k (R_{t_current}) and sim_vitals_k (X_{t_current})
                    pred_vitals_means_list = []
                    for vital_idx in range(self.dim_vitals):
                        # Input to head j: R_t, X_t^0, ..., X_t^{j-1}
                        vital_head_input = torch.cat((hr_k, sim_vitals_k[:, :vital_idx]), dim=-1)
                        vital_mean = self.vital_heads[vital_idx].build_outcome(vital_head_input)
                        pred_vitals_means_list.append(vital_mean)

                    vitals_mean_k_plus_1 = torch.cat(pred_vitals_means_list, dim=-1) # Mean X_{t_current+1}
                    vitals_noise = torch.randn_like(vitals_mean_k_plus_1) * vitals_std # Use per-dimension std
                    sim_vitals_k_plus_1 = vitals_mean_k_plus_1 + vitals_noise # X_{t_current+1}

                    # Update State for Next Iteration
                    sim_vitals_k = sim_vitals_k_plus_1             
                    sim_outcome_k_minus_1 = sim_outcome_k.squeeze(-1)

            final_potential_outcomes[:, t_start] = torch.mean(mc_outcomes_at_t_tau, dim=1)

        return final_potential_outcomes
    
    def predict_capo(self, testloader, T_intv_disc:np.ndarray, T_intv_cont:np.ndarray):
        """
        Predict the potential outcome under the intervention T_intv using the monte carlo simulation.
        Args:
            testloader: DataLoader for the test set.
            T_intv_disc: Discrete treatment intervention sequence.
            T_intv_cont: Continuous treatment intervention sequence.

        Returns:
            pred_capo: Predicted potential outcomes under the intervention. shape (N, T - tau)
        """
        self.eval()
        pred_capos = []
        T_intv = torch.from_numpy(self._combine_disc_cont(T_intv_disc, T_intv_cont)).float().to(self.device)
        for i, batch in enumerate(testloader):
            capos = self.MC_simulate(batch, T_intv)
            pred_capos.append(capos.cpu().detach().numpy())
        return np.concatenate(pred_capos, axis=0)


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


