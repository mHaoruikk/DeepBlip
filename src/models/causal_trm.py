import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from torch_ema import ExponentialMovingAverage
import numpy as np
from omegaconf import DictConfig
import logging
import mlflow
from typing import List, Tuple

from src.models.utils_transformer import AbsolutePositionalEncoding, RelativePositionalEncoding, TransformerMultiInputBlock
from src.models.basic_blocks import BRTreatmentOutcomeHead
from src.models.utils import bce

logger = logging.getLogger(__name__)


class Causal_transformer(LightningModule):
    """
    Pytorch Lightning Module for the Causal transformer
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
        self.alpha = args.model.alpha
        self.update_alpha = args.model.update_alpha
        self.balancing = args.model.balancing

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

        self.br_treatment_outcome_head = BRTreatmentOutcomeHead(self.seq_hidden_units, self.br_size,
                                                                    self.fc_hidden_units, self.dim_treatments, self.dim_outcome,
                                                                    self.alpha, self.update_alpha, balancing = 'domain_confusion')
        
    def build_br(self, prev_treatments, vitals, prev_outputs, static_features, active_entries):

        active_entries_treat_outcomes = torch.clone(active_entries)
        active_entries_vitals = torch.clone(active_entries)

        x_t = self.treatments_input_transformation(prev_treatments)
        x_o = self.output_input_transformation(prev_outputs)
        x_v = self.vitals_input_transformation(vitals) if self.has_vitals else None
        x_s = self.static_input_transformation(static_features.unsqueeze(1))  # .expand(-1, x_t.size(1), -1)

        # if active_encoder_br is None and encoder_r is None:  # Only self-attention
        for block in self.transformer_blocks:

            if self.self_positional_encoding is not None:
                x_t = x_t + self.self_positional_encoding(x_t)
                x_o = x_o + self.self_positional_encoding(x_o)
                x_v = x_v + self.self_positional_encoding(x_v) if self.has_vitals else None

            if self.has_vitals:
                x_t, x_o, x_v = block((x_t, x_o, x_v), x_s, active_entries_treat_outcomes, active_entries_vitals)
            else:
                x_t, x_o = block((x_t, x_o), x_s, active_entries_treat_outcomes)

        if not self.has_vitals:
            x = (x_o + x_t) / 2
        else:
            x = (x_o + x_t + x_v) / 3

        output = self.output_dropout(x)
        br = self.br_treatment_outcome_head.build_br(output)
        return br
    
    def forward(self, batch, detach_treatment=False):
        
        prev_outputs = batch['prev_outputs'].unsqueeze(-1)
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

        br = self.build_br(prev_treatments, vitals, prev_outputs, static_features, active_entries)
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, curr_treatments)

        return treatment_pred, outcome_pred, br
    
    def training_step(self, batch, batch_ind, optimizer_idx=0):
        for par in self.parameters():
            par.requires_grad = True

        if optimizer_idx == 0 or self.br_treatment_outcome_head.alpha==0.0:  # grad reversal or domain confusion representation update
            if self.args.model.weights_ema:
                with self.ema_treatment.average_parameters():
                    treatment_pred, outcome_pred, _ = self(batch)
            else:
                treatment_pred, outcome_pred, _ = self(batch)

            mse_loss = F.mse_loss(outcome_pred, batch['curr_outputs'], reduce=False)
            if self.balancing == 'grad_reverse':
                bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
            elif self.balancing == 'domain_confusion':
                bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='confuse')
                bce_loss = self.br_treatment_outcome_head.alpha * bce_loss
            else:
                raise NotImplementedError()

            # Masking for shorter sequences
            # Attention! Averaging across all the active entries (= sequence masks) for full batch
            bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
            mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()

            loss = bce_loss + mse_loss

            self.log(f'{self.model_type}_train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_alpha', self.br_treatment_outcome_head.alpha, on_epoch=True, on_step=False,
                     sync_dist=True)

            return loss

        elif optimizer_idx == 1:  # domain classifier update
            if self.hparams.exp.weights_ema:
                with self.ema_non_treatment.average_parameters():
                    treatment_pred, _, _ = self(batch, detach_treatment=True)
            else:
                treatment_pred, _, _ = self(batch, detach_treatment=True)

            bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
            if self.balancing == 'domain_confusion':
                bce_loss = self.br_treatment_outcome_head.alpha * bce_loss

            # Masking for shorter sequences
            # Attention! Averaging across all the active entries (= sequence masks) for full batch
            bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
            self.log(f'{self.model_type}_train_bce_loss_cl', bce_loss, on_epoch=True, on_step=False, sync_dist=True)

            return bce_loss
        

    def bce_loss(self, treatment_pred, current_treatments, kind='predict'):
        mode = self.hparams.dataset.treatment_mode
        bce_weights = torch.tensor(self.bce_weights).type_as(current_treatments) if self.hparams.exp.bce_weight else None

        if kind == 'predict':
            bce_loss = bce(treatment_pred, current_treatments, mode, bce_weights)
        elif kind == 'confuse':
            uniform_treatments = torch.ones_like(current_treatments)
            if mode == 'multiclass':
                uniform_treatments *= 1 / current_treatments.shape[-1]
            elif mode == 'multilabel':
                uniform_treatments *= 0.5
            bce_loss = bce(treatment_pred, uniform_treatments, mode)
        else:
            raise NotImplementedError()
        return bce_loss
    
    def _calculate_bce_weights(self) -> None:
        if self.hparams.dataset.treatment_mode == 'multiclass':
            current_treatments = self.dataset_collection.train_f.data['current_treatments']
            current_treatments = current_treatments.reshape(-1, current_treatments.shape[-1])
            current_treatments = current_treatments[self.dataset_collection.train_f.data['active_entries'].flatten().astype(bool)]
            current_treatments = np.argmax(current_treatments, axis=1)

            self.bce_weights = len(current_treatments) / np.bincount(current_treatments) / len(np.bincount(current_treatments))
        else:
            raise NotImplementedError()

    def on_fit_start(self) -> None:  # Issue with logging not yet existing parameters in MlFlow
        if self.trainer.logger is not None:
            self.trainer.logger.filter_submodels = list(self.possible_model_types - {self.model_type})

    def on_fit_end(self) -> None:  # Issue with logging not yet existing parameters in MlFlow
        if self.trainer.logger is not None:
            self.trainer.logger.filter_submodels = list(self.possible_model_types)
    

    def configure_optimizers(self):
        if self.balancing == 'grad_reverse' and not self.args.model.weights_ema:  # one optimizer
            optimizer = self._get_optimizer(list(self.named_parameters()))

            if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
                return self._get_lr_schedulers(optimizer)

            return optimizer

        else:  # two optimizers - simultaneous gradient descent update
            treatment_head_params = \
                ['br_treatment_outcome_head.' + s for s in self.br_treatment_outcome_head.treatment_head_params]
            treatment_head_params = \
                [k for k in dict(self.named_parameters()) for param in treatment_head_params if k.startswith(param)]
            non_treatment_head_params = [k for k in dict(self.named_parameters()) if k not in treatment_head_params]

            assert len(treatment_head_params + non_treatment_head_params) == len(list(self.named_parameters()))

            treatment_head_params = [(k, v) for k, v in dict(self.named_parameters()).items() if k in treatment_head_params]
            non_treatment_head_params = [(k, v) for k, v in dict(self.named_parameters()).items()
                                         if k in non_treatment_head_params]

            if self.args.weights_ema:
                self.ema_treatment = ExponentialMovingAverage([par[1] for par in treatment_head_params],
                                                              decay=self.hparams.exp.beta)
                self.ema_non_treatment = ExponentialMovingAverage([par[1] for par in non_treatment_head_params],
                                                                  decay=self.hparams.exp.beta)

            treatment_head_optimizer = self._get_optimizer(treatment_head_params)
            non_treatment_head_optimizer = self._get_optimizer(non_treatment_head_params)

            if self.args.model['optimizer']['lr_scheduler']:
                return self._get_lr_schedulers([non_treatment_head_optimizer, treatment_head_optimizer])

            return [non_treatment_head_optimizer, treatment_head_optimizer]
    
    def _get_optimizer(self, param_optimizer: list):
        no_decay = ['bias', 'layer_norm']
        sub_args = self.arg.model
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': sub_args['optimizer']['weight_decay'],
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        lr = sub_args['optimizer']['learning_rate']
        optimizer_cls = sub_args['optimizer']['optimizer_cls']
        if optimizer_cls.lower() == 'adamw':
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
        elif optimizer_cls.lower() == 'adam':
            optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optimizer_cls.lower() == 'sgd':
            optimizer = optim.SGD(optimizer_grouped_parameters, lr=lr,
                                  momentum=sub_args['optimizer']['momentum'])
        else:
            raise NotImplementedError()

        return optimizer
    
    def _get_lr_schedulers(self, optimizer):
        if not isinstance(optimizer, list):
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            return [optimizer], [lr_scheduler]
        else:
            lr_schedulers = []
            for opt in optimizer:
                lr_schedulers.append(optim.lr_scheduler.ExponentialLR(opt, gamma=0.99))
            return optimizer, lr_schedulers
    

        