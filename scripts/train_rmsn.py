import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_original_cwd
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from src.utils import FilteringMlFlowLogger
from src.models.rmsn import RMSN, RMSNTreatmentPropensityNetwork, RMSNHistoryPropensityNetwork, \
    RMSNEncoderNetwork, RMSNDecoderNetwork, compute_stabilized_weights
from src.utils import compute_gt_individual_dynamic_effects, sample_to_1d
import pickle
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)
OmegaConf.register_new_resolver("times", lambda x, y: x * y, replace = True)
OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace = True)
OmegaConf.register_new_resolver("sub", lambda x, y: x - y, replace = True)

@hydra.main(version_base='1.1', config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))
    seed_everything(args.exp.seed)
    #instantiate dataset pipeline
    data_pipeline = instantiate(args.dataset, _recursive_=True)
    data_pipeline.insert_necessary_args_dml_rnn(args)
    disc_dim, cont_dim = args.dataset.n_treatments_disc, args.dataset.n_treatments_cont


    experiment_name = args.exp.exp_name
    conf_strength = float(data_pipeline.get_confounding_strength())
    n_periods = args.dataset.n_periods
    train_data, val_data = data_pipeline.train_data, data_pipeline.val_data
    
    ####===========Propensity treatment / history network===================####
    mlf_logger_propensity = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name,
        tracking_uri=args.exp.mlflow_uri, run_name=f"rmsn_conf={conf_strength}_m={n_periods}_propensity")
    
    artifacts_path_propensity = hydra.utils.to_absolute_path(
        mlf_logger_propensity.experiment.get_run(
            mlf_logger_propensity.run_id).info.artifact_uri).replace('mlflow-artifacts:', 'mlruns')
    logger.info(f"Artifacts path : {artifacts_path_propensity}")

    #initialize data loader
    train_loader = DataLoader(train_data, batch_size=args.exp.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.exp.batch_size, shuffle=False)

    #initialize model
    propensity_treatment_network = RMSNTreatmentPropensityNetwork(args)
    
    #callbacks = []
    callbacks = [LearningRateMonitor(logging_interval='epoch')]

    #initialize trainer
    trainer_pt = Trainer(
                max_epochs=args.exp.max_epochs,
                callbacks=callbacks,
                devices=1,
                accelerator=args.exp.accelerator,
                deterministic=True,
                logger = mlf_logger_propensity,
                gradient_clip_val=args.exp.get('gradient_clip_val', 0.0),
                log_every_n_steps=args.exp.get('log_every_n_steps', 50),
            )
    logger.info("Train propensity treatment network")
    trainer_pt.fit(propensity_treatment_network, train_loader, val_loader)

    propensity_history_network = RMSNHistoryPropensityNetwork(args)
    traner_ph = Trainer(
                max_epochs=args.exp.max_epochs,
                callbacks=callbacks,
                devices=1,
                accelerator=args.exp.accelerator,
                deterministic=True,
                logger = mlf_logger_propensity,
                gradient_clip_val=args.exp.get('gradient_clip_val', 0.0),
                log_every_n_steps=args.exp.get('log_every_n_steps', 50),
            )
    logger.info("Train propensity history network")
    traner_ph.fit(propensity_history_network, train_loader, val_loader)

    
    ####===============Compute stablized weights===================####
    logger.info("Compute stablized weights")
    sw_train = compute_stabilized_weights(train_loader, propensity_treatment_network, propensity_history_network)
    sw_val = compute_stabilized_weights(val_loader, propensity_treatment_network, propensity_history_network)
    random_sample_num = 20
    #sample some values from sw_train/ sw_val into a 1-D tensor and print the value
    random_sw_train = sample_to_1d(sw_train, random_sample_num)
    random_sw_val = sample_to_1d(sw_val, random_sample_num)
    logger.info(f"random sample from sw_train:\n {random_sw_train}")
    logger.info(f"random sample from sw_val:\n {random_sw_val}")

    train_data.add_sw_enc_dec(sw_train)
    val_data.add_sw_enc_dec(sw_val)
    logger.info("Add stablized weights to train and val data")

    train_loader = DataLoader(train_data, batch_size=args.exp.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.exp.batch_size, shuffle=True)

    #####===========Encoder / Decoder network===================####
    mlf_logger_enc_dec = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name,
        tracking_uri=args.exp.mlflow_uri, run_name=f"rmsn_conf={conf_strength}_m={n_periods}_enc_dec")
    
    #initialize encoder
    encoder_network = RMSNEncoderNetwork(args)

    trainer_enc = Trainer(
                max_epochs=args.exp.max_epochs,
                callbacks=callbacks,
                devices=1,
                accelerator=args.exp.accelerator,
                deterministic=True,
                logger = mlf_logger_enc_dec,
                gradient_clip_val=args.exp.get('gradient_clip_val', 0.0),
                log_every_n_steps=args.exp.get('log_every_n_steps', 50),
            )
    logger.info("Train encoder network")
    trainer_enc.fit(encoder_network, train_loader, val_loader)

    decoder_network = RMSNDecoderNetwork(args, encoder_network)

    trainer_dec = Trainer(
                max_epochs=args.exp.max_epochs,
                callbacks=callbacks,
                devices=1,
                accelerator=args.exp.accelerator,
                deterministic=True,
                logger = mlf_logger_enc_dec,
                gradient_clip_val=args.exp.get('gradient_clip_val', 0.0),
                log_every_n_steps=args.exp.get('log_every_n_steps', 50),
            )
    logger.info("Train decoder network")
    trainer_dec.fit(decoder_network, train_loader, val_loader)

    ####===========Compute treatment effect===================####

    logger.info("Compute treatment effect")
    T_intv_disc, T_base_disc = (np.ones((args.dataset.n_periods, disc_dim)), np.zeros((args.dataset.n_periods, disc_dim))) \
                                    if disc_dim > 0 else (None, None)
    T_intv_cont, T_base_cont = (np.ones((args.dataset.n_periods, cont_dim)), np.zeros((args.dataset.n_periods, cont_dim))) \
                                    if cont_dim > 0 else (None, None)
    logger.info(f"Interved treatment (discrete and continuous): \n {T_intv_disc} \n {T_intv_cont}")
    logger.info(f"Baseline treatment (discrete and continuous): \n {T_base_disc} \n {T_base_cont}")
    gt_te = data_pipeline.compute_treatment_effect('test', T_intv_disc, T_intv_cont, T_base_disc, T_base_cont)
    testloader = DataLoader(data_pipeline.test_data, batch_size=args.exp.batch_size, shuffle=False)

    capo_intv = decoder_network.predict_capo(testloader, T_intv_disc)
    capo_base = decoder_network.predict_capo(testloader, T_base_disc)
    pred_te = capo_intv - capo_base
    mask =(~np.isnan(pred_te)) & (~np.isnan(gt_te))
    te_rmse = np.sqrt(np.mean((gt_te[mask] - pred_te[mask])**2))
    mlf_logger_enc_dec.log_metrics({'TE_rmse': te_rmse})
    logger.info(f"Test RMSE: {te_rmse}")




if __name__ == '__main__':
    main()

