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
from src.models.ipw_rnn import PropensityNetwork, InversePropensityWeightedNetwork
from src.models.utils import combine_disc_cont
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
    sequence_length, n_periods = args.dataset.sequence_length, args.dataset.n_periods

    logger.info('#=================1 stage: Propensity Network=================')
    if args.exp.logging:
        experiment_name = args.exp.exp_name
        conf_strength = float(args.dataset.synth_treatments_list[0]['conf_outcome_weight'])
        #conf_strength = 1.0
        n_periods = args.dataset.n_periods
        mlf_logger_prop = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name,
            tracking_uri=args.exp.mlflow_uri, run_name=f"prop_conf={conf_strength}_m={n_periods}")
        
        artifacts_path = hydra.utils.to_absolute_path(
            mlf_logger_prop.experiment.get_run(
                mlf_logger_prop.run_id).info.artifact_uri).replace('mlflow-artifacts:', 'mlruns')
        logger.info(f"Artifacts path : {artifacts_path}")
    else:
        mlf_logger_prop = None
        artifacts_path = None

    #initialize data loader
    train_data, val_data = data_pipeline.train_data, data_pipeline.val_data
    train_loader = DataLoader(train_data, batch_size=args.exp.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.exp.batch_size, shuffle=False)

    #initialize model
    prop_rnn = PropensityNetwork(args)
    
    callbacks = []
    if args.checkpoint.save:
        checkpoint_callback = ModelCheckpoint(dirpath = artifacts_path,           
            filename = "prop-{epoch}-{val_loss:.4f}",
            monitor = 'val_prop_loss', mode = "min", save_top_k = args.checkpoint.top_k, verbose = True                      
        )
        callbacks.append(checkpoint_callback)
    callbacks += [LearningRateMonitor(logging_interval='epoch')]

    #initialize trainer
    trainer = Trainer(
                max_epochs=args.exp.max_epochs_prop,
                callbacks=callbacks,
                devices=1,
                accelerator=args.exp.accelerator,
                deterministic=True,
                logger = mlf_logger_prop,
                gradient_clip_val=args.exp.get('gradient_clip_val', 0.0),
                log_every_n_steps=args.exp.get('log_every_n_steps', 50),
            )
    
    trainer.fit(prop_rnn, train_loader, val_loader)
    logger.info("Propensity network training completed")

    logger.info('#=================2 stage: Inverse Propensity Weighted Network=================')
    logger.info("Evaluate individual treatment effect")
    T_intv_disc, T_base_disc = (np.ones((args.dataset.n_periods, disc_dim)), np.zeros((args.dataset.n_periods, disc_dim))) \
                                    if disc_dim > 0 else (None, None)
    T_intv_cont, T_base_cont = (np.ones((args.dataset.n_periods, cont_dim)), np.zeros((args.dataset.n_periods, cont_dim))) \
                                    if cont_dim > 0 else (None, None)
    names = ['intv', 'base']
    capo_pred_dict = {}
    for i in range(2): # 0 for intv and 1 for base
        mlf_logger_ipw = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name,
            tracking_uri=args.exp.mlflow_uri, run_name=f"ipw_{names[i]}_conf={conf_strength}_m={n_periods}")
        artifacts_path = hydra.utils.to_absolute_path(
            mlf_logger_ipw.experiment.get_run(
                mlf_logger_ipw.run_id).info.artifact_uri).replace('mlflow-artifacts:', 'mlruns')
        logger.info(f"Artifacts path : {artifacts_path}")
        if i == 0:
            treatment_seq = combine_disc_cont(T_intv_disc, T_intv_cont)
        else:
            treatment_seq = combine_disc_cont(T_base_disc, T_base_cont)
        train_loader = DataLoader(train_data, batch_size=args.exp.batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=args.exp.batch_size, shuffle=False)

        ipw_rnn = InversePropensityWeightedNetwork(args, prop_rnn, treatment_seq)
        callbacks = [LearningRateMonitor(logging_interval='epoch')]
        if args.checkpoint.save:
            checkpoint_callback = ModelCheckpoint(dirpath = artifacts_path,           
                filename = "ipw-{epoch}-{val_loss:.4f}",
                monitor = 'val_po_loss', mode = "min", save_top_k = args.checkpoint.top_k, verbose = True                      
            )
            callbacks.append(checkpoint_callback)
        trainer = Trainer(
                    max_epochs=args.exp.max_epochs_ipw,
                    callbacks=callbacks,
                    devices=1,
                    accelerator=args.exp.accelerator,
                    deterministic=True,
                    logger = mlf_logger_ipw,
                    gradient_clip_val=args.exp.get('gradient_clip_val', 0.0),
                    log_every_n_steps=args.exp.get('log_every_n_steps', 50),
                )
        trainer.fit(ipw_rnn, train_loader, val_loader)
        logger.info(f"{names[i]} treatment ipw learner completed")

        testloader = DataLoader(data_pipeline.test_data, batch_size=args.exp.batch_size, shuffle=False)
        predictions = trainer.predict(ipw_rnn, testloader)
        capo_pred_dict[names[i]] = torch.cat(predictions, dim = 0).squeeze(-1).detach().cpu().numpy()[:, :sequence_length - n_periods + 1]

    logger.info("Evaluate individual treatment effect")

    logger.info(f"Interved treatment (discrete and continuous): \n {T_intv_disc} \n {T_intv_cont}")
    logger.info(f"Baseline treatment (discrete and continuous): \n {T_base_disc} \n {T_base_cont}")
    gt_te = data_pipeline.compute_treatment_effect('test', T_intv_disc, T_intv_cont, T_base_disc, T_base_cont)
    capo_intv = capo_pred_dict['intv']
    capo_base = capo_pred_dict['base']
    pred_te = capo_intv - capo_base
    mask =(~np.isnan(pred_te)) & (~np.isnan(gt_te))
    te_mse = np.mean((gt_te[mask] - pred_te[mask])**2)
    mlf_logger_ipw.log_metrics({'TE_mse': te_mse})
    logger.info(f"Test MSE: {te_mse}")


if __name__ == '__main__':
    main()
