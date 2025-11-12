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
from src.models.ha_rnn import PluginHistoryAdjustedNetwork
from src.utils import compute_gt_individual_dynamic_effects
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

    if args.exp.logging:
        experiment_name = args.exp.exp_name
        conf_strength = float(data_pipeline.get_confounding_strength())
        n_periods = args.dataset.n_periods
        mlf_logger = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name,
            tracking_uri=args.exp.mlflow_uri, run_name=f"harnn_conf={conf_strength}_m={n_periods}")
        
        artifacts_path = hydra.utils.to_absolute_path(
            mlf_logger.experiment.get_run(
                mlf_logger.run_id).info.artifact_uri).replace('mlflow-artifacts:', 'mlruns')
        logger.info(f"Artifacts path : {artifacts_path}")
    else:
        mlf_logger = None
        artifacts_path = None

    #initialize data loader
    train_data, val_data = data_pipeline.train_data, data_pipeline.val_data
    train_loader = DataLoader(train_data, batch_size=args.exp.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.exp.batch_size, shuffle=False)

    #initialize model
    if args.exp.load_pretrained:
        run_id = args.exp.run_id
        run_dir = os.path.join(hydra.utils.to_absolute_path(f"mlruns/{args.exp.exp_id}"), run_id)
        artifacts_dir = os.path.join(run_dir, 'artifacts')
        ckpts = [f for f in os.listdir(artifacts_dir) if f.endswith('ckpt')]
        if len(ckpts) == 0:
            raise FileNotFoundError(f"No checkpoint found in {artifacts_dir}")
        checkpoint_path = os.path.join(artifacts_dir, ckpts[0])
        ha_rnn = PluginHistoryAdjustedNetwork.load_from_checkpoint(checkpoint_path=checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        ha_rnn = PluginHistoryAdjustedNetwork(args)
    
    callbacks = []
    if args.checkpoint.save:
        checkpoint_callback = ModelCheckpoint(dirpath = artifacts_path,           
            filename = "harnn-{epoch}-{val_loss:.4f}",
            monitor = args.checkpoint.monitor, mode = "min", save_top_k = args.checkpoint.top_k, verbose = True                      
        )
        callbacks.append(checkpoint_callback)
    callbacks += [LearningRateMonitor(logging_interval='epoch')]

    #initialize trainer
    trainer = Trainer(
                max_epochs=args.exp.max_epochs,
                callbacks=callbacks,
                devices=1,
                accelerator=args.exp.accelerator,
                deterministic=True,
                logger = mlf_logger,
                gradient_clip_val=args.exp.get('gradient_clip_val', 0.0),
                log_every_n_steps=args.exp.get('log_every_n_steps', 50),
            )
    if (args.exp.resume_train) or (args.exp.load_pretrained == False):
        trainer.fit(ha_rnn, train_loader, val_loader)

    logger.info("Evaluate individual treatment effect")
    T_intv_disc, T_base_disc = (np.ones((args.dataset.n_periods, disc_dim)), np.zeros((args.dataset.n_periods, disc_dim))) \
                                    if disc_dim > 0 else (None, None)
    T_intv_cont, T_base_cont = (np.ones((args.dataset.n_periods, cont_dim)), np.zeros((args.dataset.n_periods, cont_dim))) \
                                    if cont_dim > 0 else (None, None)
    logger.info(f"Interved treatment (discrete and continuous): \n {T_intv_disc} \n {T_intv_cont}")
    logger.info(f"Baseline treatment (discrete and continuous): \n {T_base_disc} \n {T_base_cont}")
    gt_te = data_pipeline.compute_treatment_effect('test', T_intv_disc, T_intv_cont, T_base_disc, T_base_cont)
    testloader = DataLoader(data_pipeline.test_data, batch_size=args.exp.batch_size, shuffle=False)
    capo_intv = ha_rnn.predict_capo(testloader, T_intv_disc, T_intv_cont)
    capo_base = ha_rnn.predict_capo(testloader, T_base_disc, T_base_cont)
    pred_te = capo_intv - capo_base
    mask =(~np.isnan(pred_te)) & (~np.isnan(gt_te))
    #te_mse = np.mean((gt_te[mask] - pred_te[mask])**2)
    #mlf_logger.log_metrics({'TE_mse': te_mse})
    #logger.info(f"Test MSE: {te_mse}")
    te_rmse = np.sqrt(np.mean((gt_te[mask] - pred_te[mask])**2))
    mlf_logger.log_metrics({'TE_rmse': te_rmse})
    logger.info(f"Test RMSE: {te_rmse}")


if __name__ == '__main__':
    main()

