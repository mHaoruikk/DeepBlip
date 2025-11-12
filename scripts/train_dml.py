import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_original_cwd
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from src.utils import FilteringMlFlowLogger
from src.data.linear_synthetic import MarkovianHeteroDynamicDataset
from src.models.dml import HeteroDynamicPanelDML, DynamicPanelDML
from src.models.utils import dataset_to_array, preprocess_data_for_dml, evaluate_nuisance_mse, plot_residual_distribution
from src.utils import log_params_from_omegaconf_dict
from sklearn.linear_model import LassoCV, Lasso, MultiTaskLassoCV
from econml.sklearn_extensions.linear_model import SelectiveRegularization
from sklearn.model_selection import KFold
import numpy as np
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)

@hydra.main(version_base='1.1', config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):

    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))
    seed_everything(args.exp.seed)
    #create dataset
    assert (len(args.dataset.hetero_inds) == 0) or (args.dataset.hetero_inds == None), 'No hetero effects should be in dataset!'
    hddataset = MarkovianHeteroDynamicDataset(params=args.dataset)
    Y, T, X = hddataset.generate_observational_data(policy=None,  seed=args.dataset.get('seed', 2024))
    full_dataset = hddataset.get_full_dataset(Y, T, X)

    if args.exp.logging:
        experiment_name = args.exp.exp_name
        mlf_logger = FilteringMlFlowLogger(
            filter_submodels=[], 
            experiment_name=experiment_name, 
            tracking_uri=args.exp.mlflow_uri, 
            run_name=f'seed{args.dataset.seed}_nx{args.dataset.n_x}_nt{args.dataset.n_treatments}'
        )
        artifacts_path = hydra.utils.to_absolute_path(
            mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri
        ).replace('mlflow-artifacts:', 'mlruns')
        logger.info(f"Artifacts path: {artifacts_path}")
    else:
        mlf_logger = None
        artifacts_path = None

    logger.info("Running regression based DML")
    #prepare dataset
    Y_np, T_np, X_np = dataset_to_array(args, full_dataset)
    Y_np, T_np, X_np, groups_np = preprocess_data_for_dml(args, Y_np, T_np, X_np)

    lasso_model = lambda : LassoCV(cv=3, n_alphas=6, max_iter=200)
    mlasso_model = lambda : MultiTaskLassoCV(cv=3, n_alphas=6, max_iter=200)

    dml = DynamicPanelDML(model_t=mlasso_model(), model_y=lasso_model(), n_cfit_splits=2)
    resT, resY = dml.fit_nuisances(Y_np, T_np, X_np, groups_np, args.dataset.n_periods)
    if args.exp.save_residual:
        pickle.dump(resT, open(os.path.join(artifacts_path, 'resT_full.pkl'), 'wb'))
        pickle.dump(resY, open(os.path.join(artifacts_path, 'resY_full.pkl'), 'wb'))
        logger.info(f"Residual data ResT, ResY saved to {artifacts_path}")
    residual_results = evaluate_nuisance_mse(resT, resY, args.dataset.n_periods)
    mlf_logger.log_metrics(residual_results)
    dml.fit(Y_np, T_np, X_np, groups_np)

    n_x = args.dataset.n_x
    n_treatments = args.dataset.n_treatments
    n_periods = args.dataset.n_periods
    true_effect = hddataset.true_effect.flatten()

    param_hat = dml.param
    conf_ints = dml.param_interval(alpha=.05)
    est_results = {}
    for kappa in range(n_periods):
        for t in range(n_treatments):
            param_ind = kappa*n_treatments + t
            logger.info("Effect Lag={}, T={}: {:.3f} ({:.3f}, {:.6f}), (Truth={:.6f})".format(kappa, t,
                                                                                        param_hat[param_ind],
                                                                                        *conf_ints[param_ind],
                                                                                        true_effect[param_ind]))
            est_results[f"Lag{kappa}_tidx{t}_Param_hat"] = param_hat[param_ind]
            est_results[f"Lag{kappa}_tidx{t}_Param_true"] = true_effect[param_ind]
    
    mlf_logger.log_metrics(est_results)
    
            


if __name__ == "__main__":
    main()