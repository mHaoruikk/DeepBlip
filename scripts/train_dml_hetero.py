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
from src.models.dml import HeteroDynamicPanelDML
from src.models.utils import dataset_to_array, preprocess_data_for_dml, evaluate_nuisance_mse, plot_residual_distribution
from src.utils import log_params_from_omegaconf_dict
from sklearn.linear_model import LassoCV, Lasso, MultiTaskLassoCV
from econml.sklearn_extensions.linear_model import SelectiveRegularization
from sklearn.model_selection import KFold
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)

@hydra.main(version_base='1.1', config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):

    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))
    seed_everything(args.exp.seed)
    #create dataset
    hddataset = MarkovianHeteroDynamicDataset(params=args.dataset)
    Y, T, X = hddataset.generate_observational_data(policy=None,  seed=args.dataset.get('seed', 2024))
    train_dataset, val_dataset = hddataset.get_processed_data(Y, T, X)

    if args.exp.logging:
        experiment_name = args.exp.exp_name
        mlf_logger = FilteringMlFlowLogger(
            filter_submodels=[], 
            experiment_name=experiment_name, 
            tracking_uri=args.exp.mlflow_uri, 
            run_name=f'parameter_seed{args.exp.seed}'
        )
    else:
        mlf_logger = None

    logger.info("Running regression based DML")
    #prepare dataset
    Y_train, T_train, X_train = dataset_to_array(args, train_dataset)
    Y_val, T_val, X_val = dataset_to_array(args, val_dataset)
    Y_train, T_train, X_train, groups_train = preprocess_data_for_dml(args, Y_train, T_train, X_train)
    Y_val, T_val, X_val, groups_val = preprocess_data_for_dml(args, Y_val, T_val, X_val)

    lasso_model = lambda : LassoCV(cv=3, n_alphas=6, max_iter=200)
    mlasso_model = lambda : MultiTaskLassoCV(cv=3, n_alphas=6, max_iter=200)
    alpha_regs = [1e-4, 1e-3, 5e-2, 1e-1, .5, 1]
    slasso_model = lambda : SelectiveRegularization([0],
                                            LassoCV(cv=KFold(n_splits=3, shuffle=True),
                                                    alphas=alpha_regs, max_iter=200, fit_intercept=False),
                                            fit_intercept=False)
    hetero_inds = np.array(args.dataset.hetero_inds)
    dml = HeteroDynamicPanelDML(model_t=mlasso_model(),
                                model_y=lasso_model(),
                                model_final=slasso_model(),
                                n_cfit_splits=3).fit(Y_train, T_train, X_train, groups_train, hetero_inds=hetero_inds)
    #resT, resY = dml.fit_nuisances(Y_train, T_train, X_train, groups_train, args.dataset.n_periods)
    #dml_results = evaluate_nuisance_mse(resT, resY, args.dataset.n_periods)
    #mlf_logger.log_metrics(dml_results)
    dml.fit(Y_train, T_train, X_train, groups_train, hetero_inds)

    true_effect_inds = []
    n_x = args.dataset.n_x
    n_treatments = args.dataset.n_treatments
    n_periods = args.dataset.n_periods
    for t in range(args.dataset.n_treatments):
        true_effect_inds += [t * (1 + n_x)] + (list(t * (1 + n_x) + 1 + hetero_inds) if len(hetero_inds)>0 else [])
    true_effect_params = hddataset.true_hetero_effect[:, true_effect_inds].flatten()

    param_hat = dml.param
    conf_ints = dml.param_interval(alpha=.01)
    for kappa in range(n_periods):
        for t in range(n_treatments * (len(hetero_inds) + 1)):
            param_ind = kappa * (len(hetero_inds) + 1) * n_treatments + t
            print("Effect Lag={}, TX={}: {:.3f} ({:.3f}, {:.3f}), (Truth={:.3f})".format(kappa, t,
                                                                                        param_hat[param_ind],
                                                                                        *conf_ints[param_ind],
                                                                                        true_effect_params[param_ind]))




if __name__ == "__main__":
    main()