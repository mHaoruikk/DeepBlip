from omegaconf import DictConfig
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import Dataset, random_split
import matplotlib.pyplot as plt
import mlflow
import logging
import os
import torch
logger = logging.getLogger(__name__)

def plot_residual_distribution(mlf_logger, all_res_Y, all_res_T, args):
    """
    resY: shape (n_units, SL - n_period + 1, n_period)
    resT: shape (n_units, SL - n_period + 1, n_period, n_period, n_treatments)
    results logged to mlflow
    """
    logger.info("Visualising residuals of Y and T")
    n_periods = all_res_Y.shape[-1]
    all_res_Y_np = all_res_Y.cpu().numpy()
    all_res_T_np = all_res_T.cpu().numpy()

    # Plot and log res_Y distribution directly to MLflow
    fig_y, ax_y = plt.subplots(nrows = 1, ncols = n_periods, figsize=(4 * n_periods, 4))
    for p in range(n_periods):
        ax_y[p].hist(all_res_Y_np[:,:,p].flatten(), bins=50, alpha=0.7, color='blue')
        ax_y[p].set_title(f'Distribution of res_Y for p{p}')
        ax_y[p].set_xlabel('residual_Y')
        ax_y[p].set_ylabel('Frequency')
        plt.tight_layout()
    #figpath = os.path.join(artifacts_path, "res_T_distribution.png")
    #plt.savefig(figpath)
    #mlf_logger.experiment.log_artifact(mlf_logger.run_id, figpath)
    #mlflow.log_figure(fig_y, "res_Y_distribution.png")
    if mlf_logger is not None:
        mlflow.set_tracking_uri(args.exp.mlflow_uri)
        with mlflow.start_run(run_id=mlf_logger.run_id, nested=True):
            mlflow.log_figure(fig_y, "res_Y_distribution.png")
    plt.close(fig_y)

    # Plot and log res_T distribution directly to MLflow
    fig_t, ax_t = plt.subplots(nrows = n_periods, ncols = n_periods, figsize=(4 * n_periods, 4 * n_periods))
    for p in range(n_periods):
        for j in range(p, n_periods):
            ax_t[p, j].hist(all_res_T_np[:, :, p, j, :].mean(axis = -1).flatten(), bins=50, alpha=0.7, color='green')
            ax_t[p, j].set_title(f'Distribution of res_T for q{j}{p}')
            ax_t[p, j].set_xlabel('residual_T')
            ax_t[p, j].set_ylabel('Frequency')
        plt.tight_layout()
    #figpath = os.path.join(artifacts_path, "res_T_distribution.png")
    #plt.savefig(figpath)
    #mlf_logger.experiment.log_artifact(mlf_logger.run_id, figpath)
    if mlf_logger is not None:
        with mlflow.start_run(run_id=mlf_logger.run_id, nested=True):
            mlflow.log_figure(fig_t, "res_T_distribution.png")
    plt.close(fig_t)
    return

def log_param_est_rmse(mlf_logger, param_pred, true_effect, args):
    """
    Compare estimated parameter's mse against true effect, non-heterogenous
    param_pred: torch.Tensor of shape [D, SL - m + 1, m, n_t]
    true_effect: torch.Tensor of shape [m, n_t]
    """
    assert (len(args.dataset.hetero_inds) == 0) or (args.dataset.hetero_inds == None), "Only for Non-hetero estimation"
    #Reverse the rows of true_effect, since in the dgp the order is reversed
    param_pred_np = param_pred.detach().cpu().numpy()
    reversed_true_effect_np = np.flip(true_effect, axis = 0)

    D, K, m, n_t = param_pred_np.shape
    assert (m == args.dataset.n_periods) and (n_t == args.dataset.n_treatments)
    param_preds = param_pred_np.reshape(D * K, m, n_t)

    metrics = {}
    for i in range(m):
        preds_i = param_preds[:, i, :]
        true_i = reversed_true_effect_np[i, :]
        diff = preds_i - true_i
        rmse_i = np.sqrt(np.mean(diff ** 2))
        metrics[f'param_rmse{i}'] = rmse_i
    
    mlf_logger.log_metrics(metrics)
    return metrics


def plot_blip_est_distribution(mlf_logger, param_pred, true_effect, args):
    """
    distribution of dynamic effects
    Args:
        param_pred: torch.Tensor of shape [D, SL - m + 1, m, n_t]
        true_effect: numpy array of shape [m, n_t]
    results logged to mlflow
    """
    logger.info("Visualising param estimation")
    
    #assert (len(args.dataset.hetero_inds) == 0) or (args.dataset.hetero_inds == None), "Only for Non-hetero estimation"
    param_pred_np = param_pred.detach().cpu().numpy()
    D, K, m, n_t = param_pred_np.shape
    param_pred_2d = param_pred_np.reshape(D * K, m, n_t)
    reversed_true_effect_np = np.flip(true_effect, axis = 0)
    fig, axes = plt.subplots(nrows=m, ncols=n_t, 
                             figsize=(4 * n_t, 3 * m), 
                             sharex=False, sharey=False)
    if m == 1 and n_t == 1:
        axes = np.array([[axes]])
    elif m == 1:
        axes = axes.reshape(1, n_t)
    elif n_t == 1:
        axes = axes.reshape(m, 1)

    for i in range(m):
        for t in range(n_t):
            ax = axes[i, t]

            # Distribution (histogram) of predictions for row i, time t
            ax.hist(param_pred_2d[:, i, t], bins=100, alpha=0.7, color='blue')
            true_val = reversed_true_effect_np[i, t]
            ax.axvline(true_val, color='red', linestyle='--', linewidth=2,
                       label=f"GT = {true_val:.2f}")

            ax.set_title(f"period={i}, treatment ={t}")
            ax.legend(loc="best")
    
    if mlf_logger is not None:
        mlflow.set_tracking_uri(args.exp.mlflow_uri)
        with mlflow.start_run(run_id=mlf_logger.run_id, nested=True):
            mlflow.log_figure(fig, "de_est_hist_nonhetero.png")
    plt.close(fig)
    return

def plot_blip_est_diff_distribution(mlf_logger, param_pred, individual_true_effect, args, artifacts_path: str = None):
    """
    plot distribution of the difference between estimated and true effect on a individual level
    Args:
        param_pred: torch.Tensor of shape [D, SL - m + 1, m, n_t]
        individual_true_effect: numpy array of shape [D, SL - m + 1, m, n_t]
        args: DictConfig object containing experiment configurations.
        artifacts_path: Path to the directory where artifacts should be saved.
    results logged to mlflow
    """
    logger.info("Visualising param estimation on an individual level")
    param_pred_np = param_pred.detach().cpu().numpy()
    te_diff = param_pred_np - individual_true_effect

    if artifacts_path:
        te_diff_filename = "te_diff_individual.npy"
        te_diff_filepath = os.path.join(artifacts_path, te_diff_filename)
        with open(te_diff_filepath, 'wb') as f:
            np.save(f, te_diff)
        logger.info(f"Saved individual te_diff to {te_diff_filepath}")

    m = args.dataset.n_periods
    n_t = individual_true_effect.shape[-1]
    fig, axes = plt.subplots(nrows=m, ncols=n_t, figsize=(4 * n_t, 3 * m), sharex=False, sharey=False)
    if m == 1 and n_t == 1:
        axes = np.array([[axes]])
    elif m == 1:
        axes = axes.reshape(1, n_t)
    elif n_t == 1:
        axes = axes.reshape(m, 1)
    for i in range(m):
        for t in range(n_t):
            ax = axes[i, t]
            # Distribution (histogram) of predictions for row i, time t
            ax.hist(te_diff[:, :, i, t].flatten(), bins=100, alpha=0.6, color='blue')
            ax.set_title(f"period={i}, treatment={t}")
            ax.legend(loc="best")
    fig.suptitle("Distribution of the difference between estimated and \n ground truth dynamic effect for each individual")
    #plt.show()
    if mlf_logger is not None:
        mlflow.set_tracking_uri(args.exp.mlflow_uri)
        with mlflow.start_run(run_id=mlf_logger.run_id, nested=True):
            mlflow.log_figure(fig, "de_est_hist_hetero.png")
            mlflow.log_artifact(te_diff_filepath)
    plt.close(fig)
    return
    

def dataset_to_array(args:DictConfig, dataset:Dataset):
    seq_length = args.dataset.sequence_length
    n_units = len(dataset)
    n_treatments = args.dataset.n_treatments
    n_x = args.dataset.n_x
    T = np.zeros((n_units, seq_length, n_treatments), dtype = np.float32)
    X = np.zeros((n_units, seq_length, n_x), dtype = np.float32)
    Y = np.zeros((n_units, seq_length), dtype = np.float32)
    for i, data in enumerate(dataset):
        T[i] = data['curr_treatments']
        X[i] = data['curr_covariates']
        Y[i] = data['curr_outputs']
    return Y, T, X


def preprocess_data_for_dml(args: DictConfig, Y: np.array, T:np.array, X:np.array):
    """
    transform the standard format to the format required by HeteroDynamicPanelDML
    For example:
    T:array(n_units, seq_length, n_treat) -> (n_units * n_blocks * n_periods, n_treat)
    """
    n_periods = args.dataset.n_periods
    n_units = Y.shape[0]
    n_treatments = T.shape[2]
    n_x = X.shape[2]
    n_blocks = Y.shape[1] // n_periods
    L = n_blocks * n_periods
    groups = np.repeat(np.arange(n_units * n_blocks), n_periods)

    T_reshaped = T[:, :L, :].reshape(n_units, n_blocks, n_periods, n_treatments)
    T_reshaped = T_reshaped.reshape(n_units * n_blocks * n_periods, n_treatments)
    X_reshaped = X[:, :L, :].reshape(n_units, n_blocks, n_periods, n_x)
    X_reshaped = X_reshaped.reshape(n_units * n_blocks * n_periods, n_x)
    Y_reshaped = Y[:, :L].reshape(n_units, n_blocks, n_periods)
    Y_reshaped = Y_reshaped.reshape(-1,)
    
    return Y_reshaped, T_reshaped, X_reshaped, groups

def evaluate_nuisance_mse(resT, resY, n_periods):
    results = {}
    for kappa in resT:
        results[f'dml_p{kappa}_mse'] = (resY[kappa] ** 2).mean()
        for tau in np.arange(kappa, n_periods):
            results[f'dml_q{kappa}{tau}_mse'] = (resT[kappa][tau] ** 2).mean()
    return results

def transform_residual_data(resY, resT, n_periods):
    """
    transform the residual data from regression based to the form required by the neural-R learner
    resT, resY dict, resY[i]=array(N,), resT[i][j]=array(N, n_t)
    return the standard format (pytorch tensor with time steps = 1)
    """
    assert n_periods == len(resY)
    N = resY[0].shape[0]
    n_t = resT[0][0].shape[-1]
    resY_all_steps = np.zeros((N, 1, n_periods))
    resT_all_steps = np.zeros((N, 1, n_periods, n_periods, n_t))
    for kappa in np.arange(n_periods):
        resY_all_steps[:, 0, kappa] = resY[kappa]
        for tau in np.arange(kappa, n_periods):
            resT_all_steps[:, 0, kappa, tau, :] = resT[kappa][tau]
    return torch.from_numpy(resY_all_steps), torch.from_numpy(resT_all_steps)

def build_phi(option):
    if option == 'current_treatment':
        def phi(Xt_tj, Tt_tj):
            return Tt_tj[:, -1, :]
        return phi
    else:
        raise ValueError(f'{option} for phi not implemented')

def combine_disc_cont(T_disc:np.ndarray, T_cont:np.ndarray):
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


def grad_reverse(x, scale=1.0):

    class ReverseGrad(Function):
        """
        Gradient reversal layer
        """

        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad_output):
            return scale * grad_output.neg()

    return ReverseGrad.apply(x)


def bce(treatment_pred, current_treatments, mode, weights=None):
    if mode == 'multiclass':
        return F.cross_entropy(treatment_pred.permute(0, 2, 1), current_treatments.permute(0, 2, 1), reduce=False, weight=weights)
    elif mode == 'multilabel':
        return F.binary_cross_entropy_with_logits(treatment_pred, current_treatments, reduce=False, weight=weights).mean(dim=-1)
    else:
        raise NotImplementedError()


