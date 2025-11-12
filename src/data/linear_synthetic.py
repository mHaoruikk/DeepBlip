import numpy as np
from econml.utilities import cross_product
from tqdm import tqdm
from statsmodels.tools.tools import add_constant
import torch
from torch.utils.data import Dataset, random_split

from src.data.base_dataset_pipeline import BaseDatasetPipeline, ProcessedDataset
import logging
logger = logging.getLogger(__name__)


class linearMarkovianDataPipeline(BaseDatasetPipeline):
    """
    Uses code from Dynamic DML (DynamicPancelDGP class) 
    https://proceedings.neurips.cc/paper/2021/hash/bf65417dcecc7f2b0006e1f5793b7143-Abstract.html
    """

    def __init__(self, 
                 seed: int,
                 n_treatments: int,
                 n_treatments_disc: int,
                 n_treatments_cont: int,
                 s_t: int,
                 n_units: int,
                 n_periods: int,
                 sequence_length: int,
                 split: dict,
                 hetero_inds: list,
                 sigma_x: float, sigma_y: float, 
                 sigma_t: float, 
                 gamma: float,
                 n_x: int, 
                 s_x: int,
                 state_effect: float, 
                 autoreg: float, 
                 hetero_strength: float, 
                 conf_str: float,
                 **kwargs):
        """
        Initialize the dataset pipeline for linear markovian heterodynamic dataset
        Args:
            hetero_inds: list of indices of heterogenous covariates
            s_x: number of endogenous covariates variables
            s_t: number of effective treatment variables
            state_effect: state effect
            autoreg: autoregression coefficient
            conf_str: the strength of the confounding effect
        """
        super().__init__(
            seed = seed, 
            n_treatments=n_treatments,
            n_treatments_disc=n_treatments_disc,
            n_treatments_cont=n_treatments_cont,
            n_units=n_units,
            n_periods=n_periods,
            sequence_length=sequence_length,
            split=split,
        )
        self.name = 'linear_markovian_heterodynamic'
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.n_x = n_x
        self.s_x = s_x
        self.s_t = s_t
        self.sigma_t = sigma_t
        self.gamma = gamma
        self.state_effect = state_effect
        self.autoreg = autoreg
        self.hetero_strength = hetero_strength
        self.conf_str = conf_str
        self.gt_dynamic_effect_available = True

        self.hetero_inds = np.array(hetero_inds, dtype=np.int32) if (
                                        (hetero_inds is not None) and (len(hetero_inds) > 0)) else None
        self.endo_inds = np.setdiff1d(np.arange(self.n_x), self.hetero_inds).astype(int)

        np.random.seed(seed)
        self.Alpha = np.random.uniform(-1, 1, size = (self.n_x, self.n_treatments))
        self.Alpha *= self.state_effect
        if self.hetero_inds is not None:
            self.Alpha[self.hetero_inds] = 0.
        
        self.Beta = np.zeros((self.n_x, self.n_x))
        for t in range(self.n_x):
            self.Beta[t, :] = self.autoreg * np.roll(np.random.uniform(low=4.0**(-np.arange(
                0, self.n_x)), high=4.0**(-np.arange(1, self.n_x + 1))), t)
        if self.hetero_inds is not None:
            self.Beta[np.ix_(self.endo_inds, self.hetero_inds)] = 0.
            self.Beta[np.ix_(self.hetero_inds, self.endo_inds)] = 0.
        
        self.epsilon = np.random.uniform(-1, 1, size = self.n_treatments)
        self.zeta = np.zeros(self.n_x)
        self.zeta[:self.s_x] = self.conf_str / self.s_x
        
        self.y_hetero_effect = np.zeros(self.n_x)
        self.x_hetero_effect = np.zeros(self.n_x)
        if self.hetero_inds is not None:
            self.y_hetero_effect[self.hetero_inds] = np.random.uniform(0.5 * self.hetero_strength, 1.5* self.hetero_strength) / len(self.hetero_inds)
            self.x_hetero_effect[self.hetero_inds] = np.random.uniform(0.5 * self.hetero_strength, 1.5* self.hetero_strength) / len(self.hetero_inds)

        self.true_effect = np.zeros((self.n_periods, self.n_treatments))
        self.true_effect[0] = self.epsilon
        for t in range(1, self.n_periods):
            self.true_effect[t, :] = self.zeta.reshape(1, -1) @ np.linalg.matrix_power(self.Beta, t - 1) @ self.Alpha
        
        self.true_hetero_effect = np.zeros((self.n_periods, (self.n_x + 1) * self.n_treatments))
        self.true_hetero_effect[0, :] = cross_product(add_constant(self.y_hetero_effect.reshape(1, -1), has_constant = 'add'), self.epsilon.reshape(1, -1))
        for t in np.arange(1, self.n_periods):
            self.true_hetero_effect[t, :] = cross_product(add_constant(self.x_hetero_effect.reshape(1, -1), has_constant='add'), 
                                                            self.zeta.reshape(1, -1) @ np.linalg.matrix_power(self.Beta, t - 1) @ self.Alpha)
        logger.info("data parameter generation finished")
        #maintain the data and the index
        self.Y_obs = {'train': None, 'val': None, 'test': None}
        self.T_obs = {'train': None, 'val': None, 'test': None}
        self.X_obs = {'train': None, 'val': None, 'test': None}

        #store the random noises
        self.noisex = np.zeros((self.n_units, self.sequence_length, self.n_x))
        self.noisey = np.zeros((self.n_units, self.sequence_length))
        
        Y, T, X = self._generate_full_factual_data()
        self._split_data(Y, T, X)

        self.train_data = self._get_torch_dataset('train')
        self.val_data = self._get_torch_dataset('val')
        self.test_data = self._get_torch_dataset('test')


    def _generate_full_factual_data(self, policy = None):
        """
        Generate the full factual data for the dataset
        """
        logger.info(f'Generating observational linear markovian heterodynamic dataset')
        s_t = self.s_t
        sigma_t = self.sigma_t
        gamma = self.gamma

        self.Delta = np.zeros((self.n_treatments, self.n_x))
        self.Delta[:, :s_t] = self.conf_str / s_t

        if policy is None:
            def policy(Tprev, X, period):
                return gamma * Tprev + (1 - gamma) * self.Delta @ X + np.random.normal(0, sigma_t, size = self.n_treatments)
        
        np.random.seed(self.seed)
        Y = np.zeros((self.n_units, self.sequence_length))
        T = np.zeros((self.n_units, self.sequence_length, self.n_treatments))
        X = np.zeros((self.n_units, self.sequence_length, self.n_x))
        for i in tqdm(range(self.n_units), desc = 'Generating data'):
            for t in range(self.sequence_length):
                #Generate random exogeneous noise for X, y
                self.noisex[i, t] = np.random.normal(0, self.sigma_x, size = self.n_x)
                self.noisey[i, t] = np.random.normal(0, self.sigma_y)
                if t == 0:
                    X[i][0] = self.noisex[i, 0]
                    T[i][0] = policy(np.zeros(self.n_treatments), X[i][0], 0)
                else:
                    X[i][t] = (1 + np.dot(self.x_hetero_effect, X[i][t - 1])) * np.dot(self.Alpha, T[i][t - 1]) + \
                                np.dot(self.Beta, X[i][t - 1]) + self.noisex[i, t]
                    T[i][t] = policy(T[i][t - 1], X[i][t - 1], t)
                #Generate outcome
                Y[i][t] = (np.dot(self.y_hetero_effect, X[i][t]) + 1) * np.dot(self.epsilon, T[i][t]) + \
                            np.dot(X[i][t], self.zeta) + self.noisey[i, t]
        return Y, T, X
    
    def _split_data(self, Y, T, X):
        """
        Split the generated factual data into train/val/test subsets, given the ratio of self.val_split and self.test_split
        """
        #Generate indices for train, val and test
        index = np.arange(self.n_units)
        np.random.shuffle(index)
        train_pos = int(self.n_units * (1 - self.val_split - self.test_split))
        val_pos = int(self.n_units * (1 - self.test_split))
        train_index, val_index, test_index = index[:train_pos], index[train_pos:val_pos], index[val_pos:]
        
        self.Y_obs['train'], self.Y_obs['val'], self.Y_obs['test'] = Y[train_index], Y[val_index], Y[test_index]
        self.T_obs['train'], self.T_obs['val'], self.T_obs['test'] = T[train_index], T[val_index], T[test_index]
        self.X_obs['train'], self.X_obs['val'], self.X_obs['test'] = X[train_index], X[val_index], X[test_index]
        self.factual_generated = True
        
        self.train_index, self.val_index, self.test_index = train_index, val_index, test_index
        self.index = {'train': train_index, 'val': val_index, 'test': test_index}
        return
    
    
    def compute_individual_true_dynamic_effects(self, X):
        """For linear markovian hetero datasets, we could compute the individual dynamic effects directly from self.true_hetero_effect
        Args:
            X: Covariate, np.ndarray of shape (N, SL, n_x) (the generated covariate)
        
        Returns:
            individual_de (N, SL - m + 1, m, n_t)
        """
        m = self.n_periods
        SL = self.sequence_length
        individual_de = np.zeros((X.shape[0], SL - m + 1, m, self.n_treatments))
        for t in range(self.sequence_length - m + 1):
            for l in range(m - 1, -1, -1):
                individual_de[:, t, l, :] = add_constant(X[:, t+l, :], has_constant='add') @ \
                                                    self.true_hetero_effect[m - 1 - l, :].reshape((self.n_treatments, 1 + self.n_x)).T
        return individual_de


    def _simulate_counterfactuals(self, subset, T_seq):
        """Simulates outcomes for arbitrary treatment sequence using stored noise"""
        logger.info(f"Simulating rolling counterfactual outcomes on linear markovian hetero dataset for {subset}")
        assert T_seq.shape == (self.n_periods, self.n_treatments)
        m = self.n_periods
        #buffer for storing intervened values in time window of length m
        n_units = len(self.index[subset])
        X_ctf_local = np.zeros((n_units, m, self.n_x))
        Y_ctf_local = np.zeros((n_units, m))
        #expand T_seq
        T_intv = T_seq.reshape(1, self.n_periods, self.n_treatments).repeat(n_units, axis = 0)
        #initialize result
        Y_ctf = np.zeros((n_units, self.sequence_length - m + 1))

        #fetch the background noise variables and factual covariate
        noisex = self.noisex[self.index[subset]]
        noisey = self.noisey[self.index[subset]]
        X_obs = self.X_obs[subset]
        for t in range(self.sequence_length - self.n_periods + 1):
            #Compute the counterfactual X_ctf and Y_ctf
            for l in range(self.n_periods):
                if l == 0:
                    X_ctf_local[:, 0, :] = X_obs[:, t, :]
                else:
                    hetero_multiplier_x = 1 + (X_ctf_local[:, l - 1, :] * self.x_hetero_effect).sum(axis=1)
                    X_ctf_local[:, l, :] = np.expand_dims(hetero_multiplier_x, axis = 1) * np.dot(T_intv[:, l - 1, :], self.Alpha.T) + \
                                        np.dot(X_ctf_local[:, l - 1, :], self.Beta.T) + noisex[:, t + l, :]
                hetero_multiplier_y = 1 + (X_ctf_local[:, l, :] * self.y_hetero_effect).sum(axis = 1)
                Y_ctf_local[:, l] = hetero_multiplier_y * np.dot(T_intv[:, l, :], self.epsilon) + \
                                    np.dot(X_ctf_local[:, l, :], self.zeta) + noisey[:, t + l]
            Y_ctf[:, t] = Y_ctf_local[:, -1]
        return Y_ctf

    
    def get_processed_data(self, Y, T, X):
        logger.info(f'Processing markovian heterodynamic dataset')
        if not isinstance(Y, torch.Tensor):
            Y = torch.from_numpy(Y).double()
        if not isinstance(T, torch.Tensor):
            T = torch.from_numpy(T).double()
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).double()

        prev_T = torch.zeros_like(T)
        prev_T[:, 1:, :] = T[:, :-1, :]

        prev_Y = torch.zeros_like(Y)
        prev_Y[:, 1:] = Y[:, :-1]

        torch.manual_seed(self.seed)
        indices = torch.randperm(Y.shape[0])
        train_size = int(Y.shape[0] * self.train_val_split)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = ProcessedDataset(Y[train_indices], T[train_indices], X[train_indices], 
                                         prev_Y[train_indices], prev_T[train_indices])
        val_dataset = ProcessedDataset(Y[val_indices], T[val_indices], X[val_indices], 
                                         prev_Y[val_indices], prev_T[val_indices])

        return train_dataset, val_dataset
    
    def _get_torch_dataset(self, subset='train'):
        logger.info(f'Creating torch dataset for {subset} of linear markovian heterodynamic dataset')
        
        return ProcessedDataset(torch.from_numpy(self.Y_obs[subset]), 
                                T_disc = None, T_cont = torch.from_numpy(self.T_obs[subset]), 
                                X_static=None, X_dynamic= torch.from_numpy(self.X_obs[subset]),
                                active_entries=None, subset_name = subset)
    
    def insert_necessary_args_dml_rnn(self, args):
        """No additional arguments are needed for this dataset"""
        return