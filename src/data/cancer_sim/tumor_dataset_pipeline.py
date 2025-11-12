from src.data.cancer_sim.simulation import generate_params, get_standard_params, \
    simulate_factual, simulate_counterfactuals_treatment_seq
from torch.utils.data import Dataset
from omegaconf import DictConfig
from src.data.base_dataset_pipeline import BaseDatasetPipeline, ProcessedDataset
import numpy as np
import logging
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TumorGrowthDatasetPipeline(BaseDatasetPipeline):
    """
    Uses code from Dynamic DML (DynamicPancelDGP class) 
    https://proceedings.neurips.cc/paper/2021/hash/bf65417dcecc7f2b0006e1f5793b7143-Abstract.html
    """

    def __init__(self, 
                 seed: int,
                 n_units: int,
                 n_periods: int,
                 sequence_length: int,
                 split: dict,
                 window_size: int,
                 lag: int,
                 lag_y: int,
                 conf_coeff: float,
                 normalize: bool = True,
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
            n_treatments=2,
            n_treatments_disc=2,
            n_treatments_cont=0,
            n_units=n_units,
            n_periods=n_periods,
            sequence_length=sequence_length,
            split=split,
        )
        self.name = 'TumorGrowthDatasetPipeline'
        #set random seed
        self.seed = seed
        np.random.seed(seed)
        #Step 1: Generate parameters
        params = generate_params(n_units, chemo_coeff=conf_coeff, radio_coeff=conf_coeff, window_size=window_size, lag=lag, lag_y = lag_y)
        params['window_size'] = window_size
        self.conf_str = conf_coeff
        logger.info("data parameter generation finished")

        #Generate observational data
        self.data = simulate_factual(params, seq_length = sequence_length)
        self.Y = self.data['cancer_volume'][:, lag_y:]
        chemo_application = self.data['chemo_application'][:, lag_y:].reshape(n_units, -1, 1)
        radio_application = self.data['radio_application'][:, lag_y:].reshape(n_units, -1, 1)
        self.T_disc = np.concatenate([chemo_application, radio_application], axis = -1)
        self.X_dynamic = np.zeros((n_units, sequence_length, 1))
        patient_type = self.data['patient_types'].astype(int)
        self.num_types = max(patient_type)
        self.X_static = np.zeros((n_units, self.num_types))
        self.gt_dynamic_effect_available = True
        self.true_effect = None
        self.beta_c = params['beta_c'].mean()
        self.alpha = params['alpha'].mean()
        self.beta = params['beta'].mean()
        self.chemo_dose = 5
        self.radio_dose = 2.
        #map X_static to one-hot encoding vector
        for i in range(n_units):
            self.X_static[i, patient_type[i] - 1] = 1
        sequence_lengths = self.data['sequence_lengths'] - lag_y
        self.active_entries = np.zeros((n_units, sequence_length, 1))
        self.params = params
        for i in range(n_units):
            L = int(sequence_lengths[i])
            self.active_entries[i, :L, :] = 1

        self.scaling_params = {
            'cancer_volume':{'mean': 0, 'std': 1}
        }
        if normalize:
            #Normalize the data
            self.scaling_params['cancer_volume']['mean'] = self.Y.mean()
            self.scaling_params['cancer_volume']['std'] = self.Y.std()
            self.Y = (self.Y - self.Y.mean()) / self.Y.std()
            #self.X_static = (self.X_static - self.X_static.mean(axis=0)) / self.X_static.std(axis=0)
            #self.X_dynamic = (self.X_dynamic - self.X_dynamic.mean(axis=0)) / self.X_dynamic.std(axis=0)

        self._split_data()

        self.train_data = self._get_torch_dataset('train')
        self.val_data = self._get_torch_dataset('val')
        self.test_data = self._get_torch_dataset('test')



    
    def _split_data(self):
        """
        Split the generated factual data into train/val/test subsets, given the ratio of self.val_split and self.test_split
        """
        #Generate indices for train, val and test
        index = np.arange(self.n_units)
        np.random.shuffle(index)
        train_pos = int(self.n_units * (1 - self.val_split - self.test_split))
        val_pos = int(self.n_units * (1 - self.test_split))
        train_index, val_index, test_index = index[:train_pos], index[train_pos:val_pos], index[val_pos:]
        self.factual_generated = True
        
        self.train_index, self.val_index, self.test_index = train_index, val_index, test_index
        self.index = {'train': train_index, 'val': val_index, 'test': test_index}
        return
    
    
    def compute_individual_true_dynamic_effects(self, cancer_volumes:np.ndarray):
        """
        cancer_volumes: (N, SL + lag_y)
        Returns:
            individual_de (N, SL - m + 1, m, 2)
        """
        m = self.n_periods
        SL = self.sequence_length
        lag_y = self.params['lag_y']
        assert cancer_volumes.shape[1] == SL + lag_y, f'cancer_volumes should have shape (N, {SL + lag_y - m + 1}), but got {cancer_volumes.shape}'
        individual_de = np.zeros((cancer_volumes.shape[0], SL - m + 1, m, self.n_treatments))
        for t in range(self.sequence_length - m + 1):
            for k in range(1, m):
                y_tlag_mean = cancer_volumes[:, :t + k + 1].mean(axis = 1)
                de_c = -1 * self.beta_c * y_tlag_mean * self.chemo_dose
                de_r = -1 * (self.alpha * self.radio_dose + self.beta * self.radio_dose ** 2) * y_tlag_mean
                individual_de[:, t, k, 0] = de_c
                individual_de[:, t, k, 1] = de_r

        return individual_de


    def _simulate_counterfactuals(self, subset:str, T_seq:np.ndarray):

        assert T_seq.shape[0] == self.n_periods, f'T_seq should have the same length as n_periods, but got {T_seq.shape[0]} and {self.n_periods}'
        subset_data = {
            k: v[self.index[subset]] for k, v in self.data.items()
        }
        subset_params = {'window_size': self.params['window_size'], 'lag': self.params['lag'], 'lag_y': self.params['lag_y']}
        subset_params['initial_stages'] = self.params['initial_stages'][self.index[subset]]
        subset_params['patient_types'] = self.params['patient_types'][self.index[subset]]
        subset_params['initial_volumes'] = self.params['initial_volumes'][self.index[subset]]
        subset_params['alpha'] = self.params['alpha'][self.index[subset]]
        subset_params['beta'] = self.params['beta'][self.index[subset]]
        subset_params['beta_c'] = self.params['beta_c'][self.index[subset]]
        subset_params['K'] = self.params['K'][self.index[subset]]
        subset_params['rho'] = self.params['rho'][self.index[subset]]
        #subset_params['scaling_params'] = self.scaling_params

        Y_ctf, active_entries = simulate_counterfactuals_treatment_seq(subset_data, subset_params, self.n_periods, T_seq)
        #normalize the data
        Y_ctf = (Y_ctf - self.scaling_params['cancer_volume']['mean']) / self.scaling_params['cancer_volume']['std']
        return Y_ctf

    
    def _get_torch_dataset(self, subset='train'):
        logger.info(f'Creating torch dataset for {subset} of tumor growth dataset')
        if subset == 'train':
            index = self.train_index
        elif subset == 'val':
            index = self.val_index
        elif subset == 'test':
            index = self.test_index
        else:
            raise ValueError(f'Invalid subset: {subset}.')
        
        
        return ProcessedDataset(torch.from_numpy(self.Y[index]), 
                                T_disc = torch.from_numpy(self.T_disc[index]), 
                                T_cont = None, 
                                X_static=torch.from_numpy(self.X_static[index]), 
                                X_dynamic= torch.from_numpy(self.X_dynamic[index]),
                                active_entries=torch.from_numpy(self.active_entries[index]), subset_name = subset)
    
    def insert_necessary_args_dml_rnn(self, args):
        """No additional arguments are needed for this dataset"""
        return
    
    def insert_necessary_args_dml_rnn(self, args):
        args.dataset['n_treatments_disc'] = 2
        args.dataset['n_treatments_cont'] = 0
        args.dataset['n_treatments'] = 2
        args.dataset['n_periods'] = self.n_periods
        args.dataset['sequence_length'] = self.sequence_length
        args.dataset['n_x'] = 1
        args.dataset['n_static'] = int(self.num_types)
        return
    
    def get_confounding_strength(self):
        return self.conf_str
        

if __name__ == '__main__':
    import omegaconf
    from hydra.utils import instantiate
    config_path = 'configs/datasets/tumor_debug.yaml'
    args = omegaconf.OmegaConf.load(config_path)
    args.dataset.seed = 2026
    data_pipeline = instantiate(args.dataset)