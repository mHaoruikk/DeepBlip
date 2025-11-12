import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.data.mimic.semi_synthetic_dataset import SyntheticOutcomeGenerator, SyntheticTreatment
from src.data.mimic.load_data import load_mimic3_data_raw
from src.data.mimic.utils import sigmoid, SplineTrendsMixture
from src.data.base_dataset_pipeline import BaseDatasetPipeline, ProcessedDataset
#import List
from typing import List, Dict
import logging
logger = logging.getLogger(__name__)
from joblib import Parallel, delayed
from tqdm import tqdm

#Create a dataset pipeline for MIMIC-III semisynthetic data '
class MIMICSemiSyntheticDataPipeline(BaseDatasetPipeline):

    def __init__(self,
                 path: str,
                 min_seq_length: int,
                 max_seq_length: int,
                 max_number: int,
                 n_periods: int,
                 vital_list: List[str],
                 static_list: List[str],
                 synth_outcome: SyntheticOutcomeGenerator,
                 synth_treatments_list: List[SyntheticTreatment],
                 treatment_outcomes_influence: Dict[str, list[str]],
                 autoregressive: bool = True,
                 parallel: bool = False,
                 te_model: str = 'min',
                 split = {'val': 0.15, 'test': 0.15},
                 seed=2025,
                 **kwargs):
        
        super().__init__(seed = seed, 
                         n_treatments = len(synth_treatments_list),
                         n_treatments_cont= 0,
                         n_treatments_disc= len(synth_treatments_list),
                         n_units = max_number,
                         n_periods = n_periods,
                         sequence_length= max_seq_length - 1,
                         split = split,
                         kwargs = kwargs)
        
        self.synth_outcome = synth_outcome
        self.synthetic_outcomes = [synth_outcome]
        self.synthetic_treatments = synth_treatments_list
        self.treatment_outcomes_influence = treatment_outcomes_influence
        self.seed = seed
        self.path = path,
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.sequence_length = max_seq_length - 1
        self.vital_cols = vital_list
        self.static_list = static_list
        self.autoregressive = autoregressive
        self.parallel = parallel
        self.name = 'MIMIC-III Semi-Synthetic Data Pipeline'
        self.gt_dynamic_effect_available = True
        self.betas = np.array([treat.full_effect for treat in self.synthetic_treatments])
        scale_function_types = [treatment.scale_function['type'] for treatment in self.synthetic_treatments]
        if all([t == 'identity' for t in scale_function_types]) == False:
            logger.info("Creating hetero true dynamic effects")
            self.true_effect_hetero_multiplier = np.stack(
                [self.betas / (i + 1)**0.5 for i in range(n_periods)], axis = 0
            )
            self.true_effect = None
        else:
            logger.info("Creating homogenous true dynamic effects")
            self.true_effect = np.stack(
                [self.betas / (i + 1)**0.5 for i in range(n_periods)], axis = 0
            )
            self.true_effect_hetero_multiplier = None

        self.te_model = te_model

        self.all_vitals, self.static_features = load_mimic3_data_raw(path, min_seq_length, max_seq_length, 
                                                                     max_number=max_number, 
                                                                     vital_list = vital_list,
                                                                     static_list = static_list,
                                                                     data_seed = seed)
        #Reset subject_id from 0 to n_units - 1
        self._remap_index()

        self.treatment_cols = [t.treatment_name for t in self.synthetic_treatments]
        self.outcome_col = synth_outcome.outcome_name
        self.treatment_options = [0., 1.] #currently only support binary treatments

        #Simulate untreated outcome Z for all the data, appending y_endo, y_exog, 
        # y_untreated and y (as placeholder for further compuation)
        self.synth_outcome.simulate_untreated(self.all_vitals, self.static_features)

        for treatment in self.synthetic_treatments:
            self.all_vitals[treatment.treatment_name] = 0.0
        self.all_vitals['fact'] = np.nan
        self.all_vitals.loc[(slice(None), 0), 'fact'] = 1.0
        user_sizes = self.all_vitals.groupby(level='subject_id', sort=False).size()

        logger.info(f'Simulating factual treatments and applying them to outcomes.')
        par = Parallel(n_jobs=4, backend='loky')
        seeds = np.random.randint(0, 10000, size=len(self.static_features))
        if parallel:
            self.all_vitals = par(delayed(self.treat_patient_factually)(patient_ix, seed, self.te_model)
                            for patient_ix, seed in tqdm(zip(self.static_features.index, seeds), total=len(self.static_features)))
        else:
            #Process all the patients sequentially
            self.all_vitals = [self.treat_patient_factually(patient_ix, seed, self.te_model) for patient_ix, seed in \
                                        tqdm(zip(self.static_features.index, seeds), total=len(self.static_features))]
        self.factual_generated = True
        logger.info('Concatenating all the trajectories together.')
        #Each single patient dataframe has new columns: 
        # ['y1_exog', 'y1_endo', 'y1_untreated', 'y1', 'y2_exog'.., 'y2', 'fact', 't1', 't2']
        self.all_vitals = pd.concat(self.all_vitals, keys=self.static_features.index)
        #Restore the name of subjec_id to the first level index
        self.all_vitals.index = self.all_vitals.index.set_names(['subject_id', 'hours_in'])

        # Padding with nans
        self.all_vitals = self.all_vitals.unstack(fill_value=np.nan, level=0).stack(dropna=False).swaplevel(0, 1).sort_index()
        
        #Conversion to numpy arrays
        self.static_features = self.static_features.sort_index()
        self.static_features = self.static_features.values
        self.treatments = self.all_vitals[self.treatment_cols].fillna(0.0).values.reshape((-1, max(user_sizes),
                                                                                      len(self.treatment_cols)))
        self.vitals_np = self.all_vitals[self.vital_cols].fillna(0.0).values.reshape((-1, max(user_sizes), len(self.vital_cols)))
        self.outcome_unscaled = self.all_vitals[self.outcome_col].fillna(0.0).values.reshape((-1, max(user_sizes)))
        self.outcome_scaled = self._get_scaled_outcome()
        self.active_entries = (~self.all_vitals.isna().all(1)).astype(float)
        self.active_entries = self.active_entries.values.reshape((-1, max(user_sizes)))
        self.user_sizes = np.squeeze(self.active_entries.sum(1))

        logger.info(f'Shape of exploded vitals: {self.vitals_np.shape}.')

        #split data, self.train_data, self.train_index, self.index
        self._split_data()
        self.train_data = self.get_torch_dataset(subset='train')
        self.val_data = self.get_torch_dataset(subset='val')
        self.test_data = self.get_torch_dataset(subset='test')
        logger.info('Data pipeline initialized.')

    def _remap_index(self):
        """
        Remap the index of self.all_vitals to start from 0 to self.n_units - 1,
        set the subject_id of self.all_vitals and self.static_features to be the same, namely 0, 1, 2, ..., self.n_units - 1
        """
        assert self.all_vitals is not None
        assert self.static_features is not None
        self.index_mapping = self.all_vitals.index.get_level_values('subject_id').unique()
        assert self.n_units == len(self.index_mapping)
        mapping = {old_id: new_id for new_id, old_id in enumerate(self.index_mapping)}
        new_index = pd.MultiIndex.from_tuples(
            [(mapping[old_id], traj_id) for old_id, traj_id in self.all_vitals.index],
            names = ['subject_id', 'hours_in']
        )
        self.all_vitals.index = new_index
        self.static_features.index = range(self.n_units)

    
    def _split_data(self):
        """
            split data into train, val, test by self.split, store index in self.index, self.train_index, self.val_index, self.test_index
        """
        #first assert that the data is sorted by subject_id and from 0 to self.n_units - 1
        assert self.all_vitals.index.get_level_values('subject_id').is_monotonic_increasing
        assert self.all_vitals.index.get_level_values('subject_id').min() == 0
        assert self.all_vitals.index.get_level_values('subject_id').max() == self.n_units - 1
        #get the index of self.all_vitals
        index = np.arange(self.n_units)
        np.random.shuffle(index)
        train_pos = int(self.n_units * (1 - self.val_split - self.test_split))
        val_pos = int(self.n_units * (1 - self.test_split))
        train_index, val_index, test_index = index[:train_pos], index[train_pos:val_pos], index[val_pos:]
        self.train_index, self.val_index, self.test_index = train_index, val_index, test_index
        self.index = {'train': train_index, 'val': val_index, 'test': test_index}
        #logger.info(f'Split data into train, val, test by {self.index}.')

        return
    
    def _get_scaled_outcome(self):
        """
        Scale the outcome to standard normal distribution.
        """
        assert self.outcome_unscaled is not None
        logger.info("Scaling outcomes to standard normal distribution. (only along the first axis (patients))")
        self.scaling_params = {
            'mean': self.outcome_unscaled.mean(axis = 0, keepdims = True),
            'std': self.outcome_unscaled.std(axis = 0, keepdims = True)
        }
        assert (self.scaling_params['std'] == 0.0).sum() == 0
        assert self.scaling_params['mean'].shape == (1, self.outcome_unscaled.shape[1])
        outcome_scaled = (self.outcome_unscaled - self.scaling_params['mean']) / self.scaling_params['std']
        return outcome_scaled
    
    def get_torch_dataset(self, subset='train') -> ProcessedDataset:
        """
        Get the processed dataset in the form of torch.utils.data.Dataset.
        Args:
            subset (str, optional): Subset of the data to get. Defaults to 'train'.
        Returns:
            ProcessedDataset: Processed dataset.
        """
        if subset == 'train':
            index = self.train_index
        elif subset == 'val':
            index = self.val_index
        elif subset == 'test':
            index = self.test_index
        else:
            raise ValueError(f'Invalid subset: {subset}.')

        #start from t = 1, transform to torch tensor
        Y = torch.from_numpy(self.outcome_scaled[index][:, 1:])
        prev_Y = torch.from_numpy(self.outcome_scaled[index][:, :-1])
        T_disc = torch.from_numpy(self.treatments[index][:, 1:, :])
        T_disc_prev = torch.from_numpy(self.treatments[index][:, :-1, :])
        X_static = torch.from_numpy(self.static_features[index])
        X_dynamic = torch.from_numpy(self.vitals_np[index][:, 1:])
        active_entries = torch.from_numpy(self.active_entries[index][:, 1:])

        return ProcessedDataset(Y = Y, prev_Y = prev_Y,
                                T_disc = T_disc, T_cont = None, T_disc_prev = T_disc_prev, T_cont_prev = None,
                                X_static = X_static, X_dynamic = X_dynamic, 
                                active_entries = active_entries, subset_name = subset)

    def treat_patient_factually(self, patient_ix: int, seed: int, te_model = 'min'):
        """
        Generate factually treated outcomes for a patient.
            patient_ix (int): Index of the patient in the dataset.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        Returns:
            pandas.DataFrame: DataFrame containing the patient's data with factually treated outcomes.
        Note:
            The treatment at hour index t is based on the information of time [t - window, t]
            And the treatment value assigned as t_prev at hour index t + 1, which means at hour 0 no treatment is applied.
            The treatment at hour index t affects the outcome starting from t + 1 (has a limited window of effect)
        """
        patient_df = self.all_vitals.loc[patient_ix].copy()
        rng = np.random.RandomState(seed)
        curr_treatment_cols = [f'{treatment.treatment_name}' for treatment in self.synthetic_treatments]

        for t in range(len(patient_df)):

            # Sampling treatments, based on previous factual outcomes
            treat_probas, treat_flags = self._sample_treatments_from_factuals(patient_df, t, rng)
            if t < max(patient_df.index.get_level_values('hours_in')):
                # Setting factuality flags
                patient_df.loc[t + 1, 'fact'] = 1.0
                # Setting factual sampled treatments
                patient_df.loc[t + 1, curr_treatment_cols] = {t: v for t, v in treat_flags.items()}
                if te_model == 'min':
                    # Treatments applications
                    if sum(treat_flags.values()) > 0:
                        # Treating each outcome separately
                        for outcome in self.synthetic_outcomes:
                            common_treatment_range, future_outcomes = self._combined_treating(patient_df, t, outcome, treat_probas,
                                                                                            treat_flags)
                            patient_df.loc[common_treatment_range, f'{outcome.outcome_name}'] = future_outcomes
                elif te_model == 'sum':
                    # The effect of current treatment at t are added to the future outcomes [t, .., t+window]
                    if sum(treat_flags.values()) > 0:
                        treatment_range, future_added_effects = self._add_treatment_effects(patient_df, t, treat_flags)
                        for outcome in self.synthetic_outcomes:
                            patient_df.loc[treatment_range, f'{self.outcome_col}'] += future_added_effects

        return patient_df
    
    def _sample_treatments_from_factuals(self, patient_df, t, rng=np.random.RandomState(None)):
            """
            Sample treatment for patient_df and time-step t
            Args:
                patient_df: DataFrame of patient
                t: Time-step
                rng: Random numbers generator (for parallelizing)

            Returns: Propensity scores, sampled treatments
            """
            factual_patient_df = patient_df[patient_df.fact.astype(bool)]
            treat_probas = {treatment.treatment_name: treatment.treatment_proba(factual_patient_df, t) for treatment in
                            self.synthetic_treatments}
            treatment_sample = {treatment_name: rng.binomial(1, treat_proba)[0] for treatment_name, treat_proba in
                                treat_probas.items()}
            return treat_probas, treatment_sample
    
    def _combined_treating(self, patient_df, t, outcome: SyntheticOutcomeGenerator, treat_probas: dict, treat_flags: dict):
        """
        Combing application of treatments
        Args:
            patient_df: DataFrame of patient
            t: Time-step
            outcome: Outcome to treat
            treat_probas: Propensity scores
            treat_flags: Treatment application flags

        Returns: Combined effect window, combined treated outcome
        """
        treatment_ranges, treated_future_outcomes = [], []
        influencing_treatments = self.treatment_outcomes_influence[outcome.outcome_name]
        influencing_treatments = \
            [treatment for treatment in self.synthetic_treatments if treatment.treatment_name in influencing_treatments]

        for treatment in influencing_treatments:
            treatment_range, treated_future_outcome = \
                treatment.get_treated_outcome(patient_df, t, outcome.outcome_name, treat_probas[treatment.treatment_name],
                                              bool(treat_flags[treatment.treatment_name]))

            treatment_ranges.append(treatment_range)
            treated_future_outcomes.append(treated_future_outcome)

        common_treatment_range, future_outcomes = SyntheticTreatment.combine_treatments(
            treatment_ranges,
            treated_future_outcomes,
            np.array([bool(treat_flags[treatment.treatment_name]) for treatment in influencing_treatments])
        )
        return common_treatment_range, future_outcomes
    
    def _add_treatment_effects(self, patient_df, t, treat_flags):
        """
        Add treatment effects to the future outcomes
        Args:
            patient_df: DataFrame of patient
            t: Time-step
            treat_flags: Treatment application flags
            
        Returns: Treatment range, future added effects of the combined treatments
        """
        if sum(treat_flags.values()) == 0:
            return [], 0
        influencing_treatments = [treatment for treatment in self.synthetic_treatments if treat_flags[treatment.treatment_name] > 0]
        treatment_ranges, future_added_effects = [set()], []
        for treatment in influencing_treatments:
            treatment_range, added_effects = treatment.get_added_effect(patient_df, t, treat_flags[treatment.treatment_name])
            treatment_ranges.append(set(treatment_range) if treat_flags[treatment.treatment_name] > 0 else set())
            future_added_effects.append(added_effects)
        
        common_treatment_range = set.union(*treatment_ranges)
        common_treatment_range = sorted(list(common_treatment_range))
        maximal_length = len(common_treatment_range)
        #first pad the shorter added_effects to maximal length and then stack them to sum up
        future_added_effects = np.stack([np.pad(added_effects, (0, maximal_length - len(added_effects))) 
                                         for added_effects in future_added_effects], axis = 1).sum(axis = 1)
        
        return common_treatment_range, future_added_effects                 
    
    def _simulate_counterfactuals(self, subset_name: str, treatment_seq: np.array, intv_name: str = 'intv'):
        """
        Simulate counterfactual outcomes based on a fixed sequence of treatments
        The counterfactual outcomes are then saved in the dictionary self.counterfactual_outcomes
        """
        self.treatments_seq = treatment_seq
        seeds = np.random.randint(0, 10000, size=self.n_units)
        subset_indices = self.index[subset_name]
        if self.parallel:
            par = Parallel(n_jobs=4, backend='loky')
            all_vitals_subset = par(delayed(self.treat_patient_counterfactually)(patient_ix, intv_name, seed=seed, te_model=self.te_model)
                                  for patient_ix, seed in tqdm(zip(subset_indices, seeds), total=len(subset_indices)))
        else:
            all_vitals_subset = [self.treat_patient_counterfactually(patient_ix, intv_name, seed, self.te_model) for patient_ix, seed in \
                                        tqdm(zip(subset_indices, seeds), total=len(subset_indices))]

        ctf_outcomes = pd.concat(all_vitals_subset, keys=subset_indices)[f'{intv_name}_{self.synth_outcome.outcome_name}']
        Y_ctf = ctf_outcomes.values.reshape((len(subset_indices), -1))[:, self.n_periods:]

        #normalize the data
        Y_ctf = (Y_ctf - self.scaling_params['mean'][:, self.n_periods:]) / self.scaling_params['std'][:, self.n_periods:]

        return Y_ctf
    
    def treat_patient_counterfactually(self, patient_ix: int, intv_name: str = 'intv', seed: int = None, te_model = 'min'):
        """
        Generate counterfactually treated outcomes for a patient using a rolling intervention approach.
        The intervention is self.treatments_seq, which is a fixed sequence of treatments.
    
        For time steps before t=1, factual observations are kept. Starting at t=0,
        for each intervention round, we intervene over a horizon of (m = projection_horizon + 1) time-steps.
        For example, if projection_horizon = 2, then with factual covariates at t=0, we simulate interventions (and 
        compute counterfactual outcomes) at t=0, 1, 2, then roll forward (intervene at 1,2,3, etc.).
        when t = k, we use the factual data from t=0 to t=k-1 plus factual covariate at t=k to simulate counterfactual outcomes at t=k, k+1, k+2.
    
        The counterfactual outcomes for each round are stored in new columns named by f'ctf_{outcome.outcome_name}'.
        Args:
            patient_ix: Index of patient
            intv_name: Name of the intervention (intv or base)
            seed: Random seed

        Returns: 
            DataFrame of patient, with counterfactual outcomes {intv_name}_y1, {intv_name}_y2,.. appended as new columns
        """
        patient_df = self.all_vitals.loc[patient_ix].copy()
        rng = np.random.RandomState(seed)
        #computes the number of active entries for each patient (<= max(hours_in)), 
        # patient_ae.sum() - 1 is the index of last active entry
        patient_ae = np.logical_not(patient_df.drop(columns=['fact']).isna().all(1)).astype(float)
        # Counterfactual sequence starts at last active entry minus projection horizon
        m = self.n_periods
        max_active_index = int(patient_ae.sum() - 1)

        treatment_cols = [f'{treatment.treatment_name}' for treatment in self.synthetic_treatments]
        intv_seq = self.treatments_seq

        #create columns for counterfactual outcomes
        for outcome in self.synthetic_outcomes:
            patient_df[f'{intv_name}_{outcome.outcome_name}'] = np.nan
        #for t in range(max(patient_df.index.get_level_values('hours_in')) - m + 1):
        for t in range(max_active_index - m + 2):
            assert (intv_seq.shape == (m, len(self.synthetic_treatments)))
            # --------------- Counterfactual treatment treatment trajectories ---------------
            buffer_patient_df = patient_df.copy()
            for time_ind in range(m):
                #intervene at t + time_ind
                future_treat_probs = {treatment.treatment_name: 1.0 for treatment in self.synthetic_treatments}
                future_treat_flags = {treatment.treatment_name: intv_seq[time_ind][j]
                                        for j, treatment in enumerate(self.synthetic_treatments)}

                # Setting treatment flags
                buffer_patient_df.loc[t + time_ind, treatment_cols] = \
                    {f'{t}': v for t, v in future_treat_flags.items()}

                # Treating each outcome separately, updaing counterfactual outcomes for t+index,..,t+m-1
                if te_model == 'min':
                    for outcome in self.synthetic_outcomes:
                        common_treatment_range, future_outcomes = \
                            self._combined_treating(buffer_patient_df, t + time_ind, outcome,
                                                    future_treat_probs, future_treat_flags)
                        buffer_patient_df.loc[common_treatment_range, outcome.outcome_name] = future_outcomes
                elif te_model == 'sum':
                    for outcome in self.synthetic_outcomes:
                        if sum(future_treat_flags.values()) > 0:
                            treatment_range, future_added_effects = self._add_treatment_effects(buffer_patient_df, t + time_ind, future_treat_flags)
                            buffer_patient_df.loc[treatment_range, outcome.outcome_name] += future_added_effects


            for outcome in self.synthetic_outcomes:
                patient_df.loc[t + m - 1, f'{intv_name}_{outcome.outcome_name}'] = buffer_patient_df.loc[t + m - 1, outcome.outcome_name]

        return patient_df



    def plot_timeseries(self, n_patients=5, mode='factual', seed = None):
        """
        Plotting patient trajectories
        Args:
            n_patients: Number of trajectories
            mode: factual / counterfactual
        """
        fig, ax = plt.subplots(nrows=4 * len(self.synthetic_outcomes) + len(self.synthetic_treatments), ncols=1, figsize=(15, 30))
        #Sample random patients
        if seed is not None:
            np.random.seed(seed)
        patient_ixs = np.random.choice(np.arange(self.n_units), n_patients, replace=False)
        for i, patient_ix in enumerate(patient_ixs):
            ax_ind = 0
            factuals = self.all_vitals.fillna(0.0).fact.astype(bool)
            for outcome in self.synthetic_outcomes:
                outcome_name = outcome.outcome_name
                ax[ax_ind].plot(self.all_vitals[factuals].loc[patient_ix, f'{outcome_name}_exog'].
                                groupby('hours_in').head(1).values)
                ax[ax_ind + 1].plot(self.all_vitals[factuals].loc[patient_ix, f'{outcome_name}_endo'].
                                    groupby('hours_in').head(1).values)
                ax[ax_ind + 2].plot(self.all_vitals[factuals].loc[patient_ix, f'{outcome_name}_untreated'].
                                    groupby('hours_in').head(1).values)
                if mode == 'factual':
                    ax[ax_ind + 3].plot(self.all_vitals.loc[patient_ix, outcome_name].values)
                elif mode == 'counterfactual':
                    color = next(ax[ax_ind + 3]._get_lines.prop_cycler)['color']
                    ax[ax_ind + 3].plot(self.all_vitals[factuals].loc[patient_ix, outcome_name].
                                        groupby('hours_in').head(1).index.get_level_values(1),
                                        self.all_vitals[factuals].loc[patient_ix, outcome_name].
                                        groupby('hours_in').head(1).values, color=color)
                    ax[ax_ind + 3].scatter(self.all_vitals.loc[patient_ix, outcome_name].index.get_level_values(1),
                                           self.all_vitals.loc[patient_ix, outcome_name].values, color=color, s=2)
                    # for traj_ix in self.all_vitals.loc[patient_ix].index.get_level_values(0):
                    #     ax[ax_ind + 3].plot(self.all_vitals.loc[(patient_ix, traj_ix), outcome_name].index,
                    #                         self.all_vitals.loc[(patient_ix, traj_ix), outcome_name].values, color=color,
                    #                         linewidth=0.05)

                ax[ax_ind].set_title(f'{outcome_name}_exog')
                ax[ax_ind + 1].set_title(f'{outcome_name}_endo')
                ax[ax_ind + 2].set_title(f'{outcome_name}_untreated')
                ax[ax_ind + 3].set_title(f'{outcome_name}')
                ax_ind += 4

            for treatment in self.synthetic_treatments:
                treatment_name = treatment.treatment_name
                ax[ax_ind].plot(self.all_vitals[factuals].loc[patient_ix, f'{treatment_name}'].
                                groupby('hours_in').head(1).values + 2 * i)
                ax[ax_ind].set_title(f'{treatment_name}')
                ax_ind += 1

        fig.suptitle(f'Time series from {self.name}', fontsize=16)
        plt.show()
    
    def insert_necessary_args_dml_rnn(self, args):
        args.dataset['n_treatments_disc'] = len(self.synthetic_treatments)
        args.dataset['n_treatments_cont'] = 0
        args.dataset['n_treatments'] = len(self.synthetic_treatments)
        args.dataset['n_units'] = self.n_units
        args.dataset['n_periods'] = self.n_periods
        args.dataset['sequence_length'] = self.sequence_length
        args.dataset['n_x'] = len(self.vital_cols)
        args.dataset['n_static'] = self.static_features.shape[1]
        return

    def get_confounding_strength(self):
            return self.synthetic_treatments[0].conf_outcome_weight

    
    