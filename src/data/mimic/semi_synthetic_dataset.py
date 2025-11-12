import pandas as pd
import numpy as np
import logging
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from torch.utils.data import Dataset
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
from typing import List
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy

#from src import ROOT_PATH
import os, sys
root_dir = os.path.abspath(os.getcwd())
sys.path.append(root_dir)
from src.data.mimic.load_data import load_mimic3_data_raw
from src.data.mimic.utils import sigmoid, SplineTrendsMixture
from src.data.mimic.real_dataset import MIMIC3RealDataset
#from src.data.dataset_collection import SyntheticDatasetCollection

logger = logging.getLogger(__name__)


class SyntheticOutcomeGenerator:
    """
    Generator of synthetic outcome
    """

    def __init__(self,
                 exogeneous_vars: List[str],
                 exog_dependency: callable,
                 exog_weight: float,
                 endo_dependency: callable,
                 endo_rand_weight: float,
                 endo_spline_weight: float,
                 outcome_name: str):
        """
        Args:
            exogeneous_vars: List of time-varying covariates
            exog_dependency: Callable function of exogeneous_vars (f_Z)
            exog_weight: alpha_f
            endo_dependency: Callable function of endogenous variables (g)
            endo_rand_weight: alpha_g
            endo_spline_weight: alpha_S
            outcome_name: Name of the outcome variable j
        """
        self.exogeneous_vars = exogeneous_vars
        self.exog_dependency = exog_dependency
        self.exog_weight = exog_weight
        self.endo_rand_weight = endo_rand_weight
        self.endo_spline_weight = endo_spline_weight
        self.endo_dependency = endo_dependency
        self.outcome_name = outcome_name

    def simulate_untreated(self, all_vitals: pd.DataFrame, static_features: pd.DataFrame):
        """
        Simulate untreated outcomes (Z)
        Z  = endogenous effect + exogenous effect (depend on the vitals) + noise
        Args:
            all_vitals: Time-varying covariates (as exogeneous vars)
            static_features: Static covariates (as exogeneous vars)
        """
        logger.info(f'Simulating untreated outcome {self.outcome_name}')
        user_sizes = all_vitals.groupby(level='subject_id', sort=False).size()

        # Exogeneous dependency
        all_vitals[f'{self.outcome_name}_exog'] = self.exog_weight * self.exog_dependency(all_vitals[self.exogeneous_vars].values)

        # Endogeneous dependency + B-spline trend
        time_range = np.arange(0, user_sizes.max())
        y_endo_rand = self.endo_dependency(time_range, len(user_sizes))
        y_endo_splines = SplineTrendsMixture(n_patients=len(user_sizes), max_time=user_sizes.max())(time_range)
        y_endo_full = self.endo_rand_weight * y_endo_rand + self.endo_spline_weight * y_endo_splines

        all_vitals[f'{self.outcome_name}_endo'] = \
            np.array([value for (i, l) in enumerate(user_sizes) for value in y_endo_full[i, :l]]).reshape(-1, 1)

        # Untreated outcome
        all_vitals[f'{self.outcome_name}_untreated'] = \
            all_vitals[f'{self.outcome_name}_exog'] + all_vitals[f'{self.outcome_name}_endo']

        # Placeholder for treated outcome
        all_vitals[f'{self.outcome_name}'] = all_vitals[f'{self.outcome_name}_untreated'].copy()


class SyntheticTreatment:
    """
    Generator of synthetic treatment
    """

    def __init__(self,
                 confounding_vars: List[str],
                 confounder_outcomes: List[str],
                 confounding_dependency: callable,
                 window: float,
                 conf_outcome_weight: float,
                 conf_vars_weight: float,
                 bias: float,
                 full_effect: float,
                 effect_window: float,
                 treatment_name: str,
                 post_nonlinearity: callable = None,
                 **kwargs):
        """
        Args:
            confounding_vars: Confounding time-varying covariates (from all_vitals)
            confounder_outcomes: Confounding previous outcomes
            confounding_dependency: Callable function of confounding_vars (f_Y)
            window: Window of averaging of confounding previous outcomes (T_l)
            conf_outcome_weight: gamma_Y
            conf_vars_weight: gamma_X
            bias: constant bias
            full_effect: beta
            effect_window: w_l
            treatment_name: Name of treatment l
            post_nonlinearity: Post non-linearity after sigmoid
        """
        self.confounding_vars = confounding_vars
        self.confounder_outcomes = confounder_outcomes
        self.confounding_dependency = confounding_dependency
        self.treatment_name = treatment_name
        self.post_nonlinearity = post_nonlinearity
        if 'scale_function' in kwargs:
            self.scale_function = kwargs['scale_function']
            if self.scale_function['type'] == 'tanh':
                self.coeff = np.array(self.scale_function['coefficients'])
                self.kappa = lambda x: np.tanh(np.dot(x, self.coeff)) + 1
            else:
                self.kappa = lambda x: 1.0
        else:
            self.scale_function = None

        # Parameters
        self.window = window
        self.conf_outcome_weight = conf_outcome_weight
        self.conf_vars_weight = conf_vars_weight
        self.bias = bias

        self.full_effect = full_effect
        self.effect_window = effect_window

    def treatment_proba(self, patient_df, t):
        """
        Calculates propensity score for patient_df and time-step t
        Args:
            patient_df: DataFrame of patient
            t: Time-step

        Returns: propensity score
        """
        t_start = max(0, t - self.window)

        agr_range = np.arange(t_start, t + 1)
        avg_y = patient_df.loc[agr_range, self.confounder_outcomes].values.mean()
        x = patient_df.loc[t, self.confounding_vars].values.reshape(1, -1)
        f_x = self.confounding_dependency(x)
        treat_proba = sigmoid(self.bias + self.conf_outcome_weight * avg_y + self.conf_vars_weight * f_x).flatten()
        if self.post_nonlinearity is not None:
            treat_proba = self.post_nonlinearity(treat_proba)
        return treat_proba

    def get_treated_outcome(self, patient_df, t, outcome_name, treat_proba=1.0, treat=True):
        """
        !!For calculation of min aggregation of treatment effects!!
        Calculate future outcome under treatment, applied at the time-step t
        Args:
            patient_df: DataFrame of patient
            t: Time-step
            outcome_name: Name of the outcome variable j
            treat_proba: Propensity scores of treatment
            treat: Treatment application flag

        Returns: Effect window, treated outcome
        """
        scaled_effect = self.full_effect * treat_proba
        #t_stop should be limited by the maximal index of non NaN values instead of the maximal index of the dataframe
        #t_stop = min(max(patient_df.index.get_level_values('hours_in')), t + self.effect_window)
        # get index of last active entry of outcome (not Nan)
        max_acitve_index = np.logical_not(patient_df[outcome_name].isna()).sum() - 1
        t_stop = min(max_acitve_index + 1, t + self.effect_window)

        treatment_range = np.arange(t, t_stop)
        treatment_range_rel = treatment_range - t + 1

        future_outcome = patient_df.loc[treatment_range, outcome_name]
        if treat:
            future_outcome += scaled_effect / treatment_range_rel ** 0.5
        return treatment_range, future_outcome
    
    def get_added_effect(self, patient_df, t, treat_flag = True):
        """
        Calculate the added effect of the treatment at time step t
        Args:
            patient_df: DataFrame of patient
            t: Time-step
            treat_flag: Treatment application flag

        Returns: Effect window, treated effect of this single treatment to be added to the outcome

        added_effects = [kappa(x_{s}) * beta / sqrt(s - t + 1) * I_{treat_flag} for s in [t, t + w_l]]
        """
        max_acitve_index = np.logical_not(patient_df[self.treatment_name].isna()).sum() - 1
        t_stop = min(max_acitve_index + 1, t + self.effect_window)

        treatment_range = np.arange(t, t_stop)
        treatment_range_rel = treatment_range - t + 1

        added_effects = 0.0
        if treat_flag:
            # get confounding variables
            x_conf = patient_df.loc[treatment_range, self.confounding_vars].values #shape (w_l, len(confounding_vars))
            kappa = self.kappa(x_conf) #shape (w_l,)
            added_effects = kappa * self.full_effect / (treatment_range_rel ** 0.5)
        return treatment_range, added_effects


    @staticmethod
    def combine_treatments(treatment_ranges, treated_future_outcomes, treat_flags):
        """
        Min combining of different treatment effects
        Args:
            treatment_ranges: List of numpy arrays of treatment ranges
            #Example: (when t = 3, and two discrete binary treatments are applied)
            [
                array([3, 4, 5, 6, 7]), 
                array([3, 4, 5, 6])
            ]

            treated_future_outcomes: Future outcomes (in list of pandas.Series) under each individual treatment
            [ # List of treated outcomes under different treatments (example for 2 treatments with different effects and ranges)
                hours_in
                    3   -2.315110
                    4   -2.014812
                    5   -1.431916
                    6   -1.290260
                    7   -1.011298
                    Name: y1, dtype: float64, 
                hours_in
                    3   -2.115110
                    4   -1.873390
                    5   -1.316446
                    6   -1.190260
                    Name: y1, dtype: float64
            ]
            treat_flags: Treatment application flags,  example: array([ True,  True])

        Returns: Combined effect window, combined treated outcome
        """
        treated_future_outcomes = pd.concat(treated_future_outcomes, axis=1)
        if treat_flags.any():  # Min combining all the treatments
            common_treatment_range = [set(treatment_range) for i, treatment_range in enumerate(treatment_ranges) if
                                      treat_flags[i]]
            common_treatment_range = set.union(*common_treatment_range)
            common_treatment_range = sorted(list(common_treatment_range))
            treated_future_outcomes = treated_future_outcomes.loc[common_treatment_range]
            # When doing the np.nanmin operation, there is possibility that all the values are nan, which will result in nan warning,
            # However, this should not happend, and we would like to create a mechanism to catch this error
            # treated_future_outcomes['agg'] = np.nanmin(treated_future_outcomes.iloc[:, treat_flags].values, axis=1)
            treated_outcome_flag = treated_future_outcomes.iloc[:, treat_flags].values
            # Check for every row if all the values are nan
            if np.isnan(treated_outcome_flag).all(1).any():
                #find time steps where all the values are nan
                nan_time_steps = np.where(np.isnan(treated_outcome_flag).all(1))[0]
                not_all_nan_time_steps = np.where(~np.isnan(treated_outcome_flag).any(1))[0]
                print('All values are nan in the treated outcome at time steps:', nan_time_steps)
                treated_future_outcomes['agg'] = np.nan
                if len(not_all_nan_time_steps) > 0:
                    #print('The treated outcomes are not nan at time steps:', not_all_nan_time_steps)
                    treated_future_outcomes.loc[not_all_nan_time_steps, 'agg'] = np.nanmin(treated_outcome_flag[not_all_nan_time_steps], axis = 1)
            else:
                treated_future_outcomes['agg'] = np.nanmin(treated_outcome_flag, axis = 1)
        else:  # No treatment is applied
            common_treatment_range = treatment_ranges[0]
            treated_future_outcomes['agg'] = treated_future_outcomes.iloc[:, 0]  # Taking untreated outcomes
        return common_treatment_range, treated_future_outcomes['agg']


class MIMIC3SyntheticDataset(Dataset):
    """
    Pytorch-style semi-synthetic MIMIC-III dataset
    """
    def __init__(self,
                 all_vitals: pd.DataFrame,
                 static_features: pd.DataFrame,
                 synthetic_outcomes: List[SyntheticOutcomeGenerator],
                 synthetic_treatments: List[SyntheticTreatment],
                 treatment_outcomes_influence: dict,
                 subset_name: str,
                 mode='factual',
                 parallel = True):
        """
        Args:
            all_vitals: DataFrame with vitals (time-varying covariates); multiindex by (patient_id, timestep)
            static_features: DataFrame with static features
            synthetic_outcomes: List of SyntheticOutcomeGenerator
            synthetic_treatments: List of SyntheticTreatment
            treatment_outcomes_influence: dict with treatment-outcomes influences
            subset_name: train / val / test
            mode: factual / counterfactual_one_step / counterfactual_treatment_seq
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            treatments_seq: Fixed (non-random) treatment sequecne for multiple-step-ahead prediction
            n_treatments_seq: Number of random trajectories after rolling origin in test subset
        """

        self.subset_name = subset_name
        self.all_vitals = all_vitals.copy()
        self.vital_cols = all_vitals.columns
        self.static_features = static_features
        self.synthetic_outcomes = synthetic_outcomes
        self.synthetic_treatments = synthetic_treatments
        self.treatment_outcomes_influence = treatment_outcomes_influence

        self.prev_treatment_cols = [f'{treatment.treatment_name}_prev' for treatment in self.synthetic_treatments]
        self.outcome_cols = [outcome.outcome_name for outcome in self.synthetic_outcomes]
        self.treatment_options = [0.0, 1.0]
        self.parallel = parallel
        self.counterfactual_outcomes = dict()

        # Sampling untreated outcomes
        for outcome in self.synthetic_outcomes:
            outcome.simulate_untreated(self.all_vitals, static_features)
        # Placeholders
        for treatment in self.synthetic_treatments:
            self.all_vitals[f'{treatment.treatment_name}_prev'] = 0.0
        self.all_vitals['fact'] = np.nan
        self.all_vitals.loc[(slice(None), 0), 'fact'] = 1.0  # First observation is always factual
        user_sizes = self.all_vitals.groupby(level='subject_id', sort=False).size()

        # Treatment application
        self.seeds = np.random.randint(np.iinfo(np.int32).max, size=len(static_features))
        par = Parallel(n_jobs=4, backend='loky')
        # par = Parallel(n_jobs=4, backend='loky')
        logger.info(f'Simulating factual treatments and applying them to outcomes.')
        if mode == 'factual':
            if parallel:
                self.all_vitals = par(delayed(self.treat_patient_factually)(patient_ix, seed)
                               for patient_ix, seed in tqdm(zip(static_features.index, self.seeds), total=len(static_features)))
            else:
                #Process all the patients sequentially
                self.all_vitals = [self.treat_patient_factually(patient_ix, seed) for patient_ix, seed in \
                                            tqdm(zip(static_features.index, self.seeds), total=len(static_features))]
            logger.info('Concatenating all the trajectories together.')
        else:
            raise NotImplementedError()

        #Each single patient dataframe has new columns: 
        # ['y1_exog', 'y1_endo', 'y1_untreated', 'y1', 'y2_exog'.., 'y2', 'fact', 't1_prev', 't2_prev']
        self.all_vitals = pd.concat(self.all_vitals, keys=static_features.index)

        # Padding with nans
        self.all_vitals = self.all_vitals.unstack(fill_value=np.nan, level=0).stack(dropna=False).swaplevel(0, 1).sort_index()
        static_features = static_features.sort_index()
        static_features = static_features.values

        # Conversion to np arrays
        treatments = self.all_vitals[self.prev_treatment_cols].fillna(0.0).values.reshape((-1, max(user_sizes),
                                                                                      len(self.prev_treatment_cols)))
        vitals = self.all_vitals[self.vital_cols].fillna(0.0).values.reshape((-1, max(user_sizes), len(self.vital_cols)))
        outcomes_unscaled = self.all_vitals[self.outcome_cols].fillna(0.0).values.reshape((-1, max(user_sizes), len(self.outcome_cols)))
        active_entries = (~self.all_vitals.isna().all(1)).astype(float)
        active_entries = active_entries.values.reshape((-1, max(user_sizes), 1))
        user_sizes = np.squeeze(active_entries.sum(1))

        logger.info(f'Shape of exploded vitals: {vitals.shape}.')

        self.data = {
            'sequence_lengths': user_sizes - 1,
            'prev_treatments': treatments[:, :-1, :],
            'vitals': vitals[:, 1:, :],
            'next_vitals': vitals[:, 2:, :],
            'current_treatments': treatments[:, 1:, :],
            'static_features': static_features,
            'active_entries': active_entries[:, 1:, :],
            'unscaled_outputs': outcomes_unscaled[:, 1:, :],
            'prev_unscaled_outputs': outcomes_unscaled[:, :-1, :],
        }

        self.processed = False  # Need for normalisation of newly generated outcomes
        self.processed_sequential = False
        self.processed_autoregressive = False

        self.norm_const = 1.0

    
    def simulate_counterfactuals(self, treatments_seq: np.array, n_periods: int, intv_name: str = 'intv'):
        """
        Simulate counterfactual outcomes based on a fixed sequence of treatments
        The counterfactual outcomes are then saved in the dictionary self.counterfactual_outcomes
        """
        self.treatments_seq = treatments_seq
        self.projection_horizon = n_periods - 1
        self.n_periods = n_periods
        par = Parallel(n_jobs=4, backend='loky')
        if self.parallel:
                self.all_vitals = par(delayed(self.treat_patient_counterfactually)(patient_ix, intv_name, seed=seed)
                                  for patient_ix, seed in tqdm(zip(self.static_features.index, self.seeds), total=len(self.static_features)))
        else:
            self.all_vitals = [self.treat_patient_counterfactually(patient_ix, intv_name, seed) for patient_ix, seed in \
                                        tqdm(zip(self.static_features.index, self.seeds), total=len(self.static_features))]

        self.all_vitals = pd.concat(self.all_vitals, keys=self.static_features.index)

        #convert the counterfactual outcomes to np arrays
        self.counterfactual_outcomes[intv_name] = self.all_vitals[[f'intv_{outcome.outcome_name}' for outcome in self.synthetic_outcomes]].replace(np.nan, 0.0).values
        return self.all_vitals
    
    
    def plot_timeseries(self, n_patients=5, mode='factual'):
        """
        Plotting patient trajectories
        Args:
            n_patients: Number of trajectories
            mode: factual / counterfactual
        """
        fig, ax = plt.subplots(nrows=4 * len(self.synthetic_outcomes) + len(self.synthetic_treatments), ncols=1, figsize=(15, 30))
        for i, patient_ix in enumerate(self.all_vitals.index.levels[0][:n_patients]):
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
                ax[ax_ind].plot(self.all_vitals[factuals].loc[patient_ix, f'{treatment_name}_prev'].
                                groupby('hours_in').head(1).values + 2 * i)
                ax[ax_ind].set_title(f'{treatment_name}')
                ax_ind += 1

        fig.suptitle(f'Time series from {self.subset_name}', fontsize=16)
        plt.show()

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

    def treat_patient_factually(self, patient_ix: int, seed: int = None):
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
        prev_treatment_cols = [f'{treatment.treatment_name}_prev' for treatment in self.synthetic_treatments]

        for t in range(len(patient_df)):

            # Sampling treatments, based on previous factual outcomes
            treat_probas, treat_flags = self._sample_treatments_from_factuals(patient_df, t, rng)

            if t < max(patient_df.index.get_level_values('hours_in')):
                # Setting factuality flags
                patient_df.loc[t + 1, 'fact'] = 1.0

                # Setting factual sampled treatments
                patient_df.loc[t + 1, prev_treatment_cols] = {f'{t}_prev': v for t, v in treat_flags.items()}

                # Treatments applications
                if sum(treat_flags.values()) > 0:

                    # Treating each outcome separately
                    for outcome in self.synthetic_outcomes:
                        common_treatment_range, future_outcomes = self._combined_treating(patient_df, t, outcome, treat_probas,
                                                                                          treat_flags)
                        patient_df.loc[common_treatment_range, f'{outcome.outcome_name}'] = future_outcomes

        return patient_df

    def treat_patient_counterfactually(self, patient_ix: int, intv_name: str = 'intv', seed: int = None):
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
        m = self.projection_horizon + 1
        max_active_index = int(patient_ae.sum() - 1)

        prev_treatment_cols = [f'{treatment.treatment_name}_prev' for treatment in self.synthetic_treatments]
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
                buffer_patient_df.loc[t + time_ind, prev_treatment_cols] = \
                    {f'{t}_prev': v for t, v in future_treat_flags.items()}

                # Treating each outcome separately, updaing counterfactual outcomes for t+index,..,t+m-1
                for outcome in self.synthetic_outcomes:
                    common_treatment_range, future_outcomes = \
                        self._combined_treating(buffer_patient_df, t + time_ind, outcome,
                                                future_treat_probs, future_treat_flags)
                    buffer_patient_df.loc[common_treatment_range, outcome.outcome_name] = future_outcomes
            for outcome in self.synthetic_outcomes:
                patient_df.loc[t + m - 1, f'{intv_name}_{outcome.outcome_name}'] = buffer_patient_df.loc[t + m - 1, outcome.outcome_name]

        return patient_df

    def get_scaling_params(self):
        outcome_cols = [outcome.outcome_name for outcome in self.synthetic_outcomes]
        logger.info('Performing normalisation.')
        scaling_params = {
            'output_means': self.all_vitals[outcome_cols].mean(0).to_numpy(),
            'output_stds': self.all_vitals[outcome_cols].std(0).to_numpy(),
        }
        return scaling_params

    def process_data(self, scaling_params):
        """
        Pre-process dataset for one-step-ahead prediction
        Args:
            scaling_params: dict of standard normalization parameters (calculated with train subset)
        """
        if not self.processed:
            logger.info(f'Processing {self.subset_name} dataset before training')

            self.data['outputs'] = (self.data['unscaled_outputs'] - scaling_params['output_means']) / \
                scaling_params['output_stds']
            self.data['prev_outputs'] = (self.data['prev_unscaled_outputs'] - scaling_params['output_means']) / \
                scaling_params['output_stds']

            # if self.autoregressive:
            #     self.data['vitals'] = np.concatenate([self.data['vitals'], self.data['prev_outputs']], axis=2)

            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            self.scaling_params = scaling_params
            self.processed = True
        else:
            logger.info(f'{self.subset_name} Dataset already processed')

        return self.data

#Instead of using the syntheticdatasetcollection, we can use the mimic3syntheticdataset directly
#First we need to create the synthetic outcomes and treatments
#Note that the counterfactuals needs to be computed based on the same factual data
def prepare_synthetic_mimic_dataset(args_dataset:DictConfig):
    # save parameters in args
    seed = args_dataset.seed
    path = args_dataset.path
    min_seq_length = args_dataset.min_seq_length
    max_seq_length = args_dataset.max_seq_length
    max_number = args_dataset.max_number
    n_periods = args_dataset.n_periods
    #load dataset raw
    all_vitals, static_features = load_mimic3_data_raw(path, min_seq_length=min_seq_length,
                                                       max_seq_length=max_seq_length, max_number=max_number,
                                                       data_seed=seed)
    
    #define the synthetic outcomes and treatments
    #instantiate exog_dependency and the synth_outcome
    exog_dependency = instantiate(args_dataset.synth_outcome.exog_dependency)
    synth_outcome = SyntheticOutcomeGenerator(exogeneous_vars=args_dataset.synth_outcome.exogeneous_vars,
                                                exog_dependency=exog_dependency,
                                                exog_weight=args_dataset.synth_outcome.exog_weight,
                                                endo_dependency=instantiate(args_dataset.synth_outcome.endo_dependency),
                                                endo_rand_weight=args_dataset.synth_outcome.endo_rand_weight,
                                                endo_spline_weight=args_dataset.synth_outcome.endo_spline_weight,
                                                outcome_name=args_dataset.synth_outcome.outcome_name)
    #instantiate the list of synth_treatments
    synth_treatments_list = [instantiate(treatment) for treatment in args_dataset.synth_treatments_list]
    #instantiate the treatment_outcomes_influence
    treatment_outcomes_influence = args_dataset.treatment_outcomes_influence
    #instantiate the MIMIC3SyntheticDataset and generate factual data
    dataset = MIMIC3SyntheticDataset(all_vitals, static_features, 
                                       [synth_outcome], synth_treatments_list,
                                       treatment_outcomes_influence, 'full',
                                       mode = 'factual', parallel=args_dataset.parallel)
    return dataset


if __name__ == '__main__':
    dataset_config_path = r'C:\Users\User\OneDrive\Documents\Thesis\Neural-R-Learner\config\dataset\mimic_synthetic_debug.yaml'
    args_dataset = OmegaConf.load(dataset_config_path).dataset
    treatments_seq = np.array([
        [1, 0],[0, 1],[1, 1]
    ])
    #dataset = prepare_synthetic_mimic_dataset(args_dataset, mode = 'counterfactual_treatment_seq', treatments_seq=treatments_seq)
    dataset = prepare_synthetic_mimic_dataset(args_dataset)
    dataset.simulate_counterfactuals(treatments_seq, n_periods=args_dataset.n_periods)


    

