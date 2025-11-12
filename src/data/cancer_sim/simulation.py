# -*- coding: utf-8 -*-
"""
[Treatment Effects with RNNs] cancer_simulation
Created on 2/4/2018 8:14 AM

Medically realistic data simulation for small-cell lung cancer based on Geng et al 2017.
URL: https://www.nature.com/articles/s41598-017-13646-z

Notes:
- Simulation time taken to be in days

@author: limsi
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import truncnorm  # we need to sample from truncated normal distributions

sns.set()

logger = logging.getLogger(__name__)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulation Constants

# Spherical calculations - tumours assumed to be spherical per Winer-Muram et al 2002.
# URL:
# https://pubs.rsna.org/doi/10.1148/radiol.2233011026?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub%3Dpubmed


def calc_volume(diameter):
    return 4 / 3 * np.pi * (diameter / 2) ** 3


def calc_diameter(volume):
    return ((volume / (4 / 3 * np.pi)) ** (1 / 3)) * 2


# Tumour constants per
TUMOUR_CELL_DENSITY = 5.8 * 10 ** 8  # cells per cm^3
TUMOUR_DEATH_THRESHOLD = calc_volume(13)  # assume spherical

# Patient cancer stage. (mu, sigma, lower bound, upper bound) - for lognormal dist
tumour_size_distributions = {'I': (1.72, 4.70 * 0.1, 1.0, 3.0),
                             'II': (1.96, 1.63* 0.1, 1.0, 3.0),
                             'IIIA': (1.91, 9.40* 0.1, 1.0, 4.0),
                             'IIIB': (2.76, 6.87* 0.1, 1.0, 4.0),
                             'IV': (3.86, 8.82* 0.1, 1.0, 5.0)}  # 13.0 is the death condition

# Observations of stage proportions taken from Detterbeck and Gibson 2008
# - URL: http://www.jto.org/article/S1556-0864(15)33353-0/fulltext#cesec50\
cancer_stage_observations = {'I': 1432,
                             "II": 128,
                             "IIIA": 1306,
                             "IIIB": 7248,
                             "IV": 12840}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulation Functions


def generate_params(num_patients, chemo_coeff, radio_coeff, window_size, lag = 0, lag_y = 10):
    """
    Get original patient-specific simulation parameters, and add extra ones to control confounding

    :param num_patients: Number of patients to simulate
    :param chemo_coeff: Bias on action policy for chemotherapy assignments
    :param radio_activation_group: Bias on action policy for chemotherapy assignments
    :return: dict of parameters
    """

    basic_params = get_standard_params(num_patients)
    patient_types = basic_params['patient_types']
    # tumour_stage_centres = [s for s in cancer_stage_observations if 'IIIA' not in s]
    # tumour_stage_centres.sort()

    # Parameters controlling sigmoid application probabilities

    D_MAX = calc_diameter(TUMOUR_DEATH_THRESHOLD)
    basic_params['chemo_sigmoid_intercepts'] = np.array([D_MAX / 2.0 for _ in patient_types])
    basic_params['radio_sigmoid_intercepts'] = np.array([D_MAX / 2.0 for _ in patient_types])

    basic_params['chemo_sigmoid_betas'] = np.array([chemo_coeff / D_MAX for _ in patient_types])
    basic_params['radio_sigmoid_betas'] = np.array([radio_coeff / D_MAX for _ in patient_types])

    basic_params['window_size'] = window_size
    basic_params['lag'] = lag
    basic_params['lag_y'] = lag_y

    return basic_params


def get_standard_params(num_patients):  # additional params
    """
    Simulation parameters from the Nature article + adjustments for static variables

    :param num_patients: Number of patients to simulate
    :return: simulation_parameters: Initial volumes + Static variables (e.g. response to treatment); randomly shuffled
    """
    # Calculate cancer stage proportions
    TOTAL_OBS = sum(cancer_stage_observations.values())
    cancer_stage_proportions = {k: cancer_stage_observations[k] / TOTAL_OBS for k in cancer_stage_observations}

    # Sample initial stages
    possible_stages = list(tumour_size_distributions.keys())
    possible_stages.sort()
    initial_stages = np.random.choice(possible_stages, num_patients,
                                      p=[cancer_stage_proportions[k] for k in possible_stages])

    # Simulate initial tumor volumes
    output_initial_diam = []
    patient_sim_stages = []
    for stg in possible_stages:
        count = np.sum((initial_stages == stg) * 1)
        mu, sigma, lower_bound, upper_bound = tumour_size_distributions[stg]
        lower_bound = (np.log(lower_bound) - mu) / sigma
        upper_bound = (np.log(upper_bound) - mu) / sigma
        norm_rvs = truncnorm.rvs(lower_bound, upper_bound, size=count)
        initial_volume_by_stage = np.exp((norm_rvs * sigma) + mu)
        output_initial_diam += list(initial_volume_by_stage)
        patient_sim_stages += [stg for i in range(count)]

    # Simulate dynamic parameters
    K = calc_volume(30)
    ALPHA_BETA_RATIO = 10
    ALPHA_RHO_CORR = 0.87
    parameter_lower_bound = 0.0
    parameter_upper_bound = np.inf
    rho_params = (7 * 10 ** -5, 7.23 * 10 ** -7)
    alpha_params = (0.0398, 0.0068)
    beta_c_params = (0.028, 0.0007)
    alpha_rho_cov = np.array([[alpha_params[1] ** 2, ALPHA_RHO_CORR * alpha_params[1] * rho_params[1]],
                              [ALPHA_RHO_CORR * alpha_params[1] * rho_params[1], rho_params[1] ** 2]])
    alpha_rho_mean = np.array([alpha_params[0], rho_params[0]])
    simulated_params = []
    while len(simulated_params) < num_patients:
        param_holder = np.random.multivariate_normal(alpha_rho_mean, alpha_rho_cov, size=num_patients)
        for i in range(param_holder.shape[0]):
            if param_holder[i, 0] > parameter_lower_bound and param_holder[i, 1] > parameter_lower_bound:
                simulated_params.append(param_holder[i, :])

    # Adjust parameters based on patient types
    possible_patient_types = [1, 2, 3]
    patient_types = np.random.choice(possible_patient_types, num_patients)
    chemo_mean_adjustments = np.array([0.0 if i < 3 else 0.1 for i in patient_types])
    radio_mean_adjustments = np.array([0.0 if i > 1 else 0.1 for i in patient_types])
    simulated_params = np.array(simulated_params)[:num_patients, :]
    alpha_adjustments = alpha_params[0] * radio_mean_adjustments
    alpha = simulated_params[:, 0] + alpha_adjustments
    rho = simulated_params[:, 1]
    beta = alpha / ALPHA_BETA_RATIO
    beta_c_adjustments = beta_c_params[0] * chemo_mean_adjustments
    beta_c = beta_c_params[0] + beta_c_params[1] * truncnorm.rvs(
        (parameter_lower_bound - beta_c_params[0]) / beta_c_params[1],
        (parameter_upper_bound - beta_c_params[0]) / beta_c_params[1],
        size=num_patients) + beta_c_adjustments

    # Compile and shuffle parameters
    output_holder = {'patient_types': patient_types,
                     'initial_stages': np.array(patient_sim_stages),
                     'initial_volumes': calc_volume(np.array(output_initial_diam)),
                     'alpha': alpha,
                     'rho': rho,
                     'beta': beta,
                     'beta_c': beta_c,
                     'K': np.array([K for _ in range(num_patients)]),
                     }
    idx = [i for i in range(num_patients)]
    np.random.shuffle(idx)
    output_params = {}
    for k in output_holder:
        output_params[k] = output_holder[k][idx]

    return output_params


def simulate_factual(simulation_params, seq_length):
    """
    Simulation of factual patient trajectories (for train and validation subset)

    :param simulation_params: Parameters of the simulation
    :param seq_length: Maximum trajectory length
    :param assigned_actions: Fixed non-random treatment assignment policy, if None - standard biased random assignment is applied
    :return: simulated data dict
    """

    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    # radio_days = np.array([i + 1 for i in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params['initial_stages']
    initial_volumes = simulation_params['initial_volumes']
    alphas = simulation_params['alpha']
    rhos = simulation_params['rho']
    betas = simulation_params['beta']
    beta_cs = simulation_params['beta_c']
    Ks = simulation_params['K']
    patient_types = simulation_params['patient_types']
    window_size = simulation_params['window_size']
    lag = simulation_params['lag']
    lag_y = simulation_params['lag_y']
    seq_length += lag_y

    # Coefficients for treatment assignment probabilities
    chemo_sigmoid_intercepts = simulation_params['chemo_sigmoid_intercepts']
    radio_sigmoid_intercepts = simulation_params['radio_sigmoid_intercepts']
    chemo_sigmoid_betas = simulation_params['chemo_sigmoid_betas']
    radio_sigmoid_betas = simulation_params['radio_sigmoid_betas']

    num_patients = initial_stages.shape[0]

    # Commence Simulation
    cancer_volume = np.zeros((num_patients, seq_length))
    chemo_dosage = np.zeros((num_patients, seq_length))
    radio_dosage = np.zeros((num_patients, seq_length))
    chemo_application_point = np.zeros((num_patients, seq_length))
    radio_application_point = np.zeros((num_patients, seq_length))
    sequence_lengths = np.zeros(num_patients)
    death_flags = np.zeros((num_patients, seq_length))
    recovery_flags = np.zeros((num_patients, seq_length))
    chemo_probabilities = np.zeros((num_patients, seq_length))
    radio_probabilities = np.zeros((num_patients, seq_length))

    noise_terms = 0.01 * np.random.randn(num_patients, seq_length)  # 5% cell variability
    recovery_rvs = np.random.rand(num_patients, seq_length)

    chemo_application_rvs = np.random.rand(num_patients, seq_length)
    radio_application_rvs = np.random.rand(num_patients, seq_length)

    # Run actual simulation
    for i in tqdm(range(num_patients), total=num_patients):

        # logging.info("Simulating patient {} of {}".format(i + 1, num_patients))
        noise = noise_terms[i]

        # initial values
        cancer_volume[i, 0] = initial_volumes[i]
        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]

        # Setup cell volume
        b_death = False
        b_recover = False
        for t in range(1, seq_length):

            current_chemo_dose = 0.0
            previous_chemo_dose = chemo_dosage[i, t - 1]

            # Action probabilities + death or recovery simulations
            if t >= lag:
                cancer_volume_used = cancer_volume[i, max(t - window_size - lag, 0):max(t - lag, 0)]
            else:
                cancer_volume_used = np.zeros((1, ))
            cancer_diameter_used = np.array([calc_diameter(vol) for vol in cancer_volume_used]).mean()  # mean diameter over 15 days
            cancer_metric_used = cancer_diameter_used

            if t >= lag_y:
                radio_prob = (1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i] * (cancer_metric_used - radio_sigmoid_intercepts[i]))))
                chemo_prob = (1.0 / (1.0 + np.exp(- chemo_sigmoid_betas[i] * (cancer_metric_used - chemo_sigmoid_intercepts[i]))))
            else:
                radio_prob = 0.0
                chemo_prob = 0.0
            chemo_probabilities[i, t] = chemo_prob
            radio_probabilities[i, t] = radio_prob

            # Action application
            if radio_application_rvs[i, t] < radio_prob:
                radio_application_point[i, t] = 1
                radio_dosage[i, t] = radio_amt[0]

            if chemo_application_rvs[i, t] < chemo_prob:
                # Apply chemo treatment
                chemo_application_point[i, t] = 1
                current_chemo_dose = chemo_amt[0]

            # Update chemo dosage
            #chemo_dosage[i, t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose
            chemo_dosage[i, t] = current_chemo_dose
            
            if True:
                if t <= lag_y:
                    cancer_volume[i, t] = cancer_volume[i, t - 1] *\
                    (1 + rho * np.log(K / cancer_volume[i, t - 1]) + noise[t])
                else:   
                    #New model (for t \geq \tau)
                    #Y_{t+1} = Y_t + \Bigl(\rho \log(\frac{K}{Y_{\textcolor{red}{t-\tau}}}) + \epsilon_{t+1}\Bigr) Y_{\textcolor{red}{t - \tau}} - Y_{\textcolor{red}{t-\tau}}\Bigl(\alpha_c c_{t+1} + (\alpha_r d_{t+1} + \beta_r d_{t+1}^{2})\Bigr)
                    y_tlag_mean = cancer_volume[i, :t - lag_y + 1].mean()
                    cancer_volume[i, t] = cancer_volume[i, t - 1] + (rho * np.log(K / y_tlag_mean) + noise[t]) * y_tlag_mean\
                                    - y_tlag_mean * (beta_c * chemo_dosage[i, t] +\
                                                (alpha * radio_dosage[i, t] + \
                                                beta * radio_dosage[i, t] ** 2))
            else:
                # Original model (for t \geq \tau)
                cancer_volume[i, t] = cancer_volume[i, t - 1] *\
                    (1 + rho * np.log(K / cancer_volume[i, t - 1]) + noise[t]) - \
                    (beta_c * chemo_dosage[i, t] + (alpha * radio_dosage[i, t] + beta * radio_dosage[i, t] ** 2)) * \
                    cancer_volume[i, t - 1]
            
            if cancer_volume[i, t] > TUMOUR_DEATH_THRESHOLD:
                cancer_volume[i, t] = TUMOUR_DEATH_THRESHOLD
                b_death = True
                break  # patient death

            # recovery threshold as defined by the previous stuff
            if recovery_rvs[i, t] < np.exp(-cancer_volume[i, t] * TUMOUR_CELL_DENSITY):
                cancer_volume[i, t] = 0
                b_recover = True
                break


        # Package outputs
        sequence_lengths[i] = int(t)
        death_flags[i, t] = 1 if b_death else 0
        recovery_flags[i, t] = 1 if b_recover else 0

    outputs = {'cancer_volume': cancer_volume,
               'chemo_dosage': chemo_dosage,
               'radio_dosage': radio_dosage,
               'chemo_application': chemo_application_point,
               'radio_application': radio_application_point,
               'chemo_probabilities': chemo_probabilities,
               'radio_probabilities': radio_probabilities,
               'sequence_lengths': sequence_lengths,
               'death_flags': death_flags,
               'recovery_flags': recovery_flags,
               'patient_types': patient_types,
               'noise_terms': noise_terms
               }

    return outputs


def simulate_counterfactuals_treatment_seq(observed_data:dict, simulation_params:dict, n_periods:int, treatment_seq:np.ndarray):
    """
    Simulation of counterfactual trajectories for subset of patients with sliding window intervention in [t, t + n_periods) (tau = n_periods - 1)
    :param simulation_params: Parameters of the simulation (must fit with observed_data!)
    :param treat_seq: intervention sequence np.array of shape (n_periods, 2) with 0/1 values
    :return: simulated data of outcome with shape (num_patients, seq_length - lagy - tau)
    """

    logger.info(f"Simulating counterfactuals data for subset. \n" 
                f"NUM_PERIODS = {n_periods}, NUM_PATIENTS = {observed_data['cancer_volume'].shape[0]}")
    print("treatment_seq:", treatment_seq)
    assert observed_data['cancer_volume'].shape[0] == simulation_params['initial_stages'].shape[0], \
        "Simulation parameters and observed data must have the same number of patients!"
    assert n_periods == treatment_seq.shape[0], \
        "Number of periods must be equal to the number of treatment sequences!"
    
    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    # radio_days = np.array([i + 1 for i in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params['initial_stages']
    initial_volumes = simulation_params['initial_volumes']
    alphas = simulation_params['alpha']
    rhos = simulation_params['rho']
    betas = simulation_params['beta']
    beta_cs = simulation_params['beta_c']
    Ks = simulation_params['K']
    patient_types = simulation_params['patient_types']
    window_size = simulation_params['window_size']  # controls the lookback of the treatment assignment policy
    lag = simulation_params['lag']
    lag_y = simulation_params['lag_y']

    seq_length = observed_data['cancer_volume'].shape[1]
    num_patients = observed_data['cancer_volume'].shape[0]

    test_idx = 0

    #apply same background noise as in the observed data
    noise = observed_data['noise_terms']
    
    assert alphas.shape == (num_patients, )
    assert rhos.shape == (num_patients, )
    assert betas.shape == (num_patients, )
    assert beta_cs.shape == (num_patients, )
    assert Ks.shape == (num_patients, )

    #observed data
    cancer_volume = observed_data['cancer_volume']
    #placeholder for ctf_cancer_volume
    cancer_volume_ctf = np.zeros((num_patients, seq_length - lag_y - n_periods + 1), dtype = np.float32)
    active_entries = np.ones_like(cancer_volume_ctf, dtype = np.float32)
    ctf_volume = np.zeros((num_patients, seq_length), dtype = np.float32)

    for t in range(lag_y, seq_length - n_periods + 1):
        # t is the starting point of the treatment sequence, before t use observed data
        #clear the entries for ctf_volume
        ctf_volume[:, t:] = 0.0
        for k in range(n_periods):
            # apply treatment sequence
            treatment = treatment_seq[k]
            ctf_chemo_dosage, ctf_radio_dosage = 0.0, 0.0
            if treatment[0] == 1: #chemo
                ctf_chemo_dosage = chemo_amt[0]
            if treatment[1] == 1: #radio
                ctf_radio_dosage = radio_amt[0]
            
            
            #simulate ctf volume for t+k
            y_tlag_mean = cancer_volume[:, :t + k - lag_y + 1].mean(axis = 1)
            assert y_tlag_mean.shape == (num_patients, )
            assert noise[:, t + k].shape == (num_patients, )

            if k == 0:
                ctf_volume[:, t] = cancer_volume[:, t]
            else:
                ctf_volume[:, t + k] = ctf_volume[:, t + k - 1] + (rhos * np.log(Ks / y_tlag_mean) + noise[:, t + k]) * y_tlag_mean\
                                    - y_tlag_mean * (beta_cs * ctf_chemo_dosage +\
                                                (alphas * ctf_radio_dosage + \
                                                betas * ctf_radio_dosage ** 2))
            #check if ctf volume satisfies recovery condition
            recovery_idx = np.exp(-ctf_volume[:, t + k] * TUMOUR_CELL_DENSITY) > 0.95
            active_entries[recovery_idx, t - lag_y] = 0.0
        
        cancer_volume_ctf[:, t - lag_y] = ctf_volume[:, t + n_periods - 1]

    y_ctf = cancer_volume_ctf

    return y_ctf, active_entries


def get_scaling_params(sim):
    real_idx = ['cancer_volume', 'chemo_dosage', 'radio_dosage']

    # df = pd.DataFrame({k: sim[k] for k in real_idx})
    means = {}
    stds = {}
    seq_lengths = sim['sequence_lengths']
    for k in real_idx:
        active_values = []
        for i in range(seq_lengths.shape[0]):
            end = int(seq_lengths[i])
            active_values += list(sim[k][i, :end])

        means[k] = np.mean(active_values)
        stds[k] = np.std(active_values)

    # Add means for static variables`
    means['patient_types'] = np.mean(sim['patient_types'])
    stds['patient_types'] = np.std(sim['patient_types'])

    return pd.Series(means), pd.Series(stds)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plotting Functions


def plot_treatments(data: dict, patient: int):
    df = pd.DataFrame({'N(t)': data['cancer_volume'][patient],
                       'C(t)': data['chemo_dosage'][patient],
                       'd(t)': data['radio_dosage'][patient],
                       })
    df = df[['N(t)', "C(t)", "d(t)"]]
    df.plot(secondary_y=['C(t)', 'd(t)'])
    plt.xlabel("$t$")
    plt.show()

def plot_ctf_trajectories(data_obs, ctf_volume, treatment_seq, patient = 0):
    """
    Plot the counterfactual trajectories for a patient with treatment sequence
    data_obs: observed data (factual)
    ctf_volume: counterfactual volume
    treatment_seq: treatment sequence
    """
    n_periods = treatment_seq.shape[0]
    seq_length = data_obs['cancer_volume'].shape[1]
    lag_y = seq_length - ctf_volume.shape[1] + 1 - n_periods
    #first plot observed traj until start_time + periods - 1
    plt.figure(figsize=(10, 5))
    #plot observed data
    #plot a curve with x and y specified


    plt.plot(range(seq_length), data_obs['cancer_volume'][patient], label = "Observed")
    plt.plot(range(lag_y + n_periods - 1, seq_length), ctf_volume[patient], label = "Counterfactual")
    plt.show()


def plot_sigmoid_function(data: dict):
    """
    Simple plots to visualise probabilities of treatment assignments

    :return:
    """

    # Profile of treatment application sigmoid
    for coeff in [i for i in range(11)]:
        tumour_death_threshold = calc_volume(13)
        assigned_beta = coeff / tumour_death_threshold
        assigned_interp = tumour_death_threshold / 2
        idx = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        volumes = idx * tumour_death_threshold

        def sigmoid_fxn(volume, beta, intercept):
            return (1.0 / (1.0 + np.exp(-beta * (volume - intercept))))

        data[coeff] = pd.Series(sigmoid_fxn(volumes, assigned_beta, assigned_interp), index=idx)

    df = pd.DataFrame(data)
    df.plot()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    np.random.seed(100)

    seq_length = 60  # about half a year
    window_size = 15
    lag = 0  # lag of treatment assignment
    num_patients = 1000
    chemo_coeff = radio_coeff = 10.0

    params = generate_params(num_patients, chemo_coeff=chemo_coeff, radio_coeff=radio_coeff, window_size=window_size, lag=lag)
    params['window_size'] = window_size
    training_data = simulate_factual(params, seq_length)

    params = generate_params(int(num_patients / 10), chemo_coeff=chemo_coeff, radio_coeff=radio_coeff, window_size=window_size,lag=lag)
    params['window_size'] = window_size
    validation_data = simulate_factual(params, seq_length)

    params_test = generate_params(int(num_patients / 10), chemo_coeff=chemo_coeff, radio_coeff=radio_coeff, window_size=window_size, lag=lag)
    params_test['window_size'] = window_size
    test_data_factuals = simulate_factual(params_test, seq_length)
    #test_data_counterfactuals = simulate_counterfactual_1_step(params, seq_length)

    #params = generate_params(int(num_patients / 10), chemo_coeff=chemo_coeff, radio_coeff=radio_coeff, window_size=window_size, lag=lag)
    params['window_size'] = window_size
    #treatment_seq = np.array([[1, 1], [1, 1], [1, 1]])
    treatment_seq = np.array([[0, 0], [0, 0], [0, 0]])
    ctf_volmue, active_entries = simulate_counterfactuals_treatment_seq(test_data_factuals, params_test, 3, treatment_seq)
    for i in range(10):
        plot_ctf_trajectories(test_data_factuals, ctf_volmue, treatment_seq, patient = i)

    # Plot patient
    # for i in range(5, 10):
    #    plot_treatments(training_data, i)
    
