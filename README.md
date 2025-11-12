DeepBlip
==============================

DeepBlip: Estimating Conditional Average Treatment Effects Over Time

![Architecture of DeepBlip](./images/Architecture-new.png)

### Environment Setup
First, create a new environment with conda and then change directory to main root of the project.
```console
conda create -n new_env python=3.10.1
python -m pip install -r requirements.txt
```

## MlFlow
Our code requires Mlflow to log the training process. To start an experiments server, run: 

`mlflow server --port=3333`

## Experiments

### Dataset preparation:
Our experiments use two datasets: (1) The tumor growth synthetic dataset, and (2) the MIMIC-III semi-synthetic datasets. (1) is completed simulated with the source code. (2) needs the input of original dataset [MIMIC-III data](https://physionet.org/content/mimiciii/1.4/) preprocessed by the MIMIC-Extract tool ([https://github.com/MLforHealth/MIMIC_Extract](https://github.com/MLforHealth/MIMIC_Extract)) to create the hourly aggregated time series clinical dataset.

### Configurations
The main training script `config/config.yaml` is used for all experiments. Besides, user also needs to specify the dataset and the method:
1. dataset: `mimic_synthetic` or `tumor_growth`
2. model:
    - `deepblip`
    - `ha`
    - `rmsn`
    - `g_net`
    - `g_transformer`
    - `causal_trm`
    - `crn`
The neural backbone (LSTM or transformer) could be selected in the model configuration file.
___

### Example usage
To run our DeepBlip on tumor growth synthetic data with random seeds 101--105 under confounding level 1.0:
```console
python scripts/train_deepblip.py --multirun +dataset=tumor_growth +model=deepblip dataset.conf_coeff=1.0 exp.seed=101,102,103,104,105
```

To run our DeepBlip on MIMIC-III semi-synthetic data (2000 sample) with random seeds 101--105 and projection horizon 5:
```console
python scripts/train_deepblip.py --multirun +dataset=mimic_synthetic +model=deepblip dataset.conf_coeff=1.0 dataset.n_periods=6 exp.seed=101,102,103,104,105
```