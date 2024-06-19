import torch
import numpy as np

import random
from lightning.pytorch import seed_everything

from functools import partial

from util.config_util import *

machine_name = "your_machine_name"

# LaT-PFN hyperprior params

seasonality_base = 2.0
w,m,a = seasonality_base*1,seasonality_base*2,seasonality_base*4

# config

config = dotdict(

    machine_name=machine_name,

    ## training

    # batch

    batch_size=int(16 // max(torch.cuda.device_count(), 1)),  
    batch_size_val=4,
    batch_accumulation=2,

    # optimizer

    start_lr=9e-4,
    end_lr=1e-9,
    ema_decay=0.9952,
    ema_warmup_epochs=95,
    adam_eps=1e-8,
    lambda_di=1e-7,
    lambda_latent=3.77e-3,
    weight_decay=1.77e-4,
    weight_decay_warmup_epochs=95,
    weight_decay_max=4.9e-2,
    scheduler_decay=0.96,
    scheduler_T0=9,

    # dynamics

    epochs=2000,
    eval_freq=50,
    real_eval_freq_multiplier=10,
    include_real_evals=False,
    val_length=int(max(torch.cuda.device_count(), 1)),
    train_length=250,
    train_noise=0.02,
    backcast_samples=True,
    selection_metric_name="train_latent_loss",
    early_stopping=500,
    early_stopping_epoch=500,
    selection_decay=0.2,
    save_models=True,


    ## engineering

    tf32=True,
    ddp=False,  # multi-gpu training with DistributedDataParallel
    device=torch.device("cuda"),
    project="LaT-PFN",
    # seed=torch.randint(43, 100, (1,)).item(),
    seed = 42,
    debug_mode=False,
    evaluation_project_name="time-pfn_real_data_exp_1",
    
    ## model

    model_size=dotdict(
        num_layers=3, ff_scale=2, nhead=4, dropout=0.1  # 600M
    ),
    loss_cross_entropy = dotdict(
        label_smoothing=0.01,
        n_bins=100,
        range_max=3.5,
        range_min=-3.5
    ),
    loss_bar = dotdict(
        borders=torch.linspace(-3.5, 3.5, 100)
    ),
    loss="cross_entropy",
    mup=True,
    width=8,
    masking_type="independent",

    ## data
    
    prior="univariate_time_pfn",
    T_mapping=lambda x: x / 365,
    V_mapping=normalize_V_mean_2std,
    n_domain_params=10,
    disable_context = False,  # puts the context to 0
    shape=ShapeConfig(
        n_context=16, n_sequence=240, n_features=1, n_heldout=4, n_prompt=60 
    ),
    shape_val=ShapeConfig(
        n_context=16, n_sequence=240, n_features=1, n_heldout=4, n_prompt=60
    ),
    time_pfn_hyperprior_params=dotdict(
        # Seasonality
        a_min=-a,
        a_max=a,
        a_fixed_variance=0.15,
        m_min=-m,
        m_max=m,
        m_fixed_variance=0.15,
        w_min=-w,
        w_max=w,
        w_fixed_variance=0.15,
        # Zero Inflation
        f_zi_min=0.0,
        f_zi_max=1.0,
        f_zi_fixed_variance=0.4,
        # Trend
        trend_lin_min=-0.015,
        trend_lin_max=0.015,
        trend_lin_fixed_variance=0.005,
        trend_exp_min=1 - 0.004,
        trend_exp_max=1 + (0.004 / 2.5),
        trend_exp_fixed_variance=0.001,
        trend_exp_multiplier=507,
        # Noise
        noise_k_min=0.8,
        noise_k_max=5,
        # Resolution
        resolution_min=0.1,
        resolution_max=1.,
        resolution_multiplier=53.6,
        # Discreteness
        discreteness_min=1,
        discreteness_max=10,
        # ZI Bias
        bias_zi_min=1,
        bias_zi_max=10,
        # Amplitude
        amplitude_min=1,
        amplitude_max=10,
        # Non-negative probability
        non_negative_prob=0.5,
        # Offset
        offset_lin_min=-1,
        offset_lin_max=2,
        offset_exp_min=-1,
        offset_exp_max=2,
        # Harmonics
        harmonics_min=4,
        harmonics_max=12,
    ),

    ## other

    model_path="LaT-PFN/2tbs1d2c/checkpoints/epoch=1999-step=250000.ckpt",
)

# set seeds
if torch.cuda.device_count() < 2:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    seed_everything(config.seed, workers=True)
