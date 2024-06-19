from train import train, EarlyStopping
from config import config
import logging

from util.config_util import dotdict, ShapeConfig

from copy import deepcopy

import wandb

import optuna
from optuna.storages import JournalStorage, JournalFileStorage

from util.persist import flush_gpu


logger = logging.getLogger(__name__)


MAX_N_EPOCHS = 250
STUDY_NAME = "initial_lr_tune_april_2024_v4"
STUDY_PATH = "/tuning"
N_TRIAL = 512


def config_deepcopy(x: dotdict):

    output = dotdict()

    for key, value in x.items():
        if isinstance(value, dotdict):
            output[key] = config_deepcopy(value)
        elif isinstance(value, ShapeConfig):
            output[key] = ShapeConfig(**value.__dict__())
        else:
            try:
                output[key] = deepcopy(value)
            except:
                output[key] = value
                print(f"Could not deepcopy {key} so using the original value.")

    return output


def objective(trial: optuna.Trial):

    # deepcopy the config

    config_trial = config_deepcopy(config)

    # # insert them into the config

    config_trial.ema_decay = trial.suggest_float("ema_decay", 0.99, 1.0)

    warmup_epochs = trial.suggest_int("warmup_epochs", 10, 100)
    config_trial.ema_warmup_epochs = warmup_epochs
    config_trial.weight_decay_warmup_epochs = warmup_epochs

    config_trial.lambda_latent = 10 ** trial.suggest_float("lambda_latent", -6, -2)

    weight_decay_log = trial.suggest_float("weight_decay", -6, -3)
    config_trial.weight_decay = 10 ** weight_decay_log
    config_trial.weight_decay_max = 10 ** trial.suggest_float("weight_decay_max", weight_decay_log, -1)


    config_trial.epochs = MAX_N_EPOCHS
    config_trial.project = STUDY_NAME
    config_trial.save_models = False

    # train the model

    try:

        metric = train(config_trial)

    except EarlyStopping as e:
        metric = float(1e6)
        logger.exception(e)
        wandb.finish()

    flush_gpu()

    # report the best metric

    return metric


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=JournalStorage(
            JournalFileStorage(STUDY_PATH + "/" + STUDY_NAME + ".log")
        ),  # https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-solve-the-error-that-occurs-when-performing-parallel-optimization-with-sqlite3
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIAL)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
