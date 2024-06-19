from pytorch_lightning.loggers import WandbLogger
import wandb


from evaluation.eval_our_model import evaluate as LaTPFN_evaluate

from glob import glob

from config import config
from util.config_util import config_deepcopy, dotdict
from evaluation.real_eval import (
    create_eval_dataloaders,
    illness_eval_config,
    ett_h_1_eval_config,
    ett_h_2_eval_config,
    traffic_eval_config,
    etl,
)

from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm


def set_config(config, abstract, denorm):
    config.denormalize = denorm
    config.denormalize_targets = denorm
    config.denormalize = denorm
    config.smoother = 1
    config.abstract_time = abstract
    return config


if __name__ == "__main__":

    for denormalize in [False]:

        eval_config = config_deepcopy(config)
        eval_config.epochs = 1
        eval_config.project = "LaTPFN_Eval"
        eval_config.stride = 1
        eval_config.denormalise = denormalize
        eval_config.denormalize_targets = denormalize
        eval_config.denormalize = denormalize

        norm_suffix = "real" if denormalize else "norm"

        eval_datasets = create_eval_dataloaders(
            config=eval_config,
            illness_eval_config=set_config(
                config_deepcopy(dotdict(illness_eval_config)), True, denormalize
            ),
            ett_h_1_eval_config=set_config(
                config_deepcopy(dotdict(ett_h_1_eval_config)), True, denormalize
            ),
            ett_h_2_eval_config=set_config(
                config_deepcopy(dotdict(ett_h_2_eval_config)), True, denormalize
            ),
            traffic_eval_config=set_config(
                config_deepcopy(dotdict(traffic_eval_config)), True, denormalize
            ),
            etl=set_config(config_deepcopy(dotdict(etl)), True, denormalize),
            use_infinity_loader=False,
        )

        non_abstract_datasets = create_eval_dataloaders(
            config=eval_config,
            illness_eval_config=set_config(
                config_deepcopy(dotdict(illness_eval_config)), False, denormalize
            ),
            ett_h_1_eval_config=set_config(
                config_deepcopy(dotdict(ett_h_1_eval_config)), False, denormalize
            ),
            ett_h_2_eval_config=set_config(
                config_deepcopy(dotdict(ett_h_2_eval_config)), False, denormalize
            ),
            traffic_eval_config=set_config(
                config_deepcopy(dotdict(traffic_eval_config)), False, denormalize
            ),
            etl=set_config(config_deepcopy(dotdict(etl)), False, denormalize),
            use_infinity_loader=False,
        )

        for (ds_name, loader), non_abstract_loader in tqdm(
            zip(eval_datasets.items(), non_abstract_datasets.values())
        ):

            print(f"evaluating baselines on: {ds_name}")

            eval_config.eval_dataset_name = ds_name

            for modelpath in list(glob("LaT-PFN/**/**/*.ckpt")):

                print(f"evaluating {modelpath}")

                seed = modelpath.split("/")[-3]

                LaTPFN_evaluate(
                    config=eval_config,
                    model_path=modelpath,
                    loader=iter(loader()),
                    logger=WandbLogger(
                        project=eval_config.project,
                        name=f"LaTPFN_{seed}_{ds_name}_{norm_suffix}",
                    ),
                    is_tuning=False,
                    denormalise=denormalize,
                )
                wandb.finish()
