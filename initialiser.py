import logging
from mup import get_shapes, make_base_shapes, set_base_shapes, MuAdamW
from model.m4 import LaTPFNV4
from pathlib import Path

import torch

from util.persist import flush_gpu

from data.priors.LaTPFN_dataset import LaTPFNDataset

from functools import partial

logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path("./model/mup_imp/")
BASE_WIDTH = 64


def build_model(
    width: int,
    config: dict,
    n_outputs: int,
    use_mup_parametrization: bool = True,
    load_base_shapes: bool = False,
    build_base_shapes: bool = False,
):

    model = LaTPFNV4(
        shape=config.shape,
        d_model=int(BASE_WIDTH * width),
        d_ff=int(BASE_WIDTH * config.model_size.ff_scale * width),
        nhead=config.model_size.nhead,
        num_layers=config.model_size.num_layers,
        dropout=config.model_size.dropout,
        device=config.device,
        n_outputs=n_outputs,
        use_mup_parametrization=use_mup_parametrization,
        n_domain_params=config.n_domain_params,
        masking_type=config.masking_type,
        train_noise=config.train_noise,
        ema_decay=config.ema_decay,
        ema_warmup_iterations=config.ema_warmup_epochs * config.train_length,
    ).to(config.device)

    shapes_path=None

    if use_mup_parametrization and load_base_shapes:

        shapes_path = model_pretrain_artifacts(config, n_outputs, build_base_shapes)

        model = set_base_shapes(model, str(shapes_path))


    model.width = width
    return model



def model_pretrain_artifacts(
    config: dict,
    n_outputs: int,
    build_base_shapes: bool = False,
):
    shapes_path = ARTIFACT_PATH / "mup_shapes.bsh"
    if build_base_shapes:
        logger.info("Creating MUP base shapes")
        logger.info("Building small model")

        model_small = build_model(
            config=config,
            width=1,
            use_mup_parametrization=True,
            load_base_shapes=False,
            n_outputs=n_outputs,
        )

        small_shapes = get_shapes(model_small)

        logger.info("Building large model")
        model_large = build_model(
            config=config,
            width=2,
            use_mup_parametrization=True,
            load_base_shapes=False,
            n_outputs=n_outputs,
        )
        large_shapes = get_shapes(model_large)

        logger.info("Making base shapes")

        make_base_shapes(small_shapes, large_shapes, savefile=shapes_path)
        del model_small
        del model_large
        flush_gpu()
    return shapes_path


def build_optimizer(
    lr: float,
    eps: float,
    weight_decay: float,
    parameters,
    use_mup_parametrization: bool,
):
    opt_clss = MuAdamW if use_mup_parametrization else torch.optim.Adam
    print("optimizer", opt_clss)
    return opt_clss(
        parameters,
        lr=lr,
        eps=eps,
        weight_decay=weight_decay,
    )

def build_dataset(config, train=True):

    if train:
        registry = dict(
            univariate_time_pfn=partial(
                LaTPFNDataset,
                shape=config.shape,
                hyperprior_params=config.time_pfn_hyperprior_params,
                batch_size=config.batch_size,
                length=config.train_length,
                is_train=True,
                device="cpu",
                return_components=False,
                separate_noise=True,
                scale_noise=False,
                T_mapping=config.T_mapping,
                V_mapping=config.V_mapping,
                disable_context=config.disable_context,
                backcast_samples=config.backcast_samples,
            ),
        )
    else:

        registry = dict(
            univariate_time_pfn=partial(
                LaTPFNDataset,
                shape=config.shape,
                hyperprior_params=config.time_pfn_hyperprior_params,
                batch_size=config.batch_size_val,
                length=config.val_length,
                is_train=False,
                device="cpu" if config.ddp else config.device,
                return_components=False,
                scale_noise=False,
                separate_noise=False,
                T_mapping=config.T_mapping,
                V_mapping=config.V_mapping,
                disable_context=config.disable_context,
                backcast_samples=False,
            ),
        )

    return registry[config.prior]