import torch
import torch.nn as nn

import wandb

from initialiser import (
    build_model,
    build_optimizer,
    build_dataset,
)

import torch
import lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import multiprocessing as mp

from model.loss import losses

from util.metrics import (
    log_image,
    create_forecast_plot,
    compute_metrics,
    ema_metric_update,
    extract_gradients,
)
from util.schedulers import CosineAnnealingWarmRestartsDecay
from util.config_util import dictify, dotdict

from evaluation.real_eval import create_eval_dataloaders

from util.persist import flush_gpu

from torch.optim.lr_scheduler import ReduceLROnPlateau


class EarlyStopping(Exception):
    pass


class TrainingProcess(L.LightningModule):
    """
    The training process for the model, includes training and validation steps.
    Before optimizer hook is used to log learning rate, for debugging purposes.
    Batch end hook is used to log weight decay ,for debugging purposes.
    Epoch end hook is used for model persistence and early stopping.
    """
    def __init__(
        self,
        pfn_model,
        loss_module,
        selection_metric_name,
        save_models,
        selection_decay,
        config,
        real_eval=None,
        debug_mode=False,
    ):
        super().__init__()

        self.config = config

        # optimization

        self.weight_decay_warmup = config.weight_decay_warmup_epochs * config.train_length

        # model

        self.model = pfn_model
        self.loss = loss_module
        self.error_loss = nn.MSELoss()
        self.di_keys = None

        # selection metric

        self.best = float("inf")
        self.selection_metric_name = selection_metric_name
        self.last_selection_metric = None
        self.save_models = save_models
        self.selection_decay = selection_decay

        # debug

        self.real_eval = real_eval
        self.debug_mode = debug_mode

    def update_selection_metric(self):
        new_value = self.trainer.logged_metrics[self.selection_metric_name]

        if self.last_selection_metric is not None:
            self.last_selection_metric = ema_metric_update(
                self.last_selection_metric, new_value, self.selection_decay
            )
        else:
            self.last_selection_metric = new_value

        self.log("selection_metric", self.last_selection_metric)

    def training_step(self, batch, batch_idx):

        (
            T_context_history,  # examples X history
            T_context_prompt,  # examples X prompts
            V_context_history,  # examples Y history
            V_context_prompt,  # examples targets
            T_heldout_history,  # decoder X history
            T_heldout_prompt,  # decoder X prompts
            V_heldout_history,  # decoder Y history
            V_heldout_prompt,  # targets
            hyperprior_params_context,  # hyperprior params for context
            hyperprior_params_heldout,  # hyperprior params for heldout
        ) = batch

        if self.di_keys is None:
            self.di_keys = list(sorted(hyperprior_params_context.keys()))

        # forward

        predictions = self.model(
            T_context_history,
            T_context_prompt,
            V_context_history,
            V_context_prompt,
            T_heldout_history,
            T_heldout_prompt,
            V_heldout_history,
            V_heldout_prompt,
            predict_all_heads=True,
        )

        target = self.loss.prepare_target(V_heldout_prompt)

        model_output = self.loss.prepare_output(predictions["forecast"])

        # loss

        batch_loss = self.loss(**model_output, target=target)

        # di

        di_pred = torch.cat(
            (
                predictions["domain_identification_prediction_context"],
                predictions["domain_identification_prediction_heldout"],
            ),
            dim=1,
        )

        di_target = torch.cat(
            (
                torch.stack(
                    tuple(hyperprior_params_context[key] for key in self.di_keys),
                    dim=-1,
                ).to(self.device),
                torch.stack(
                    tuple(hyperprior_params_heldout[key] for key in self.di_keys),
                    dim=-1,
                ).to(self.device),
            ),
            dim=1,
        )

        di_loss = nn.functional.mse_loss(di_pred, di_target, reduction="none")

        # latent

        latent_loss = self.error_loss(
            predictions["latent_prediction"], predictions["latent_target"]
        )

        # total

        total_loss = (
            batch_loss
            + (self.config.lambda_di * di_loss.mean())
            + (self.config.lambda_latent * latent_loss)
        )

        # log

        with torch.no_grad():
            predictor_bypass = self.error_loss(
                predictions["bypass"], predictions["latent_target"]
            )
            predictor_identity = self.error_loss(
                predictions["bypass"], predictions["latent_prediction"]
            )

            self.log("train_di_loss", di_loss.mean().item())
            self.log("train_raw_loss", batch_loss.item())
            self.log("train_latent_loss", latent_loss.item())
            self.log("train_total_loss", total_loss.item())
            for i, key in enumerate(self.di_keys):
                self.log(f"di/train_di_loss_{key}", di_loss[:, :, i].mean().item())
            self.log("train_predictor_bypass_distance", predictor_bypass.item())
            self.log("train_predictor_identity_distance", predictor_identity.item())
            self.log("ema_decay", self.model.ts_ema_constant._value.item())

        # selection metric

        if "train" in self.selection_metric_name:
            self.update_selection_metric()

        return total_loss

    def on_train_batch_end(self, *args, **kwargs) -> None:

        config_decay = self.config.weight_decay
        target_decay = self.config.weight_decay_max

        diff = target_decay - config_decay

        step = diff / self.weight_decay_warmup
        
        for group in self.trainer.optimizers[0].param_groups + [self.trainer.optimizers[0].defaults]:
            old_decay = group["weight_decay"]
            group["weight_decay"] = torch.clip(torch.scalar_tensor(old_decay + step, dtype=torch.float64), config_decay, target_decay).item()

        self.log("weight_decay", self.trainer.optimizers[0].defaults["weight_decay"])

        return super().on_train_batch_end(*args, **kwargs)  

    def on_train_epoch_end(self):
        if self.last_selection_metric is not None:
            if self.last_selection_metric < self.best:
                self.best = self.last_selection_metric
                self.log("best_selection_metric", self.best)
                print(f"new best model found with selection metric: {self.best}")
                if self.save_models:
                    torch.save(
                        self.model.state_dict(),
                        f"{self.logger.name}/{self.logger.version}/checkpoints/best.pt",
                    )
            if (
                self.last_selection_metric > self.config.early_stopping
                and self.current_epoch > self.config.early_stopping_epoch
            ):
                raise EarlyStopping(
                    f"early stopping at epoch {self.current_epoch} because selection metric is {self.last_selection_metric} and early stopping is {self.config.early_stopping}"
                )

    def validation_step(self, batch, batch_idx):
        """
        Evaluate model on both synthetic validation data and real datasets.
        """

        if batch_idx == 0 and self.global_step == 0 and self.current_epoch == 0:
            return

        self.validation_implementation("prior", *batch)

        if "val" in self.selection_metric_name:
            self.update_selection_metric()

        if (
            self.real_eval is not None
            and self.global_step
            % (self.config.real_eval_freq_multiplier * self.config.eval_freq)
            == 0
        ):
            print("evaluating on real data")
            for eval_name, loader in self.real_eval.items():
                print(f"evaluating on {eval_name}")
                try:
                    batch = next(loader)
                except StopIteration as e:
                    print(f"{eval} has been exhausted")
                    continue
                self.validation_implementation(
                    eval_name, *[x.to(self.device) for x in batch]
                )
                flush_gpu()

    def validation_implementation(
        self,
        name,
        T_context_history,  # examples X history
        T_context_prompt,  # examples X prompts
        V_context_history,  # examples Y history
        V_context_prompt,  # examples targets
        T_heldout_history,  # decoder X history
        T_heldout_prompt,  # decoder X prompts
        V_heldout_history,  # decoder Y history
        V_heldout_prompt,  # targets
        hyperprior_params_context=None,  # hyperprior params for context
        hyperprior_params_heldout=None,  # hyperprior params for heldout
    ) -> None:

        # forward with the batch as is

        run_with = self.model

        model_output = run_with(
            T_context_history,
            T_context_prompt,
            V_context_history,
            V_context_prompt,
            T_heldout_history,
            T_heldout_prompt,
            V_heldout_history,
            V_heldout_prompt,
            predict_all_heads=False,
        )

        model_output = self.loss.prepare_output(model_output["forecast"])

        mean = self.loss.output_to_mean(model_output)

        # calculate validation metrics

        mae, rrsme_cum, mse = compute_metrics(V_heldout_prompt, mean)

        batch_loss = self.loss(
            **model_output, target=self.loss.prepare_target(V_heldout_prompt)
        )

        # log metrics

        self.log(f"val_loss_{name}", batch_loss.item())
        self.log(f"val_timestep_mae_{name}", mae.item())
        self.log(f"val_cumulative_rrsme_{name}", rrsme_cum.item())
        self.log(f"val_timestep_mse_{name}", mse.item())

        # forward with a linspace and create a plot

        fig = create_forecast_plot(
            T_context_history,
            T_context_prompt,
            V_context_history,
            V_context_prompt,
            T_heldout_history,
            T_heldout_prompt,
            V_heldout_history,
            V_heldout_prompt,
            run_with,
            self.device,
            self.loss,
        )

        # log the plot

        log_image(
            fig,
            f"fit_on_{name}",
            self.logger,
            step=self.global_step,
        )

    def configure_optimizers(self):

        opt = build_optimizer(
            lr=self.config.start_lr,
            eps=self.config.adam_eps,
            parameters=self.model.parameters(),
            weight_decay=self.config.weight_decay,
            use_mup_parametrization=self.config.mup,
        )
        scheduler = CosineAnnealingWarmRestartsDecay(
            opt,
            T_0=self.config.scheduler_T0,
            decay=self.config.scheduler_decay,
            eta_min=self.config.end_lr,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
        }

    def on_before_optimizer_step(self, optimizer):
        self.model.update_emas()
        lr = optimizer.param_groups[0]["lr"]
        self.log("lr", lr)
        if self.debug_mode:
            parameter_iterator = self.model.named_parameters()
            gradients, min_placeholder, key = extract_gradients(
                parameter_iterator
            )
            for key, value in gradients.items():
                for metric, val in value.items():
                    if not metric == "grad_min" and val != min_placeholder:
                        self.log(f"debug/{key}_{metric}", val)


def train(config: dotdict) -> float:

    ## speedup
    if config.tf32:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

    ## Training Process

    loss = losses[config.loss](
        reduction="mean", device=config.device, **config.get(f"loss_{config.loss}", {})
    )

    logger = WandbLogger(project=config.project)

    logger.log_hyperparams(dictify(config))

    L.Trainer(
        limit_train_batches=config.train_length,
        max_epochs=config.epochs,
        logger=logger,
        val_check_interval=config.eval_freq,
        accelerator="gpu" if config.device == "cuda" else str(config.device),
        devices=torch.cuda.device_count() if config.ddp else 1,
        strategy="ddp" if config.ddp else "auto",
        accumulate_grad_batches=config.batch_accumulation,
    ).fit(
        model=(
            process := TrainingProcess(
                pfn_model=build_model(
                    width=config.width,
                    config=config,
                    n_outputs=loss.n_outputs,
                    use_mup_parametrization=config.mup,
                    load_base_shapes=True,
                    build_base_shapes=True,
                ),
                real_eval=(
                    create_eval_dataloaders(config=config)
                    if config.include_real_evals
                    else {}
                ),
                loss_module=loss,
                debug_mode=config.debug_mode,
                selection_metric_name=config.selection_metric_name,
                save_models=config.save_models,
                selection_decay=config.selection_decay,
                config=config,
            )
        ),
        train_dataloaders=DataLoader(
            build_dataset(config, train=True)(),
            batch_size=config.batch_size,
            num_workers=int(mp.cpu_count() // max(torch.cuda.device_count(), 1)),
            persistent_workers=True,
            pin_memory=True,
        ),
        val_dataloaders=DataLoader(
            build_dataset(config, train=False)(),
            batch_size=config.batch_size_val,
        ),
    )

    wandb.finish()

    return process.last_selection_metric


if __name__ == "__main__":
    from config import config as conf

    print(f"best score reached was: {train(conf)}")
