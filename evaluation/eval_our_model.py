from util.persist import load_model
from initialiser import build_model
from model.loss import losses
from util.metrics import compute_metrics, create_forecast_plot, log_image
from tqdm import tqdm
from util.metrics import smape
from util.config_util import V_denormalize
import torch


def evaluate(config, model_path, loader, logger, is_tuning=False, denormalise=False):
    """
    Evaluate the model on the given dataset.
    Args:
    config: Configuration object
    model_path: location of the frozen model
    loader: Dataloader
    logger: Logger object
    is_tuning: If the model is being tuned
    denormalise: If the data needs to be denormalised
    """
    total_mse = []
    total_rrsme_cum = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss = losses[config.loss](
        reduction="mean", device=config.device, **config.get(f"loss_{config.loss}", {})
    )

    model_arch = build_model(
        width=config.width,
        config=config,
        n_outputs=loss.n_outputs,
        use_mup_parametrization=config.mup,
        load_base_shapes=True,
        build_base_shapes=True,
    )

    model = load_model(model_path, model_arch, device)

    for i, batch in tqdm(enumerate(loader)):

        norm_means, norm_stds = batch[-2], batch[-1]

        with torch.no_grad():
            mse, _, _rrsme_cum = validation_implementation(
                *[x.to(device) for x in batch[:8]],
                loss,
                model,
                device,
                config.eval_dataset_name,
                logger,
                is_tuning,
                i,
                norm_means,
                norm_stds,
                denormalise=denormalise,
            )
            total_mse.append(mse)
            total_rrsme_cum.append(_rrsme_cum)

    logger.log_metrics(
        {
            "dataset_mse": sum(total_mse) / len(total_mse),
            "dataset_rrsme_cum": sum(total_rrsme_cum) / len(total_rrsme_cum),
        }
    )

    print(
        f"{sum(total_mse) / len(total_mse)},{sum(total_rrsme_cum) / len(total_rrsme_cum)},LaTPFN,{config.eval_dataset_name},{'real' if denormalise else 'norm'}\n",
        file=open("results.csv", "a"),
        end="",
        flush=True,
    )


def validation_implementation(
    T_context_history,  # examples X history
    T_context_prompt,  # examples X prompts
    V_context_history,  # examples Y history
    V_context_prompt,  # examples targets
    T_heldout_history,  # decoder X history
    T_heldout_prompt,  # decoder X prompts
    V_heldout_history,  # decoder Y history
    V_heldout_prompt,  # targets
    # _,
    # __,
    loss,
    model,
    device,
    name,
    logger,
    ___,
    i,
    norm_means=None,
    norm_stds=None,
    denormalise=False,
) -> float:
    """
    Validation step.
    Accepts a single batch of processed data, returns the loss, metrics and logs the forecast plot.

    """

    run_with = model

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

    model_output = loss.prepare_output(model_output["forecast"])

    mean = loss.output_to_mean(model_output)

    if denormalise:
        mean = V_denormalize(
            mean, norm_means.squeeze(-1).to(device), norm_stds.squeeze(-1).to(device)
        )

    _mae, _rrsme_cum, mse = compute_metrics(V_heldout_prompt, mean)
    smape_metric = smape(V_heldout_prompt, mean, tf=False)

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
        device,
        loss,
    )

    log_image(
        fig,
        f"fit_on_{name}",
        logger,
        step=i,
    )

    logger.log_metrics(
        {
            "rrsme_cum": _rrsme_cum,
            "mse": mse,
        },
        step=i,
    )

    return mse.item(), smape_metric.cpu().numpy().mean(), _rrsme_cum.item()
