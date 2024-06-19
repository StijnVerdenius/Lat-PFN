import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import io
from PIL import Image

from collections import defaultdict


def log_image(image, figure_name, logger, step):
    logger.log_image(
        key=figure_name,
        images=[wandb.Image(image, caption=figure_name)],
        step=step,
    )


def add_attention_hook(attn_layer):
    """
    Add a forward hook to the attention layer to store the attention weights
    """

    hook_memory = {}

    def hook_function(module, input, _):
        # avoid recursion
        hook_memory.pop("handle").remove()

        # store the attention weights
        _, attn_weights = module(*input, need_weights=True)
        hook_memory["attn"] = attn_weights[:, : input[0].shape[1], : input[-1].shape[1]]

    # Attach the forward hook
    hook = attn_layer.register_forward_hook(hook_function)

    # store the handle to remove it later
    hook_memory["handle"] = hook

    return hook_memory, hook


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg")
    buf.seek(0)
    image = Image.open(buf)
    return image


def plot(
    mean,
    var,
    T_heldout_history,
    linspace,
    V_heldout_history,
    T_context_history,
    V_context_history,
    T_context_prompt,
    V_context_prompt,
    V_heldout_prompt,
    T_heldout_prompt,
    ylim: tuple = None,
    attn=None,
):

    plt.close()

    fig = plt.figure(figsize=(15, 15))

    if len(mean.shape) == 2:
        mean = mean.unsqueeze(0)
        var = var.unsqueeze(0)
    for subplot in range(min(4, mean.shape[0])):

        plt.subplot(2, 2, subplot + 1)

        if ylim is not None:
            plt.ylim(ylim)

        # extract data
        mean_ = mean[subplot, 0, :].cpu().detach().squeeze().float()
        var_ = var[subplot, 0, :].cpu().detach().squeeze().float()
        Dx_ = torch.cat(
            (
                T_context_history[subplot, :, :, :].cpu().detach().squeeze().float(),
                T_context_prompt[subplot, :, :, :].cpu().detach().squeeze().float(),
            ),
            dim=-1,
        )
        Dy_ = torch.cat(
            (
                V_context_history[subplot, :, :, :].cpu().detach().squeeze().float(),
                V_context_prompt[subplot, :, :, :].cpu().detach().squeeze().float(),
            ),
            dim=-1,
        )
        T_heldout_history_ = (
            T_heldout_history[subplot, 0, :].cpu().detach().squeeze().float()
        )
        linspace_ = linspace[subplot, 0, :, :].cpu().detach().squeeze().float()
        V_heldout_history_ = (
            V_heldout_history[subplot, 0, :].cpu().detach().squeeze().float()
        )
        V_heldout_prompt_ = (
            V_heldout_prompt[subplot, 0, :].cpu().detach().squeeze().float()
        )
        T_heldout_prompt_ = (
            T_heldout_prompt[subplot, 0, :].cpu().detach().squeeze().float()
        )

        # plot background context

        grey = np.array([0.5, 0.5, 0.5])
        green = np.array([0, 1, 0])

        if attn is not None:
            alphas = attn["D"][subplot].cpu().detach().squeeze().float()
            alphas = (
                (torch.log(alphas + 1.1) / torch.log(torch.tensor(3.5)))
                .sqrt()
                .clip(0, 1)
            )
        else:
            alphas = [0.5] * len(Dx_)
        i = 0
        for x_, y_, alpha in zip(Dx_, Dy_, alphas):
            # ignore if nan in alpha
            if torch.isnan(alpha):
                continue

            c = grey * (1 - alpha.item()) + green * alpha.item()
            plt.plot(x_, y_, alpha=alpha.item(), color=c, linewidth=4 * alpha.item(), label="context series" if i == 0 else "")
            i += 1

        # plot PPD uncertainty

        plt.fill_between(
            linspace_,
            (mean_ - var_),
            (mean_ + var_),
            alpha=0.5,
            label="PPD uncertainty"
        )

        # plot prompt

        plt.scatter(T_heldout_prompt_, V_heldout_prompt_, color="blue", label="target")

        # plot in-sample history

        plt.scatter(
            T_heldout_history_,
            V_heldout_history_,
            color="red",
            label="history"
        )

        # plot mean prediction

        plt.plot(linspace_, mean_, label="PPD mean")

        plt.legend()

    return fig


def plot_error_metrics(metrics, batch_counter, save_path):
    for idx, metric in enumerate(metrics):
        fig = plt.figure(figsize=(15, 15))
        plt.plot(metric, list(range(batch_counter)))
        fig.savefig(f"{save_path}/{idx}.png")


def compute_metrics(target, mean):

    # error 1: mean absolute error (MAE) per timestep

    mae = (mean.squeeze() - target.squeeze()).abs().mean()

    # error 2: root relative squared mean error (RRSME) cumulative over all timesteps

    translate = min(target.min(), mean.min())
    pos_mean = (mean - translate).squeeze().sum(-1)
    pos_target = (target - translate).squeeze().sum(-1)
    rrsme_cum = torch.mean(
        torch.abs(pos_mean - pos_target).pow(2) / target.squeeze().abs().sum(-1)
    ).sqrt()

    # error 3: mean squared error (MSE) per timestep

    mse = torch.functional.F.mse_loss(mean.squeeze(), target.squeeze())

    return mae, rrsme_cum, mse


def create_forecast_plot(
    T_context_history,
    T_context_prompt,
    V_context_history,
    V_context_prompt,
    T_heldout_history,
    T_heldout_prompt,
    V_heldout_history,
    V_heldout_prompt,
    model,
    device,
    loss_fn,
):

    # add attention hook

    run_with = model

    hook_memory, hook = add_attention_hook(run_with.pfn.decoder.layers[0].multihead_attn)
    
    # forward with the linspace prompt
    
    model_out = model(
        T_context_history,
        T_context_prompt,
        V_context_history,
        V_context_prompt,
        T_heldout_history,
        T_heldout_prompt,
        V_heldout_history,
        V_heldout_prompt,
        predict_all_heads=False,
        backcast=True,
    )

    logits, backcast = model_out["forecast"], model_out["backcast"]

    model_output = loss_fn.prepare_output(torch.cat([backcast, logits], dim=-2))

    mean = loss_fn.output_to_mean(model_output)
    var = loss_fn.output_to_std(model_output)

    # remove hook and extract attention weights

    # extract attention weights

    batch_size, heldout_context_size, n_prompts, _ = T_heldout_prompt.shape
    _, example_context_size, n_history, _ = T_context_history.shape

    # select the first heldout context and average over prompts

    context_attn = (
        (torch.ones(batch_size, heldout_context_size, n_prompts, example_context_size) * 0.005)
        .view(batch_size, heldout_context_size, n_prompts, example_context_size)[:, 0]
        .mean(1)
    )

    # remove the hooks

    try:
        hook.remove()
    except Exception as e:
        print(e)

    # plot

    fig = fig2img(
        plot(
            mean,
            var,
            T_heldout_history,
            torch.cat([T_heldout_history, T_heldout_prompt], dim=-2),
            V_heldout_history,
            T_context_history,
            V_context_history,
            T_context_prompt,
            V_context_prompt,
            V_heldout_prompt,
            T_heldout_prompt,
            attn={"D": context_attn},
        ),
    )

    return fig


def smape(y_true, y_pred, tf=True):
    """ Calculate Armstrong's original definition of sMAPE between `y_true` & `y_pred`.
        `loss = 200 * mean(abs((y_true - y_pred) / (y_true + y_pred), axis=-1)`
        Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        Returns:
        Symmetric mean absolute percentage error values. shape = `[batch_size, d0, ..
        dN-1]`.
        """
    if tf:
        import tensorflow as tf
        from keras import backend
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        diff = tf.abs(
            (y_true - y_pred) /
            backend.maximum(y_true + y_pred, backend.epsilon())
        )
        return 200.0 * backend.mean(diff, axis=-1)
    else:
        y_true = y_true.squeeze()
        diff = torch.abs((y_true - y_pred) / (y_true + y_pred + 1e-8))
        return 200.0 * torch.mean(diff, axis=-1)


def ema_metric_update(old_value, new_value, decay=0.2):
    return decay * new_value + (1 - decay) * old_value


def extract_gradients(parameter_iterator):
        min_placeholder = 100000000
        gradients = defaultdict(
            lambda: dict(
                grad_max=0,
                grad_min=min_placeholder,
                grad_norm=0,
                param_norm=0,
            )
        )
        for name, param in parameter_iterator:
            key = ".".join(name.split(".")[:2])
            gradients[key]["param_norm"] += param.norm().item()
            if param.grad is not None:
                gradients[key]["grad_norm"] += param.grad.norm().item()
                gradients[key]["grad_max"] = max(
                    gradients[key]["grad_max"], param.grad.abs().max().item()
                )
                gradients[key]["grad_min"] = min(
                    gradients[key]["grad_min"], param.grad.abs().min().item()
                )
        return gradients, min_placeholder, key

