import torch
from torch import nn

from model.base_pfn import BasePFN, AttentionAVGPool

from .mup_imp.mup_custom import MupMultiheadAttention
from mup.init import xavier_uniform_ as mup_xavier_uniform_

import torch
import torch.nn as nn
from .base_pfn import BasePFN, AttentionAVGPool
from functools import partial
from model.mup_imp.mup_custom import MuReadoutModified

from torch.optim.swa_utils import AveragedModel


class MobileNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        **kwargs
    ):
        super(MobileNetBlock, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                **kwargs,
            ),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                **kwargs,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.proj = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        bs, ds, sl, _ = x.shape
        projected = self.proj(x)
        x = x.transpose(2, 3).flatten(0, 1)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2).reshape(bs, ds, sl, self.out_channels)
        return x + projected


class FFBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super(FFBlock, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.LayerNorm(d_out),
            nn.Dropout(dropout),
        )
        if d_in != d_out:
            self.proj = nn.Linear(d_in, d_out)

    def forward(self, x):
        residual = x
        x = self.ff(x)
        if x.shape[-1] != residual.shape[-1]:
            residual = self.proj(residual)
        return x + residual


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout, use_mup_parametrization=True):
        super(SelfAttentionBlock, self).__init__()

        attn_cls = (
            MupMultiheadAttention if use_mup_parametrization else nn.MultiheadAttention
        )

        self.attn = attn_cls(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        n_dims = len(x.shape)

        if n_dims > 3:
            a, b, c, d = x.shape
            x = x.flatten(0, 1)

        residual = x
        x, _ = self.attn(x, x, x)
        x = x + residual

        residual = x
        x = self.ff(x)
        x = x + residual

        if n_dims > 3:
            x = x.view(a, b, c, d)

        return x


class Convttention(nn.Module):
    """
    Implements the Lat-PFN (JEPA-style) Embedder via Dilated Mobilenets and Self-Attention.
    """

    def __init__(self, d_in, d_out, base=2, depth=8, use_mup_parametrization=True):
        super().__init__()

        self.mobilenet = nn.Sequential(
            MobileNetBlock(d_in + 1, d_out, 3, 1, 1, 1, padding_mode="replicate"),
            *[
                MobileNetBlock(d_out, d_out, 3, 1, base**i, base**i)
                for i in range(1, depth + 1)
            ],
        )

        self.mlp = nn.Sequential(
            FFBlock(d_out, d_out, 0.1),
            SelfAttentionBlock(
                d_out, 4, 0.1, use_mup_parametrization=use_mup_parametrization
            ),
            nn.Linear(d_out, d_out),
        )

        self.d_model = d_out

    def forward(self, history_T, history_V, prompt_T=None):

        # sort the sequence by time

        for_sort = torch.cat(
            [history_T] + ([prompt_T] if prompt_T is not None else []), dim=-2
        ).argsort(dim=-2)

        # concatenate thev features

        history = torch.cat([history_T, history_V, torch.zeros_like(history_T)], dim=-1)

        # add the prompt if it exists

        if prompt_T is not None:
            prompt = torch.cat(
                [prompt_T, torch.zeros_like(prompt_T), torch.ones_like(prompt_T)],
                dim=-1,
            )

            history = torch.cat([history, prompt], dim=-2)

        # push through the network and undo the sorting

        output = self.mlp(
            self.mobilenet(history.gather(dim=-2, index=for_sort.repeat(1, 1, 1, 3)))
        ).gather(dim=-2, index=for_sort.argsort(dim=-2).repeat(1, 1, 1, self.d_model))

        # split the forecast from the embeddings

        if prompt_T is not None:
            prompt_dim = prompt_T.shape[-2]
            return (
                output[..., :-prompt_dim, :],
                output[..., -prompt_dim:, :],
            )
        else:
            return output, None


class ScheduledEma(float):
    """
    Class used as a wrapper for the EMA decay constant, so that we can update by pointer.
    """

    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value.float()

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def __float__(self):
        return self.value

    def __int__(self):
        return int(self.value)

    def __add__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __mul__(self, other):
        return self.value * other

    def __truediv__(self, other):
        return self.value / other

    def __radd__(self, other):
        return other + self.value

    def __rsub__(self, other):
        return other - self.value

    def __rmul__(self, other):
        return other * self.value

    def __rtruediv__(self, other):
        return other / self.value


class LaTPFNV4(nn.Module):
    """
    LatentLaTPFN (LaT-PFN) implementation.

    For inference, use the `create_embeddings` and `create_forecast methods`.

    Args:
        d_model: int, the size of the model's hidden states.
        d_ff: int, the size of the feedforward network's hidden states.
        nhead: int, the number of heads in the multiheadattention models.
        num_layers: int, the number of layers in various parts of the network.
        dropout: float, the dropout value.
        n_outputs: int, the number of bins to predict in the decoder.
        use_mup_parametrization: bool, whether to use the MUP parametrization.
        n_domain_params: int, the number of domain parameters to predict in system identifcation.
        device: str, the device to use.
        train_noise: float, the amount of noise to add during training.
        masking_type: str, the type of masking to use in PFN.
        ema_decay: float, the decay constant for the EMA of th target embedder.
        ema_warmup_iterations: int, the number of warmup iterations for the EMA.
        shape: , the shape of the input data. (deprecated here)
    """

    def __init__(
        self,
        d_model=128,
        d_ff=256,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        n_outputs=1,
        use_mup_parametrization: bool = True,
        n_domain_params=10,
        device="cuda",
        train_noise=0.02,
        masking_type="independent",
        ema_decay=0.999,
        ema_warmup_iterations=250 * 50,
        *,
        shape,
        **kwargs
    ):
        super().__init__()

        print("Ignoring kwargs:", self, kwargs)
        print("d_model", d_model)
        print("using mup parametrization", use_mup_parametrization)

        self.n_outputs = n_outputs
        self.train_noise = train_noise

        # X-embedder

        self.TS_encoder = Convttention(
            2, d_model, base=2, depth=8, use_mup_parametrization=use_mup_parametrization
        )

        # Y-embedder (EMA of X-embedder)

        self.ts_ema_constant = ScheduledEma(
            value=torch.scalar_tensor(ema_decay, dtype=torch.float64)
        )

        self.ema_decay = ema_decay
        self.ema_warmup_iterations = ema_warmup_iterations

        self.TS_ema = AveragedModel(
            self.TS_encoder,
            device=device,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                self.ts_ema_constant
            ),
        )

        # Sequence summarizer

        self.proj = nn.Sequential(
            *[FFBlock(d_model, d_model, dropout) for _ in range(1)]
        )

        self.avg_pool = AttentionAVGPool(
            d_model, nhead, dropout, use_mup_parametrization=use_mup_parametrization
        )

        # PFN Predictor

        self.pfn = BasePFN(
            d_model=d_model,
            d_ff=d_ff,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            use_mup_parametrization=use_mup_parametrization,
            masking_type=masking_type,
        )

        # Decoder

        last_layer_clss = (
            partial(MuReadoutModified, output_mult=2)
            if use_mup_parametrization
            else nn.Linear
        )

        self.head_raw = nn.Sequential(
            *[FFBlock(d_model, d_model, dropout) for _ in range(num_layers)],
            last_layer_clss(d_model, n_outputs, bias=False),
        )

        # System identification head

        self.head_di = nn.Sequential(
            *[FFBlock(d_model, d_model, dropout) for _ in range(2)]
        )
        self.last_layer_di = last_layer_clss(d_model, n_domain_params)

    def update_emas(self):
        """
        Training util function to take a step for the JEPA ema decay constant linear warmup schedule.
        """
        self.TS_ema.update_parameters(self.TS_encoder)
        self.ts_ema_constant._value = torch.clip(
            self.ts_ema_constant._value
            + ((1 - self.ema_decay) / self.ema_warmup_iterations),
            0.995,
            1.0,
        )

    def create_embeddings(
        self, T: torch.Tensor, V: torch.Tensor, supress_warnings: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Use this method in inference to create embeddings for the time series data.
        Args:
            T: abstract time:  [batch_size, context_size, sequence_length, 1]
            V: Z-2std-normalized values: [batch_size, context_size, sequence_length, 1]
        Returns:
            dict[str, torch.Tensor]: Dictionary containing the embeddings.
                - "per_timestep_embedding":   [batch_size, context_size, sequence_length, d_model]
                - "per_series_summary_embedding":   [batch_size, context_size, d_model]
        """

        # verify inputs

        assert T.shape == V.shape, "T and V must have the same shape"
        assert (
            len(T.shape) == 4
        ), "T and V must have shape [batch_size, context_size, sequence_length, 1]"
        assert not self.training, "This method should only be called in eval mode"
        assert (
            torch.isnan(T).sum() == 0 and torch.isnan(V).sum() == 0
        ), "Please deal with NaNs before calling this method"
        assert all(
            [x > 1 for x in V.shape[:-1]]
        ), "All dimensions except the last one must be > 1"

        percentage_out_of_bounds = (V.abs() > 3.5).float().mean()

        if percentage_out_of_bounds > 0.1 and not supress_warnings:
            print(
                "Warning: ",
                percentage_out_of_bounds,
                " of the values out of bounds (-3.5, 3.5) for bins",
            )

        # do the inference

        with torch.no_grad():
            returnables = {}

            ema_output = self.TS_ema(T, V)[0]

            returnables["per_timestep_embedding"] = ema_output

            returnables["per_series_summary_embedding"] = self.avg_pool(
                self.proj(ema_output)
            )

            return returnables

    def create_forecast(
        self,
        T_context_history,
        T_context_prompt,
        V_context_history,
        V_context_prompt,
        T_heldout_history,
        T_heldout_prompt,
        V_heldout_history,
        supress_warnings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Use this method in inference to create forecasts for the time series data.
        Args:
            T_context_history: abstract time context-series history  [batch_size, context_size, history_length, 1]
            T_context_prompt: abstract time context-series future   [batch_size, context_size, future_length, 1]
            V_context_history: Z-2std-normalized context values history  [batch_size, context_size, history_length, 1]
            V_context_prompt:  Z-2std-normalized context values future   [batch_size, context_size, future_length, 1]
            T_heldout_history: abstract time heldout-series history   [batch_size, heldout_size, history_length, 1]
            T_heldout_prompt:  abstract time heldout-series future [batch_size, heldout_size, future_length, 1]
            V_heldout_history: Z-2std-normalized heldout values history   [batch_size, heldout_size, history_length, 1]
        Returns:
            dict[str, torch.Tensor]: Dictionary containing the forecasts.
                - "forecast": decoded logits for bins of forecast   [batch_size, heldout_size, future_length, n_outputs]
                - "latent_forecast": latent next stats   [batch_size, heldout_size, future_length, d_model]
                - "backcast": decoded logits for bins of forecast: [batch_size, heldout_size, history_length, n_outputs]

        """

        # verify inputs

        assert not self.training, "This method should only be called in eval mode"

        assert (
            len(T_context_history.shape)
            == len(T_context_prompt.shape)
            == len(V_context_history.shape)
            == len(V_context_prompt.shape)
            == len(T_heldout_history.shape)
            == len(T_heldout_prompt.shape)
            == len(V_heldout_history.shape)
            == 4
        ), "All inputs are 4d tensors"

        assert (
            T_context_history.shape[0]
            == T_context_prompt.shape[0]
            == V_context_history.shape[0]
            == V_context_prompt.shape[0]
            == T_heldout_history.shape[0]
            == T_heldout_prompt.shape[0]
            == V_heldout_history.shape[0]
        ), "All inputs must have the same batch size"

        assert (
            T_context_history.shape[1]
            == T_context_prompt.shape[1]
            == V_context_history.shape[1]
            == V_context_prompt.shape[1]
        ), "All context inputs must have the same context size"

        assert (
            T_heldout_history.shape[1]
            == T_heldout_prompt.shape[1]
            == V_heldout_history.shape[1]
        ), "All heldout inputs must have the same context size"

        assert (
            T_context_history.shape[2]
            == V_context_history.shape[2]
            == T_heldout_history.shape[2]
            == V_heldout_history.shape[2]
        ), "All history inputs must have the same sequence length"

        assert (
            T_context_prompt.shape[2]
            == V_context_prompt.shape[2]
            == T_heldout_prompt.shape[2]
        ), "All future inputs must have the same sequence length"

        assert (
            T_context_history.shape[3]
            == T_context_prompt.shape[3]
            == V_context_history.shape[3]
            == V_context_prompt.shape[3]
            == T_heldout_history.shape[3]
            == T_heldout_prompt.shape[3]
            == V_heldout_history.shape[3]
            == 1
        ), "All inputs must have the same number of features"

        assert all([x > 1 for x in V_heldout_history.shape[:-1]]) and all(
            [x > 1 for x in T_context_prompt.shape[:-1]]
        ), "All dimensions except the last one must be > 1"

        for V in [
            V_context_history,
            V_context_prompt,
            V_heldout_history,
        ]:
            percentage_out_of_bounds = (V.abs() > 3.5).float().mean()

            if percentage_out_of_bounds > 0.1 and not supress_warnings:
                print(
                    "Warning: ",
                    percentage_out_of_bounds,
                    " of the input values are out of bounds (-3.5, 3.5) for decoder bins",
                )

        # do the inference

        with torch.no_grad():

            # embed context

            embedding_context, _ = self.TS_encoder(
                torch.cat([T_context_history, T_context_prompt], dim=-2),
                torch.cat([V_context_history, V_context_prompt], dim=-2),
            )
            mean_context = self.avg_pool(self.proj(embedding_context))

            # embed heldout

            embedding_heldout_history, prompt = self.TS_encoder(
                T_heldout_history, V_heldout_history, T_heldout_prompt
            )

            # predict the next latent state

            pred = self.pfn(mean_context, prompt)

            # decode the latent states

            prediction_raw = self.head_raw(pred.detach())

            returnables = dict(forecast=prediction_raw)

            returnables["latent_forecast"] = pred

            returnables["backcast"] = self.head_raw(embedding_heldout_history)

        return returnables

    def forward(
        self,
        T_context_history,
        T_context_prompt,
        V_context_history,
        V_context_prompt,
        T_heldout_history,
        T_heldout_prompt,
        V_heldout_history,
        V_heldout_prompt=None,  # for creating the latent target during training
        predict_all_heads: bool = False,  # returns all the returnables
        backcast: bool = False,  # returns the backcast for plotting
        **kwargs
    ):
        """
        Util method used by training, eval and inference scripts. Not part of public API.
        """

        # embed context

        embedding_context, _ = self.TS_encoder(
            torch.cat([T_context_history, T_context_prompt], dim=-2),
            torch.cat([V_context_history, V_context_prompt], dim=-2),
        )
        mean_context = self.avg_pool(self.proj(embedding_context))

        # embed heldout

        embedding_heldout_history, prompt = self.TS_encoder(
            T_heldout_history, V_heldout_history, T_heldout_prompt
        )

        # predict

        pred = self.pfn(mean_context, prompt)

        noise_fn = torch.randn_like if self.training else torch.zeros_like
        noise = noise_fn(pred.detach()) * self.train_noise

        # decode

        prediction_raw = self.head_raw(pred.detach() + noise)

        returnables = dict(forecast=prediction_raw)

        if predict_all_heads:
            with torch.no_grad():

                # create latent target

                ema = self.TS_ema(
                    torch.cat([T_heldout_history, T_heldout_prompt], dim=-2),
                    torch.cat([V_heldout_history, V_heldout_prompt], dim=-2),
                )[0]

                returnables["latent_target"] = ema[
                    :, :, -V_heldout_prompt.shape[-2] :, :
                ]

                # extra returnables

                returnables["latent_full"] = ema

                returnables["latent_history"] = embedding_heldout_history

                returnables["bypass"] = prompt

            returnables["latent_prediction"] = pred

            # system identification

            domain_identification_prediction_context = self.last_layer_di(
                self.head_di(mean_context)
            )

            heldout = torch.cat([embedding_heldout_history, pred], dim=-2)

            mean_heldout = self.avg_pool(self.proj(heldout))

            returnables["avg"] = torch.cat([mean_context, mean_heldout], dim=-2)

            domain_identification_prediction_heldout = self.last_layer_di(
                self.head_di(mean_heldout)
            )

            returnables["domain_identification_prediction_context"] = (
                domain_identification_prediction_context
            )
            returnables["domain_identification_prediction_heldout"] = (
                domain_identification_prediction_heldout
            )

        if backcast:  # for plotting
            with torch.no_grad():
                returnables["backcast"] = self.head_raw(embedding_heldout_history)

        return returnables
