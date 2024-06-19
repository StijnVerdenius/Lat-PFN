from util.config_util import ShapeConfig

from data.base_dataset import BaseDataset
from data.real.utils.registry import real_time_maps
from util.preprocess import calendar_embedding
from util.config_util import V_normalize
from util.preprocess import smooth
import json

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Any, Union


class RealCSVDataset(Dataset):
    """
    Dataset definition for real data, used in experiments.
    Args:
        - context_sources: List of paths to context time series
        - target_sources: List of paths to target time series
        - shape: Shape of the dataset:
            - n_context: Number of context samples
            - n_sequence: Number of sequence samples
            - n_features: Number of features
            - n_prompt: Number of prompt samples
            - n_history: Number of history samples
        - dataset_name: Name of the dataset
        - disable_context: Bool flag to zero-out the context
        - device: Device to use
        - T_mapping: Function to map real time to abstract time
        - is_train: Bool flag to indicate if the dataset is for training
        - stride: Stride for sampling along sequence dimension
        - abstract_time: Bool flag to indicate whether dataset should use abstract time dimension
        - smoother: Smoothing factor for the time series
        - denormalize_targets: Bool flag to denormalize the targets
        - name: Name of the dataset
    """
    def __init__(
        self,
        context_sources: List[str],
        target_sources: List[str],
        shape: ShapeConfig,
        dataset_name: str,
        disable_context: bool = False,
        device: str = "cuda",
        V_mapping=lambda Dv: (Dv - Dv.mean(dim=1).unsqueeze(-1))
        / (Dv.std(dim=1) * 2).unsqueeze(-1),  # todo: make configurable
        T_mapping=lambda t: t * 2 * torch.pi,
        is_train: bool = False,
        stride: int = 1,
        abstract_time: bool = True,
        smoother:int = 1,
        denormalize_targets: bool = False,
        name: str = None,
        **kwargs
    ):
        super().__init__()

        self.name = name

        self.target_sources = target_sources
        self.context_sources = context_sources
        self.is_train = is_train
        self.abstract_time = abstract_time
        self.dataset_name = dataset_name
        self.real_time_map = real_time_maps[dataset_name]
        self.smoother = smoother
        self.denormalize_targets = denormalize_targets

        self.shape = shape
        self.history_length = self.shape.n_history
        self.sequence_length = self.shape.n_sequence
        self.prompt_length = self.shape.n_prompt

        self.device = device
        self.disable_context = disable_context

        self.V_mapping = V_normalize # using norm function directly from utils, not via config
        self.T_mapping = T_mapping

        self.start = 0
        self.stride = stride

        self.full_context_series = torch.stack(
            [
                torch.tensor(
                    self._get_series(
                        source=source, total_sequence_length=self.sequence_length
                    )
                )
                .unsqueeze(-1)
                .float()
                for source in self.context_sources
            ],
            dim=0,
        ).to(self.device)

        self.full_context_time = torch.stack(
            [
                torch.tensor(
                    calendar_embedding(
                        self.real_time_map(
                            self._get_series(
                                source=source,
                                total_sequence_length=self.sequence_length,
                                col="t",
                            )
                        )
                    )
                )
                .unsqueeze(-1)
                for source in self.context_sources
            ],
            dim=0,
        )

        # If we need more datapoints than are available, interpolate
        if self.sequence_length > self.full_context_series.shape[1] - 2:

            self.new_S = self.sequence_length + 2
            S = self.full_context_series.shape[1]

            self.r = self.new_S / S

            self.indices = torch.linspace(0, S-1, steps=self.new_S).floor().long().unsqueeze(0).unsqueeze(-1)

            self.full_context_time = torch.gather(self.full_context_time.squeeze(), 1, self.indices.expand(self.full_context_time.shape[0], self.new_S, 5))

            self.full_context_series = self.rescale_series(
                self.full_context_series.unsqueeze(0), self.new_S
            ).squeeze(0)

        self.full_heldout_series = torch.stack(
            [
                torch.tensor(
                    self._get_series(
                        source=source, total_sequence_length=self.sequence_length
                    )
                )
                .unsqueeze(-1)
                .float()
                for source in self.target_sources
            ],
            dim=0,
        ).to(self.device)

        self.full_heldout_time = torch.stack(
            [
                torch.tensor(
                    calendar_embedding(
                        self.real_time_map(
                            self._get_series(
                                source=source,
                                total_sequence_length=self.sequence_length,
                                col="t",
                            )
                        )
                    )
                , dtype=torch.int64)
                .unsqueeze(-1)
                for source in self.target_sources
            ],
            dim=0,
        )

        if self.sequence_length > self.full_heldout_series.shape[1] - 2:
            self.full_heldout_series = self.rescale_series(
                self.full_heldout_series.unsqueeze(0), self.sequence_length + 2
            ).squeeze(0)

            self.full_heldout_time = torch.gather(self.full_heldout_time.squeeze(), 1, self.indices.expand(self.full_heldout_time.shape[0], self.new_S, 5))


        self.length = (
            max((self.full_context_series.shape[1] - self.sequence_length), 1)
            // self.stride
        )

        assert self.length > 1, f"need at least length=2"

        self.counter = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> Any:
        T_context, V_context, _ = self.get_a_context()
        T_heldout, V_heldout = self.get_target()

        self.start += self.stride

        if self.abstract_time:
            V_context, _, _ = V_normalize(V_context)
            T_heldout = self.T_mapping(T_heldout)
            T_context = self.T_mapping(T_context)

        T_context_history = T_context[:, : self.history_length, :]
        T_context_prompt = T_context[:, self.history_length :, :]
        V_context_history = V_context[:, : self.history_length, :]
        V_context_prompt = V_context[:, self.history_length :, :]

        T_heldout_history = T_heldout[:, : self.history_length, :]
        T_heldout_prompt = T_heldout[:, self.history_length :, :]
        V_heldout_history = V_heldout[:, : self.history_length, :]
        V_heldout_prompt = V_heldout[:, self.history_length :, :]

        V_heldout_history = smooth(V_heldout_history.unsqueeze(0), self.smoother).squeeze(0)

        if self.abstract_time:
            V_context_history = smooth(V_context_history.unsqueeze(0), self.smoother).squeeze(0)
            V_context_prompt = smooth(V_context_prompt.unsqueeze(0), self.smoother).squeeze(0)


        if self.disable_context:
            # zero out the context
            T_context_history, T_context_prompt, V_context_history, V_context_prompt = (
                torch.zeros_like(T_context_history),
                torch.zeros_like(T_context_prompt),
                torch.zeros_like(V_context_history),
                torch.zeros_like(V_context_prompt),
            )

        if self.abstract_time:
            V_heldout_history, mean, std = V_normalize(V_heldout_history)
            if not self.denormalize_targets:
                V_heldout_prompt, _, _ =  V_normalize(V_heldout_prompt, mean, std)
            out = (
                T_context_history.float(),  # example timesteps in backcast
                T_context_prompt.float(),  # example decoder prompts
                V_context_history.float(),  # example backcast time series
                V_context_prompt.float(),  # example targets
                T_heldout_history.float(),  # timesteps in backcast
                T_heldout_prompt.float(),  # decoder prompts for forcasting
                V_heldout_history.float(),  # backcast time series
                V_heldout_prompt.float(),  # targets
            )

            if self.denormalize_targets:
                out += (mean, std)

        else:
            if not self.denormalize_targets:
                V_heldout_history, mean, std = V_normalize(V_heldout_history)
                V_heldout_prompt, _, _ =  V_normalize(V_heldout_prompt, mean, std)

            out = (
                T_context_history,  # example timesteps in backcast
                T_context_prompt,  # example decoder prompts
                V_context_history,  # example backcast time series
                V_context_prompt,  # example targets
                T_heldout_history,  # timesteps in backcast
                T_heldout_prompt,  # decoder prompts for forcasting
                V_heldout_history,  # backcast time series
                V_heldout_prompt,  # targets
            )

        return out

    def get_a_context(self):
        """
        Get context series from real data sources
        """
        if self.abstract_time:
            Dt = torch.stack(
                [
                    torch.linspace(0, 1, self.sequence_length).unsqueeze(-1).float()
                    for x in self.context_sources
                ],
                dim=0,
            ).to(self.device)
        else:
            Dt = self.full_context_time[:, self.start : self.start + self.sequence_length, :].to(self.device)

        Dv = self.full_context_series[
            :, self.start : self.start + self.sequence_length, :
        ]

        return Dt, Dv, None

    def get_target(self):
        """
        Get held-out series from real data sources
        """
        if self.abstract_time:
            T_heldouts = torch.stack(
                [
                    torch.linspace(0, 1, self.sequence_length).unsqueeze(-1).float()
                    for _ in self.target_sources
                ],
                dim=0,
            ).to(self.device)
        else:
            T_heldouts = self.full_heldout_time[:, self.start : self.start + self.sequence_length, :].to(self.device)

        V_heldouts = self.full_heldout_series[
            :, self.start : self.start + self.sequence_length, :
        ]

        return T_heldouts, V_heldouts

    def _get_series(self, source, total_sequence_length, col="v"):
        if isinstance(source, str):
            return pd.read_csv(source)[col].values

        return pd.concat(
            [pd.read_csv(s)[col] for s in source], axis=0
        ).values

    def _get_total_sequence_length(self, sources: List[Union[str, List[str]]]):
        if isinstance(sources[0], str):
            return self.sequence_length
        ls = []
        for source in sources:
            for s in source:
                ls.append(pd.read_csv(s).shape[0])

        return sum(ls)

    def rescale_series(self, original_series: torch.Tensor, new_sequence_dim: int):
        feature_dim = original_series.shape[-1]

        interpolated = F.interpolate(
            original_series, [new_sequence_dim, feature_dim], mode="bicubic"
        )

        return interpolated
