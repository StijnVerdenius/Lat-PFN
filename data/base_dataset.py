import torch
from torch.utils.data import Dataset
from util.sizes import randomly_divide_shape
from util.config_util import ShapeConfig
from abc import abstractmethod
import logging

logger = logging.getLogger(__name__)



class BaseDataset(Dataset):
    """
    Implements a base dataset for any timeseries context dataset or prior / simulation
    """

    def __init__(
        self,
        shape: ShapeConfig,
        batch_size: int,
        is_train: bool = True,
        T_mapping: callable = lambda x: x,  # use for mapping to abstract time space
        V_mapping: callable = lambda x: x,  # use for mapping to normalize,
        kwargs_get_a_context: dict = {},
        train_percentage_backcast: float = 0.25,
        separate_noise: bool = False,
        abstract_time: bool = True,
        disable_context: bool = False,
        backcast_samples: bool = False,
        **kwargs,
    ):
        print("Ignoring kwargs:", self, kwargs)
        self.shape = shape
        self.is_train = is_train
        self.T_mapping = T_mapping
        self.V_mapping = V_mapping
        self.kwargs_get_a_context = kwargs_get_a_context
        self.train_percentage_backcast = train_percentage_backcast
        self.batch_size = batch_size
        self.separate_noise = separate_noise
        self.abstract_time = abstract_time
        self.disable_context = disable_context
        self.backcast_samples = backcast_samples
        if is_train:
            P = shape.n_prompt
            p1 = int((1 - self.train_percentage_backcast) * shape.n_prompt)
            p2 = int(self.train_percentage_backcast * P)
            assert (
                p1 + p2 == P
            ), f"{self.train_percentage_backcast=} is not compatible with {shape.n_prompt=} due to numerical errors"

    @abstractmethod
    def get_a_context(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """method returns time series from within 1 context"""
        pass

    def __getitem__(self, _index):
        """
        Returns an item within a batch.
        An item is defined as a tuple of tensors:
        - T_context_history: historical time-axis for context series
        - T_context_prompt: time-axis prompts for context series
        - V_context_history: historical values for context series
        - V_context_prompt: context targets
        - T_heldout_history: historical time-axis for held-out series
        - T_heldout_prompt: time-axis prompts for held-out series
        - V_heldout_history: historical values for held-out series
        - V_heldout_prompt: held-out targets
        - hyperprior_params_context: hyperprior parameters that generated context context - only for SI loss during training.
        - hyperprior_params_heldout: hyperprior parameters that generated held-out series - only for SI loss during training.
        """

        # get a context
        T_context, V_context, hyperprior_params = self.get_a_context(
            **self.kwargs_get_a_context
        )

        if self.separate_noise:
            noise = V_context[:, :, 1].unsqueeze(-1)
            V_context = V_context[:, :, 0].unsqueeze(-1)

        if self.abstract_time:
            # sort by T_context
            idx = T_context.argsort(dim=-2)
            T_context = T_context.gather(dim=-2, index=idx)
            V_context = V_context.gather(dim=-2, index=idx)

        # apply the mappings
        T_context = self.T_mapping(T_context)
        V_context = self.V_mapping(V_context)

        # derive some fields

        device = T_context.device
        n_sequence = T_context.shape[1]
        n_context = T_context.shape[0]
        n_heldout = int(n_context * self.shape.percent_heldout)
        n_prompt = int(n_sequence * self.shape.percent_prompt)
        n_history = int(n_sequence * self.shape.percent_history)

        # determine the indices

        idx_context, idx_heldout = randomly_divide_shape(
            T_context.shape[0], n_heldout, device=device
        )

        assert (
            not torch.isin(idx_context, idx_heldout).any() and 
            not torch.isin(idx_heldout, idx_context).any()
        ), "context and heldout indices should be disjoint"

        # divide the history
        all_steps = torch.arange(n_sequence, dtype=torch.long, device=device)
        idx_prompt, idx_history = all_steps[-n_prompt:], all_steps[:-n_prompt]

        # put the past in the negative range and the future in the positive range
        prediction_moment = idx_prompt.min()
        T_context = T_context - T_context[:, prediction_moment, :].unsqueeze(-1)

        # when training, do it differently for heldout
        if self.backcast_samples:
            idx_heldout_prompt = torch.cat(
                (
                    torch.multinomial(
                        torch.ones(n_history, device=device)
                        / n_history,  # uniform probability
                        int(n_prompt * self.train_percentage_backcast),
                        replacement=False,
                    ),  # sampled from the history
                    all_steps[
                        -int(n_prompt * (1 - self.train_percentage_backcast)) :
                    ],  # deterministic from the future
                )
            )
            idx_heldout_history = all_steps[
                ~torch.isin(all_steps, idx_heldout_prompt)
            ]  # complement of idx_heldout_prompt
        else:
            idx_heldout_prompt = idx_prompt
            idx_heldout_history = idx_history

        # sort all idxs

        idx_context = idx_context.sort().values
        idx_heldout = idx_heldout.sort().values
        idx_history = idx_history.sort().values
        idx_prompt = idx_prompt.sort().values
        idx_heldout_history = idx_heldout_history.sort().values
        idx_heldout_prompt = idx_heldout_prompt.sort().values

        # split the context into the parts

        T_context_history, T_context_prompt, T_heldout_history, T_heldout_prompt = (
            T_context[idx_context, ...][:, idx_history, :],
            T_context[idx_context, ...][:, idx_prompt, :],
            T_context[idx_heldout, ...][:, idx_heldout_history, :],
            T_context[idx_heldout, ...][:, idx_heldout_prompt, :],
        )
        V_context_history, V_context_prompt, V_heldout_history, V_heldout_prompt = (
            V_context[idx_context, ...][:, idx_history, :],
            V_context[idx_context, ...][:, idx_prompt, :],
            V_context[idx_heldout, ...][:, idx_heldout_history, :],
            V_context[idx_heldout, ...][:, idx_heldout_prompt, :],
        )


        if self.separate_noise:
            V_context_history = (
                V_context_history * noise[idx_context, ...][:, idx_history, :]
            )
            V_heldout_history = (
                V_heldout_history * noise[idx_heldout, ...][:, idx_heldout_history, :]
            )

            V_context_prompt = (
                V_context_prompt * noise[idx_context, ...][:, idx_prompt, :]
            )
            if not self.is_train:
                # For validation, add noise to all
                V_heldout_prompt = (
                    V_heldout_prompt * noise[idx_heldout, ...][:, idx_heldout_prompt, :]
                )

        if self.disable_context:
            # zero out the context
            T_context_history, T_context_prompt, V_context_history, V_context_prompt = (
                torch.zeros_like(T_context_history),
                torch.zeros_like(T_context_prompt),
                torch.zeros_like(V_context_history),
                torch.zeros_like(V_context_prompt),
            )

        result = [
            T_context_history.float(),  # example timesteps in backcast
            T_context_prompt.float(),  # example decoder prompts
            V_context_history.float(),  # example backcast time series
            V_context_prompt.float(),  # example targets
            T_heldout_history.float(),  # timesteps in backcast
            T_heldout_prompt.float(),  # decoder prompts for forcasting
            V_heldout_history.float(),  # backcast time series
            V_heldout_prompt.float(),  # targets
        ]

        if hyperprior_params is not None:
            hyperprior_params_context = {
                k: v[idx_context, ...].float()
                for k, v in hyperprior_params.items()
                if v.shape[0] == n_context
            }
            hyperprior_params_heldout = {
                k: v[idx_heldout, ...].float()
                for k, v in hyperprior_params.items()
                if v.shape[0] == n_context
            }
            result.extend([hyperprior_params_context, hyperprior_params_heldout])
            

        return tuple(result)

class BasePrior(BaseDataset):
    """
    Implements a base prior
    """

    hyperpriors: dict

    @classmethod
    def with_hyperpriors(cls, hyperpriors: dict, *args, **kwargs):
        """
        Use this constructor when building a prior with configurable hyperpriors
        """
        instance = cls(*args, **kwargs, hyperpriors=hyperpriors)
        instance.hyperpriors = hyperpriors
        return instance

    @abstractmethod
    def sample_from_hyperpriors(self, *args, **kwargs) -> dict:
        pass
