from data.base_dataset import BasePrior
from util.config_util import ShapeConfig, dotdict
from data.priors.LaTPFN_prior.series_components import (
    generate_trend_component,
    generate_seasonal_component,
    generate_noise_component,
)
from data.priors.LaTPFN_prior.sampling import (
    double_sampling,
    triple_sampling,
    noise_scale_sampling,
)

from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical

import torch
import numpy as np


class LaTPFNDataset(BasePrior):
    """
    LaT-PFN Dataset definition
    Args:
        - shape: Shape of the dataset:
            - n_context: Number of context samples
            - n_sequence: Number of sequence samples
            - n_features: Number of features
        - hyperprior_params: Hyperprior ranges from which to sample the parameters
        - batch_size: Batch size
        - length: Length of the dataset
        - is_train: Whether the dataset is for training
        - device: Device to run the code
        - return_components: Whether to return the components of the time series simulation -- for debugging and visualization purposes
        - scale_noise: Whether to scale the noise according to the trend
        - separate_noise: Whether to separate the noise from the values of the time series
    """
    def __init__(
        self,
        shape: ShapeConfig,
        hyperprior_params: dotdict,
        batch_size: int,
        length: int,
        is_train: bool = False,
        roll_out: float = 2 * np.pi,
        device: str = "cpu",
        return_components: bool = False,
        scale_noise: bool = False,
        separate_noise: bool = True,
        **kwargs,
    ): 
        self.shape = shape
        self.batch_size = batch_size
        self.length = length
        self.device = device
        self.n_context = shape.n_context
        self._roll_out = roll_out

        self.hyperprior_params = hyperprior_params
        self.return_components = return_components
        self.scale_noise = scale_noise
        self.separate_noise = separate_noise

        super().__init__(
            shape=shape,
            is_train=is_train,
            batch_size=batch_size,
            separate_noise=self.separate_noise,
            **kwargs,
        )

    def normalize_components(
        self,
        components,
    ):
        """
        Normalize the components to be between 0 and 1 to be used as targets for the model
        """

        new_components = {}

        hpp = self.hyperprior_params

        # euclidean normalization

        for param, min_val, max_val in [
            ("annual_param", hpp.a_min, hpp.a_max),
            ("monthly_param", hpp.m_min, hpp.m_max),
            ("weekly_param", hpp.w_min, hpp.w_max),
            ("trend_lin", hpp.trend_lin_min, hpp.trend_lin_max),  # noqa
            ("trend_exp", hpp.trend_exp_min, hpp.trend_exp_max),  # noqa
            ("offset_lin", hpp.offset_lin_min, hpp.offset_lin_max),  # noqa
            ("offset_exp", hpp.offset_exp_min, hpp.offset_exp_max),  # noqa
        ]:
            new_components[param] = normalize_component(
                components[param], min_val, max_val
            )

        # log normalization

        for param, min_val, max_val in [
            ("noise_k", hpp.noise_k_min, hpp.noise_k_max),
            ("resolution", hpp.resolution_min, hpp.resolution_max),
        ]:
            new_components[param] = log_normalize_component(
                components[param], min_val, max_val
            )

        new_components["noise_scale"] = components["noise_scale"]

        return {key: torch.clamp(val, 0, 1) for key, val in new_components.items()}

    def get_a_context(self, *args, **kwargs):
        component_params = self.sample_from_hyperpriors()

        t, v, noise = make_multiple_series(
            n_context=self.shape.n_context,
            sequence_length=self.shape.n_sequence,
            num_features=1,
            device=self.device,
            component_params=component_params,
            return_components=self.return_components,
            scale_noise=False,
        )

        component_params = self.normalize_components(component_params)

        t = t.unsqueeze(-1)

        if self.separate_noise:
            return t, torch.stack((v, noise), dim=-1), component_params

        return t, (v * noise).unsqueeze(-1), component_params

    def __len__(self):
        return self.length * self.batch_size

    def sample_from_hyperpriors(self, *args, **kwargs):
        hpp = self.hyperprior_params
        device = self.device
        n_context = self.shape.n_context
        n_sequence = self.shape.n_sequence

        result = dotdict()

        for param, min_val, max_val, fixed_variance in [
            ("annual_param", hpp.a_min, hpp.a_max, hpp.a_fixed_variance),
            ("monthly_param", hpp.m_min, hpp.m_max, hpp.m_fixed_variance),
            ("weekly_param", hpp.w_min, hpp.w_max, hpp.w_fixed_variance),
            ("frequency_zero_inflation", hpp.f_zi_min, hpp.f_zi_max, hpp.f_zi_fixed_variance),  # noqa
            ("trend_lin", hpp.trend_lin_min, hpp.trend_lin_max, hpp.trend_lin_fixed_variance),  # noqa
        ]:
            result[param] = triple_sampling(
                hyperprior_min=min_val,
                hyperprior_max=max_val,
                fixed_variance=fixed_variance,
                num_samples=n_context,
                device=device,
            )

        # make it equally likely to have a positive or negative exp trend

        mm = hpp.trend_exp_multiplier  
        f_exp = lambda x: 2 ** ((x - 1) * mm)
        f_exp_inv = lambda x: (torch.log2(x) / mm) + 1

        result.trend_exp = f_exp_inv(
            triple_sampling(
                hyperprior_min=f_exp(torch.scalar_tensor(hpp.trend_exp_min)),
                hyperprior_max=f_exp(torch.scalar_tensor(hpp.trend_exp_max)),
                fixed_variance=hpp.trend_exp_fixed_variance,
                num_samples=n_context,
                device=device,
            )
        )

        # ensure consistent sign for trends
    
        median_lin_sign = result.trend_lin.median().sign()
        result.trend_lin = result.trend_lin.abs() * median_lin_sign

        assert (result.trend_lin >= 0).all() or (
            result.trend_lin <= 0
        ).all(), f"non-consistent sign {result.trend_lin=} in trend_lin"

        median_exp_sign = (result.trend_exp - 1).median().sign()
        result.trend_exp = (result.trend_exp - 1).abs() * median_exp_sign + 1

        assert (result.trend_exp >= 1).all() or (
            result.trend_exp <= 1
        ).all(), f"non-consistent {result.trend_exp=} in trend_exp"

        # sub-context-specific params

        result.noise_k = double_sampling(
            hyperprior_min=hpp.noise_k_min,
            hyperprior_max=hpp.noise_k_max,
            num_samples=n_context,
            device=device,
        )
        result.noise_scale = noise_scale_sampling(n_context, device=device)

        # domain-specific params

        result.discreteness = (
            Uniform(hpp.discreteness_min, hpp.discreteness_max)
            .sample([n_context])
            .to(device)
        )

        result.bias_zi = (
            Uniform(hpp.bias_zi_min, hpp.bias_zi_max).sample([n_context]).to(device)
        )

        result.amplitude = (
            Uniform(hpp.amplitude_min, hpp.amplitude_max).sample([n_context]).to(device)
        )

        result.non_negative = (
            Categorical(
                torch.tensor([1 - hpp.non_negative_prob, hpp.non_negative_prob])
            )
            .sample()
            .to(device)
            .repeat(n_context)
        )

        result.offset_lin = (
            Uniform(hpp.offset_lin_min, hpp.offset_lin_max)
            .sample([n_context])
            .to(device)
        )

        result.offset_exp = (
            Uniform(hpp.offset_exp_min, hpp.offset_exp_max)
            .sample([n_context])
            .to(device)
        )

        result.harmonics = torch.randint(hpp.harmonics_min, hpp.harmonics_max, (3,)).to(
            device
        )

        # keep the n-days at a set median

        mm = hpp.resolution_multiplier
        f_res = lambda x: torch.log2(x * mm + 1)
        f_res_inv = lambda x: (2**x - 1) / mm

        result.resolution = (
            f_res_inv(
                Uniform(
                    f_res(torch.scalar_tensor(hpp.resolution_min)),
                    f_res(torch.scalar_tensor(hpp.resolution_max)),
                ).sample()
            )
            .to(device)
            .repeat(n_context)
        )

        result.n_units = torch.ceil(n_sequence / result.resolution)

        return result


def make_multiple_series(
    n_context: int,
    sequence_length: int,
    num_features: int,
    device: str,
    component_params: dotdict,
    return_components: bool = False,  # For debugging and visualization purposes
    scale_noise: bool = True,
    equal_spacing: bool = True,
):
    if not equal_spacing:
        x = (
            (
                torch.rand(n_context, sequence_length, num_features, device=device)
                * component_params.n_units.unsqueeze(-1).unsqueeze(-1)
            )
            .squeeze()
            .to(device)
        )

        x, _ = torch.sort(x, dim=1)
    else:
        x = torch.linspace(0, 1, sequence_length, device=device).unsqueeze(0).repeat(
            n_context, 1
        ).unsqueeze(-1) * component_params.n_units.unsqueeze(-1).unsqueeze(-1)
        x = x.squeeze().to(device)

    trend_comp_total, trend_comp_linear, trend_comp_exponential = (
        generate_trend_component(
            trend_linear_scaler=component_params.trend_lin,
            trend_exp_scaler=component_params.trend_exp,
            offset_linear=component_params.offset_lin,
            offset_exp=component_params.offset_exp,
            x=x,
        )
    )

    seasonal_components = generate_seasonal_component(
        annual_param=component_params.annual_param,
        monthly_param=component_params.monthly_param,
        weekly_param=component_params.weekly_param,
        x=x,
        n_units=component_params.n_units,
        n_harmonics=component_params.harmonics,
        device=device,
    )

    total_seasonality, annual_seasonality, monthly_seasonality, weekly_seasonality = (
        seasonal_components[:, :, 0],
        seasonal_components[:, :, 1],
        seasonal_components[:, :, 2],
        seasonal_components[:, :, 3],
    )

    noisless_values = trend_comp_total * total_seasonality

    noise_mean = torch.ones_like(component_params.noise_k)

    weibull_noise_term = generate_noise_component(
        k=component_params.noise_k,
        noise_mean=noise_mean,
        shape=(x.shape[0], x.shape[1]),
        device=device,
    )

    noise = 1 + component_params.noise_scale.unsqueeze(-1) * (
        weibull_noise_term - noise_mean.unsqueeze(-1)
    )

    if scale_noise:
        noise = noise * trend_comp_total

    if return_components:
        return (
            x,
            trend_comp_total,
            trend_comp_linear,
            trend_comp_exponential,
            total_seasonality,
            annual_seasonality,
            monthly_seasonality,
            weekly_seasonality,
            noise,
            noisless_values,
            component_params,
        )
    return x, noisless_values, noise


def normalize_component(component, min_val, max_val):
    return (component - min_val) / (max_val - min_val)


def log_normalize_component(component, min_val, max_val):
    return (component.log2() - torch.log2(torch.tensor(min_val))) / (
        torch.log2(torch.tensor(max_val)) - torch.log2(torch.tensor(min_val))
    )
