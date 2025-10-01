# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from scipy.io import loadmat
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput
from diffusers.utils import BaseOutput, is_scipy_available, logging
from pathlib import Path



@dataclass
class STORKSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


current_file = Path(__file__)
CONSTANTSFOLDER = f"{current_file.parent}/STORK_constants"





class STORKScheduler(SchedulerMixin, ConfigMixin):
    """
    `STORKScheduler` uses modified stabilized Runge-Kutta method for the backward ODE in the diffusion or flow matching models.
    This include the original STORK method and the modified STORK++ methods.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
        use_dynamic_shifting (`bool`, defaults to False):
            Whether to apply timestep shifting on-the-fly based on the image resolution.
        base_shift (`float`, defaults to 0.5):
            Value to stabilize image generation. Increasing `base_shift` reduces variation and image is more consistent
            with desired output.
        max_shift (`float`, defaults to 1.15):
            Value change allowed to latent vectors. Increasing `max_shift` encourages more variation and image may be
            more exaggerated or stylized.
        base_image_seq_len (`int`, defaults to 256):
            The base image sequence length.
        max_image_seq_len (`int`, defaults to 4096):
            The maximum image sequence length.
        invert_sigmas (`bool`, defaults to False):
            Whether to invert the sigmas.
        shift_terminal (`float`, defaults to None):
            The end value of the shifted timestep schedule.
        use_karras_sigmas (`bool`, defaults to False):
            Whether to use Karras sigmas for step sizes in the noise schedule during sampling.
        use_exponential_sigmas (`bool`, defaults to False):
            Whether to use exponential sigmas for step sizes in the noise schedule during sampling.
        use_beta_sigmas (`bool`, defaults to False):
            Whether to use beta sigmas for step sizes in the noise schedule during sampling.
        solver_order (`int`, defaults to 2):
            The STORK order which can be `2` or `4`. It is recommended to use `solver_order=2` uniformly.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process) or `flow_prediction`.
        time_shift_type (`str`, defaults to "exponential"):
            The type of dynamic resolution-dependent timestep shifting to apply. Either "exponential" or "linear".
        derivative_order (`int`, defaults to 1):
            The order of the Taylor expansion derivative to use for the sub-step velocity approximation. Only supports 1, 2 or 3.
        s (`int`, defaults to 50):
            The number of sub-steps to use in the STORK.
        precision (`str`, defaults to "float32"):
            The precision to use for the scheduler; supports "float32", "bfloat16", or "float16".
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        stopping_eps: float = 1e-2,
        solver_order: int = 4,
        prediction_type: str = "epsilon",
        time_shift_type: str = "exponential",
        derivative_order: int = 1,
        s: int = 50,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
    ):
        
        super().__init__()
        # if prediction_type == "flow_prediction" and sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
        #     raise ValueError(
        #         "Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used."
        #     )
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError("`time_shift_type` must either be 'exponential' or 'linear'.")
 
        # We manually enforce precision to float32 for numerical issues.Add commentMore actions
        self.np_dtype = np.float32
        self.dtype = torch.float32


        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=self.np_dtype)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=self.dtype)
        sigmas = timesteps / num_train_timesteps



        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = None    #sigmas * num_train_timesteps
        self._step_index = None
        self._begin_index = None
        self._shift = shift
        self.sigmas = sigmas #.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        # Store the predictions for the velocity/noise for higher order derivative approximations
        self.velocity_predictions = []
        self.noise_predictions = []
        self.s = s
        self.derivative_order = derivative_order

        self.solver_order = solver_order
        self.prediction_type = prediction_type


        # Set the betas for noise-based models
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")
        

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0
        
        # Noise-based models epsilon to avoid numerical issues
        self.stopping_eps = stopping_eps




    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        timesteps: Optional[List[float]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            sigmas (`List[float]`, *optional*):
                Custom values for sigmas to be used for each diffusion step. If `None`, the sigmas are computed
                automatically.
            mu (`float`, *optional*):
                Determines the amount of shifting applied to sigmas when performing resolution-dependent timestep
                shifting.
            timesteps (`List[float]`, *optional*):
                Custom values for timesteps to be used for each diffusion step. If `None`, the timesteps are computed
                automatically.
        """
        
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `True`")

        if sigmas is not None and timesteps is not None:
            if len(sigmas) != len(timesteps):
                raise ValueError("`sigmas` and `timesteps` should have the same length")

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length as num_inference_steps, if `num_inference_steps` is provided"
                )
        else:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)

        self.num_inference_steps = num_inference_steps

        if self.prediction_type == "epsilon":
            self.set_timesteps_noise(num_inference_steps, device)
        elif self.prediction_type == "flow_prediction":
            self.set_timesteps_flow_matching(num_inference_steps, device, sigmas, mu, timesteps)
        else:
            raise ValueError(f"Prediction type {self.prediction_type} is not yet supported")
        
        # Reset the step index and begin index
        self._step_index = None
        self._begin_index = None

        

    def set_timesteps_noise(self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference), for noise-based models.

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        seq = np.linspace(0, 1, self.num_inference_steps+1)
        seq[0] = self.stopping_eps
        seq = seq[:-1]
        seq = seq[::-1]


        # The following lines are for the uniform timestepping case
        self.dt = seq[0] - seq[1]
        seq = seq * self.config.num_train_timesteps
        seq[-1] = self.stopping_eps * self.config.num_train_timesteps
        self._timesteps = seq
        self.timesteps = torch.from_numpy(seq.copy()).to(device)


        self._step_index = None
        self._begin_index = None

        self.noise_predictions = []


    def set_timesteps_flow_matching(self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        timesteps: Optional[List[float]] = None,
    ):
        """
        Sets the discrete timesteps used for the flow matching based models (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            sigmas (`List[float]`, *optional*):
                Custom values for sigmas to be used for each diffusion step. If `None`, the sigmas are computed
                automatically.
            mu (`float`, *optional*):
                Determines the amount of shifting applied to sigmas when performing resolution-dependent timestep
                shifting.
            timesteps (`List[float]`, *optional*):
                Custom values for timesteps to be used for each diffusion step. If `None`, the timesteps are computed
                automatically.
        """
        self.num_inference_steps = num_inference_steps

        # 1. Prepare default sigmas
        is_timesteps_provided = timesteps is not None

        if is_timesteps_provided:
            timesteps = np.array(timesteps).astype(np.float32)

        if sigmas is None:
            if timesteps is None:
                timesteps = np.linspace(
                    self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
                )
            sigmas = timesteps / self.config.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas)

        # 2. Perform timestep shifting. Either no shifting is applied, or resolution-dependent shifting of
        #    "exponential" or "linear" type is applied
        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        # 3. If required, stretch the sigmas schedule to terminate at the configured `shift_terminal` value
        if self.config.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        # 4. If required, convert sigmas to one of karras, exponential, or beta sigma schedules
        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        # 5. Convert sigmas and timesteps to tensors and move to specified device
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        if not is_timesteps_provided:
            timesteps = sigmas * self.config.num_train_timesteps
        else:
            timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)

        # 6. Append the terminal sigma value.
        #    If a model requires inverted sigma schedule for denoising but timesteps without inversion, the
        #    `invert_sigmas` flag can be set to `True`. This case is only required in Mochi
        if self.config.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        else:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.timesteps = timesteps
        self.sigmas = sigmas


        # Create the dt list
        self.dt_list = self.sigmas[:-1] - self.sigmas[1:]
        self.dt_list = self.dt_list.reshape(-1)

        self.dt_list = self.dt_list.tolist()
        self.dt_list = torch.tensor(self.dt_list).to(self.dtype)

        self.velocity_predictions = []


    @property
    def shift(self):
        """
        The value used for shifting.
        """
        return self._shift

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index



    def set_shift(self, shift: float):
        self._shift = shift
    
    def set_begin_index(self, begin_index: int):
        """
        Set the begin index for the scheduler.

        Args:
            begin_index (`int`):
                The begin index to set.
        """
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(sample.device, dtype=self.dtype)
            timestep = timestep.to(sample.device, dtype=self.dtype)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps
    
    def index_for_timestep(self, timestep, schedule_timesteps):
        """
        Get the index for a given timestep in the schedule.

        Args:
            timestep (`torch.Tensor`):
                The timestep to find the index for.
            schedule_timesteps (`torch.Tensor`):
                The schedule timesteps.

        Returns:
            `int`:
                The index for the timestep.
        """
        # Find the closest timestep in the schedule
        indices = torch.searchsorted(schedule_timesteps, timestep, right=True)
        indices = torch.clamp(indices, 0, len(schedule_timesteps) - 1)
        return indices.item()
    


    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor = None,
        return_dict: bool = True,
        **kwargs
    ) -> torch.Tensor:
        '''
        One step of the STORK update for flow matching or noise-based diffusion models.

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~schedulers.STORKSchedulerOutput`] instead of a plain tuple.
                
        Returns:
            result (Union[Tuple, STORKSchedulerOutput]):
                The next sample in the diffusion chain, either as a tuple or as a [`~schedulers.STORKSchedulerOutput`]. The value is converted back to the original dtype of `model_output` to avoid numerical issues.
        '''
        original_model_output_dtype = model_output.dtype
        # Cast model_output and sample to "torch.float32" to avoid numerical issues
        model_output = model_output.to(self.dtype)
        sample = sample.to(self.dtype)
        # Move sample to model_output's device
        sample = sample.to(model_output.device)
        
        """
        self.velocity_predictions always contain upcasted model_output in torch.float32 dtype.
        """
        
        if self.prediction_type == "epsilon":
            if self.solver_order == 2:
                result = self.step_noise_2(model_output, timestep, sample, return_dict)
            elif self.solver_order == 4:
                result = self.step_noise_4(model_output, timestep, sample, return_dict)
            else:
                raise ValueError(f"Solver order {self.solver_order} is not yet supported for noise-based models")
        elif self.prediction_type == "flow_prediction":
            if self.solver_order == 1:
                result = self.step_flow_matching_1(model_output, timestep, sample, return_dict)
            elif self.solver_order == 2:
                result = self.step_flow_matching_2(model_output, timestep, sample, return_dict)
            elif self.solver_order == 4:
                result = self.step_flow_matching_4(model_output, timestep, sample, return_dict)
            else:
                raise ValueError(f"Solver order {self.solver_order} is not yet supported for flow matching models")
        else:
            raise ValueError(f"Prediction type {self.prediction_type} is not yet supported")
        
        # Convert the result back to the original dtype of model_output, as this result will be used as the next input to the model
        if return_dict:
            result.prev_sample = result.prev_sample.to(original_model_output_dtype)
        else:
            result = (result[0].to(original_model_output_dtype),)
        return result
        

    def step_flow_matching_1(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor = None,
        return_dict: bool = False
    ) -> torch.Tensor:
        # Initialize the step index if it's the first step
        if self._step_index is None:
            self._step_index = 0


        # Compute the startup phase or the derivative approximation for the main step
        if self._step_index == 0:
            img_next = sample - model_output * self.dt_list[self._step_index]
            self._step_index += 1
            self.velocity_predictions.append(model_output)

            if not return_dict:
                return (img_next,)
            return STORKSchedulerOutput(prev_sample=img_next)
        else:
            t = self.sigmas[self._step_index]
            t_start = torch.ones(model_output.shape, device=sample.device) * t
            t_next = self.sigmas[self._step_index + 1]
            
            h1 = self.dt_list[self._step_index-1]

            if self.derivative_order == 1:
                # Ensure h1 is a tensor for proper broadcasting
                h1_tensor = torch.tensor(h1, device=model_output.device, dtype=model_output.dtype)
                velocity_derivative = (self.velocity_predictions[-1] - model_output) / h1_tensor
                velocity_second_derivative = None
                velocity_third_derivative = None
            elif self.derivative_order == 2:
                # Ensure h1 and h2 are tensors for proper broadcasting
                h1_tensor = torch.tensor(h1, device=model_output.device, dtype=model_output.dtype)
                if self._step_index == 1:
                    img_next = sample - 1.5 * model_output * self.dt_list[self._step_index] + 0.5 * self.velocity_predictions[-1] * self.dt_list[self._step_index-1]
                    self._step_index += 1
                    self.velocity_predictions.append(model_output)

                    if not return_dict:
                        return (img_next,)
                    return STORKSchedulerOutput(prev_sample=img_next)
                else:
                    h2 = self.dt_list[self._step_index-2]
                    h2_tensor = torch.tensor(h2, device=model_output.device, dtype=model_output.dtype)
                    velocity_derivative = (-self.velocity_predictions[-2] + 4 * self.velocity_predictions[-1] - 3 * model_output) / (2 * h1_tensor)
                    velocity_second_derivative = 2 / (h1_tensor * h2_tensor * (h1_tensor + h2_tensor)) * (self.velocity_predictions[-2] * h1_tensor - self.velocity_predictions[-1] * (h1_tensor + h2_tensor) + model_output * h2_tensor)
                    velocity_third_derivative = None
            elif self.derivative_order == 3:

                if self._step_index == 1 or self._step_index == 2:
                    img_next = sample - 1.5 * model_output * self.dt_list[self._step_index] + 0.5 * self.velocity_predictions[-1] * self.dt_list[self._step_index-1]
                    self._step_index += 1
                    self.velocity_predictions.append(model_output)

                    if not return_dict:
                        return (img_next,)
                    return STORKSchedulerOutput(prev_sample=img_next)
                else:
                    h2 = h1 + self.dt_list[self._step_index-2]
                    h3 = h2 + self.dt_list[self._step_index-3]
                    # Ensure h1, h2, and h3 are tensors for proper broadcasting
                    h1_tensor = torch.tensor(h1, device=model_output.device, dtype=model_output.dtype)
                    h2_tensor = torch.tensor(h2, device=model_output.device, dtype=model_output.dtype)
                    h3_tensor = torch.tensor(h3, device=model_output.device, dtype=model_output.dtype)
                    velocity_derivative = ((h2_tensor * h3_tensor) * (self.velocity_predictions[-1] - model_output) - (h1_tensor * h3_tensor) * (self.velocity_predictions[-2] - model_output) + (h1_tensor * h2_tensor) * (self.velocity_predictions[-3] - model_output)) / (h1_tensor * h2_tensor * h3_tensor)
                    velocity_second_derivative = 2 * ((h2_tensor + h3_tensor) * (self.velocity_predictions[-1] - model_output) - (h1_tensor + h3_tensor) * (self.velocity_predictions[-2] - model_output) + (h1_tensor + h2_tensor) * (self.velocity_predictions[-3] - model_output)) / (h1_tensor * h2_tensor * h3_tensor)
                    velocity_third_derivative = 6 * ((h2_tensor - h3_tensor) * (self.velocity_predictions[-1] - model_output) + (h3_tensor - h1_tensor) * (self.velocity_predictions[-2] - model_output) + (h1_tensor - h2_tensor) * (self.velocity_predictions[-3] - model_output)) / (h1_tensor * h2_tensor * h3_tensor)
            else:
                print("The noise approximation order is not supported!")
                exit()
            
            self.velocity_predictions.append(model_output)
            self._step_index += 1



        Y_j_2 = sample
        Y_j_1 = sample
        Y_j = sample

        
        # Implementation of our Runge-Kutta-Gegenbauer second order method
        for j in range(1, self.s + 1):
            # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
            fraction = (j - 1) * (j + 2) / (self.s * (self.s + 3))
            
            if j == 1:
                mu_tilde = 4 / (self.s * (self.s + 1))
                dt = (t - t_next) * torch.ones(model_output.shape, device=sample.device)
                Y_j = Y_j_1 - dt * mu_tilde * model_output
            else:
                mu = (2 * j + 1) * self.coeff_rock1(j) / (j * self.coeff_rock1(j - 1))
                nu = -(j + 1) * self.coeff_rock1(j) / (j * self.coeff_rock1(j - 2))
                mu_tilde = mu * 4 / (self.s * (self.s + 1))


                # Probability flow ODE update
                diff = -fraction * (t - t_next) * torch.ones(model_output.shape, device=sample.device)
                velocity = self.taylor_approximation(self.derivative_order, diff, model_output, velocity_derivative, velocity_second_derivative, velocity_third_derivative)
                Y_j = mu * Y_j_1 + nu * Y_j_2 - dt * mu_tilde * velocity
                
            Y_j_2 = Y_j_1
            Y_j_1 = Y_j



        img_next = Y_j
        img_next = img_next.to(model_output.dtype)

        return SchedulerOutput(prev_sample=img_next)
    



    def step_flow_matching_2(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        '''
        One step of the STORK2 update for flow matching based models.

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~schedulers.STORKSchedulerOutput`] instead of a plain tuple.
                
        Returns:
            result (Union[Tuple, STORKSchedulerOutput]):
                The next sample in the diffusion chain, either as a tuple or as a [`~schedulers.STORKSchedulerOutput`]. The value is converted back to the original dtype of `model_output` to avoid numerical issues.
        '''
        # Initialize the step index if it's the first step
        if self._step_index is None:
            self._step_index = 0


        # Compute the startup phase or the derivative approximation for the main step
        if self._step_index == 0:
            img_next = sample - model_output * self.dt_list[self._step_index]
            self._step_index += 1
            self.velocity_predictions.append(model_output)

            if not return_dict:
                return (img_next,)
            return STORKSchedulerOutput(prev_sample=img_next)
        else:
            t = self.sigmas[self._step_index]
            t_start = torch.ones(model_output.shape, device=sample.device) * t
            t_next = self.sigmas[self._step_index + 1]
            
            h1 = self.dt_list[self._step_index-1]

            if self.derivative_order == 1:
                # Ensure h1 is a tensor for proper broadcasting
                h1_tensor = torch.tensor(h1, device=model_output.device, dtype=model_output.dtype)
                velocity_derivative = (self.velocity_predictions[-1] - model_output) / h1_tensor
                velocity_second_derivative = None
                velocity_third_derivative = None
            elif self.derivative_order == 2:
                # Ensure h1 and h2 are tensors for proper broadcasting
                h1_tensor = torch.tensor(h1, device=model_output.device, dtype=model_output.dtype)
                if self._step_index == 1:
                    img_next = sample - 1.5 * model_output * self.dt_list[self._step_index] + 0.5 * self.velocity_predictions[-1] * self.dt_list[self._step_index-1]
                    self._step_index += 1
                    self.velocity_predictions.append(model_output)

                    if not return_dict:
                        return (img_next,)
                    return STORKSchedulerOutput(prev_sample=img_next)
                else:
                    h2 = self.dt_list[self._step_index-2]
                    h2_tensor = torch.tensor(h2, device=model_output.device, dtype=model_output.dtype)
                    velocity_derivative = (-self.velocity_predictions[-2] + 4 * self.velocity_predictions[-1] - 3 * model_output) / (2 * h1_tensor)
                    velocity_second_derivative = 2 / (h1_tensor * h2_tensor * (h1_tensor + h2_tensor)) * (self.velocity_predictions[-2] * h1_tensor - self.velocity_predictions[-1] * (h1_tensor + h2_tensor) + model_output * h2_tensor)
                    velocity_third_derivative = None
            elif self.derivative_order == 3:

                if self._step_index == 1 or self._step_index == 2:
                    img_next = sample - 1.5 * model_output * self.dt_list[self._step_index] + 0.5 * self.velocity_predictions[-1] * self.dt_list[self._step_index-1]
                    self._step_index += 1
                    self.velocity_predictions.append(model_output)

                    if not return_dict:
                        return (img_next,)
                    return STORKSchedulerOutput(prev_sample=img_next)
                else:
                    h2 = h1 + self.dt_list[self._step_index-2]
                    h3 = h2 + self.dt_list[self._step_index-3]
                    # Ensure h1, h2, and h3 are tensors for proper broadcasting
                    h1_tensor = torch.tensor(h1, device=model_output.device, dtype=model_output.dtype)
                    h2_tensor = torch.tensor(h2, device=model_output.device, dtype=model_output.dtype)
                    h3_tensor = torch.tensor(h3, device=model_output.device, dtype=model_output.dtype)
                    velocity_derivative = ((h2_tensor * h3_tensor) * (self.velocity_predictions[-1] - model_output) - (h1_tensor * h3_tensor) * (self.velocity_predictions[-2] - model_output) + (h1_tensor * h2_tensor) * (self.velocity_predictions[-3] - model_output)) / (h1_tensor * h2_tensor * h3_tensor)
                    velocity_second_derivative = 2 * ((h2_tensor + h3_tensor) * (self.velocity_predictions[-1] - model_output) - (h1_tensor + h3_tensor) * (self.velocity_predictions[-2] - model_output) + (h1_tensor + h2_tensor) * (self.velocity_predictions[-3] - model_output)) / (h1_tensor * h2_tensor * h3_tensor)
                    velocity_third_derivative = 6 * ((h2_tensor - h3_tensor) * (self.velocity_predictions[-1] - model_output) + (h3_tensor - h1_tensor) * (self.velocity_predictions[-2] - model_output) + (h1_tensor - h2_tensor) * (self.velocity_predictions[-3] - model_output)) / (h1_tensor * h2_tensor * h3_tensor)
            else:
                print("The noise approximation order is not supported!")
                exit()
            
            self.velocity_predictions.append(model_output)
            self._step_index += 1


        Y_j_2 = sample
        Y_j_1 = sample
        Y_j = sample

        
        # Implementation of our Runge-Kutta-Gegenbauer second order method
        for j in range(1, self.s + 1):
            # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
            if j > 1:
                if j == 2:
                    fraction = 4 / (3 * (self.s**2 + self.s - 2))
                else:
                    fraction = ((j - 1)**2 + (j - 1) - 2) / (self.s**2 + self.s - 2)
            
            if j == 1:
                mu_tilde = 6 / ((self.s + 4) * (self.s - 1))
                dt = (t - t_next) * torch.ones(model_output.shape, device=sample.device)
                Y_j = Y_j_1 - dt * mu_tilde * model_output
            else:
                mu = (2 * j + 1) * self.b_coeff(j) / (j * self.b_coeff(j - 1))
                nu = -(j + 1) * self.b_coeff(j) / (j * self.b_coeff(j - 2))
                mu_tilde = mu * 6 / ((self.s + 4) * (self.s - 1))
                gamma_tilde = -mu_tilde * (1 - j * (j + 1) * self.b_coeff(j-1)/ 2)


                # Probability flow ODE update
                diff = -fraction * (t - t_next) * torch.ones(model_output.shape, device=sample.device)
                velocity = self.taylor_approximation(self.derivative_order, diff, model_output, velocity_derivative, velocity_second_derivative, velocity_third_derivative)
                Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * sample - dt * mu_tilde * velocity - dt * gamma_tilde * model_output
                
            Y_j_2 = Y_j_1
            Y_j_1 = Y_j



        img_next = Y_j
        img_next = img_next.to(model_output.dtype)

        if not return_dict:
            return (img_next,) 
        return STORKSchedulerOutput(prev_sample=img_next)


    def step_flow_matching_4(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        '''
        One step of the STORK4 update for flow matching models

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.FloatTensor`: The next sample in the diffusion chain.
        '''        
        # Initialize the step index if it's the first step
        if self._step_index is None:
            self._step_index = 0


        # Compute the startup phase or the derivative approximation for the main step
        if self._step_index == 0:
            img_next = sample - model_output * self.dt_list[self._step_index]
            self._step_index += 1
            self.velocity_predictions.append(model_output)

            if not return_dict:
                return (img_next,)
            return STORKSchedulerOutput(prev_sample=img_next)
        else:
            t = self.sigmas[self._step_index]
            t_start = torch.ones(model_output.shape, device=sample.device) * t
            t_next = self.sigmas[self._step_index + 1]
            
            h1 = self.dt_list[self._step_index-1]

            if self.derivative_order == 1:
                # Ensure h1 is a tensor for proper broadcasting
                h1_tensor = torch.tensor(h1, device=model_output.device, dtype=model_output.dtype)
                velocity_derivative = (self.velocity_predictions[-1] - model_output) / h1_tensor
                velocity_second_derivative = None
                velocity_third_derivative = None
            elif self.derivative_order == 2:
                # Ensure h1 and h2 are tensors for proper broadcasting
                h1_tensor = torch.tensor(h1, device=model_output.device, dtype=model_output.dtype)
                if self._step_index == 1:
                    img_next = sample - 1.5 * model_output * self.dt_list[self._step_index] + 0.5 * self.velocity_predictions[-1] * self.dt_list[self._step_index-1]
                    self._step_index += 1
                    self.velocity_predictions.append(model_output)

                    if not return_dict:
                        return (img_next,)
                    return STORKSchedulerOutput(prev_sample=img_next)
                else:
                    h2 = self.dt_list[self._step_index-2]
                    h2_tensor = torch.tensor(h2, device=model_output.device, dtype=model_output.dtype)
                    velocity_derivative = (-self.velocity_predictions[-2] + 4 * self.velocity_predictions[-1] - 3 * model_output) / (2 * h1_tensor)
                    velocity_second_derivative = 2 / (h1_tensor * h2_tensor * (h1_tensor + h2_tensor)) * (self.velocity_predictions[-2] * h1_tensor - self.velocity_predictions[-1] * (h1_tensor + h2_tensor) + model_output * h2_tensor)
                    velocity_third_derivative = None
            elif self.derivative_order == 3:

                if self._step_index == 1 or self._step_index == 2:
                    img_next = sample - 1.5 * model_output * self.dt_list[self._step_index] + 0.5 * self.velocity_predictions[-1] * self.dt_list[self._step_index-1]
                    self._step_index += 1
                    self.velocity_predictions.append(model_output)

                    if not return_dict:
                        return (img_next,)
                    return STORKSchedulerOutput(prev_sample=img_next)
                else:
                    h2 = h1 + self.dt_list[self._step_index-2]
                    h3 = h2 + self.dt_list[self._step_index-3]
                    # Ensure h1, h2, and h3 are tensors for proper broadcasting
                    h1_tensor = torch.tensor(h1, device=model_output.device, dtype=model_output.dtype)
                    h2_tensor = torch.tensor(h2, device=model_output.device, dtype=model_output.dtype)
                    h3_tensor = torch.tensor(h3, device=model_output.device, dtype=model_output.dtype)
                    velocity_derivative = ((h2_tensor * h3_tensor) * (self.velocity_predictions[-1] - model_output) - (h1_tensor * h3_tensor) * (self.velocity_predictions[-2] - model_output) + (h1_tensor * h2_tensor) * (self.velocity_predictions[-3] - model_output)) / (h1_tensor * h2_tensor * h3_tensor)
                    velocity_second_derivative = 2 * ((h2_tensor + h3_tensor) * (self.velocity_predictions[-1] - model_output) - (h1_tensor + h3_tensor) * (self.velocity_predictions[-2] - model_output) + (h1_tensor + h2_tensor) * (self.velocity_predictions[-3] - model_output)) / (h1_tensor * h2_tensor * h3_tensor)
                    velocity_third_derivative = 6 * ((h2_tensor - h3_tensor) * (self.velocity_predictions[-1] - model_output) + (h3_tensor - h1_tensor) * (self.velocity_predictions[-2] - model_output) + (h1_tensor - h2_tensor) * (self.velocity_predictions[-3] - model_output)) / (h1_tensor * h2_tensor * h3_tensor)
            else:
                print("The noise approximation order is not supported!")
                exit()
            
            self.velocity_predictions.append(model_output)
            self._step_index += 1

        Y_j_2 = sample
        Y_j_1 = sample
        Y_j = sample

        ci1 = t_start
        ci2 = t_start
        ci3 = t_start

        # Coefficients of ROCK4
        ms, fpa, fpb, fpbe, recf = self.coeff_rock4()
        # Choose the degree that's in the precomputed table
        mdeg, mp = self.mdegr(self.s, ms)
        mz = int(mp[0])
        mr = int(mp[1])

        '''
        The first part of the STORK4 update
        '''
        for j in range(1, mdeg + 1):

            # First sub-step in the first part of the STORK4 update
            if j == 1:
                temp1 = -(t - t_next) * recf[mr] * torch.ones(model_output.shape, device=sample.device)
                ci1 = t_start + temp1
                ci2 = ci1
                Y_j_1 = sample + temp1 * model_output
                # Y_j = sample + temp1 * model_output
            # Second and the following sub-steps in the first part of the STORK4 update
            else:
                diff = ci1 - t_start
                velocity = self.taylor_approximation(self.derivative_order, diff, model_output, velocity_derivative, velocity_second_derivative, velocity_third_derivative)

                temp1 = -(t - t_next) * recf[mr + 2 * (j-2) + 1] * torch.ones(model_output.shape, device=sample.device)
                temp3 = -recf[mr + 2 * (j-2) + 2] * torch.ones(model_output.shape, device=sample.device)
                temp2 = torch.ones(model_output.shape, device=sample.device) - temp3

                ci1 = temp1 + temp2 * ci2 + temp3 * ci3
                Y_j = temp1 * velocity + temp2 * Y_j_1 + temp3 * Y_j_2

            # Update the intermediate variables
            Y_j_2 = Y_j_1
            Y_j_1 = Y_j

            ci3 = ci2
            ci2 = ci1

        '''
        The finishing four-step procedure as a composition method
        '''
        # First finishing step
        temp1 = -(t - t_next) * fpa[mz,0] * torch.ones(model_output.shape, device=sample.device)
        diff = ci1 - t_start
        velocity = self.taylor_approximation(self.derivative_order, diff, model_output, velocity_derivative, velocity_second_derivative, velocity_third_derivative)
        Y_j_1 = velocity
        Y_j_3 = Y_j + temp1 * Y_j_1

        # Second finishing step
        ci2 = ci1 + temp1
        temp1 = -(t - t_next) * fpa[mz,1] * torch.ones(model_output.shape, device=sample.device)
        temp2 = -(t - t_next) * fpa[mz,2] * torch.ones(model_output.shape, device=sample.device)
        diff = ci2 - t_start
        velocity = self.taylor_approximation(self.derivative_order, diff, model_output, velocity_derivative, velocity_second_derivative, velocity_third_derivative)
        Y_j_2 = velocity
        Y_j_4 = Y_j + temp1 * Y_j_1 + temp2 * Y_j_2

        # Third finishing step
        ci2 = ci1 + temp1 + temp2
        temp1 = -(t - t_next) * fpa[mz,3] * torch.ones(model_output.shape, device=sample.device)
        temp2 = -(t - t_next) * fpa[mz,4] * torch.ones(model_output.shape, device=sample.device)
        temp3 = -(t - t_next) * fpa[mz,5] * torch.ones(model_output.shape, device=sample.device)
        diff = ci2 - t_start
        velocity = self.taylor_approximation(self.derivative_order, diff, model_output, velocity_derivative, velocity_second_derivative, velocity_third_derivative)
        Y_j_3 = velocity
        # This is the counterpart of the final step in the noise-based diffusion models STORK4
        # fnt = Y_j + temp1 * Y_j_1 + temp2 * Y_j_2 + temp3 * Y_j_3

        # Fourth finishing step
        ci2 = ci1 + temp1 + temp2 + temp3
        temp1 = -(t - t_next) * fpb[mz,0] * torch.ones(model_output.shape, device=sample.device)
        temp2 = -(t - t_next) * fpb[mz,1] * torch.ones(model_output.shape, device=sample.device)
        temp3 = -(t - t_next) * fpb[mz,2] * torch.ones(model_output.shape, device=sample.device)
        temp4 = -(t - t_next) * fpb[mz,3] * torch.ones(model_output.shape, device=sample.device)
        diff = ci2 - t_start
        velocity = self.taylor_approximation(self.derivative_order, diff, model_output, velocity_derivative, velocity_second_derivative, velocity_third_derivative)
        Y_j_4 = velocity
        Y_j = Y_j + temp1 * Y_j_1 + temp2 * Y_j_2 + temp3 * Y_j_3 + temp4 * Y_j_4
        img_next = Y_j

        if not return_dict:
            return (img_next,)
        return STORKSchedulerOutput(prev_sample=img_next)
    

    def step_noise_2(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        '''
        One step of the STORK2 update for noise-based diffusion models.

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~schedulers.STORKSchedulerOutput`] instead of a plain tuple.

        Returns:
            `torch.FloatTensor`: The next sample in the diffusion chain.
        '''
        # Initialize the step index if it's the first step
        if self._step_index is None:
            self._step_index = 0
            self.initial_noise = model_output


        total_step = self.config.num_train_timesteps
        t = self.timesteps[self._step_index] / total_step

        beta_0, beta_1 = self.betas[0], self.betas[-1]
        t_start = torch.ones(model_output.shape, device=sample.device) * t
        beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step
        log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

        # Tweedie's trick
        if self._step_index == len(self.timesteps) - 1:
            noise_last = model_output
            img_next = sample - std * noise_last
            if not return_dict:
                return (img_next,)
            return STORKSchedulerOutput(prev_sample=img_next)
        
        t_next = self.timesteps[self._step_index + 1] / total_step

        # drift, diffusion -> f(x,t), g(t)
        drift_initial, diffusion_initial = -0.5 * beta_t * sample, torch.sqrt(beta_t) * torch.ones(sample.shape, device=sample.device)
        noise_initial = model_output
        score = -noise_initial / std  # score -> noise
        drift_initial = drift_initial - diffusion_initial ** 2 * score * 0.5 # drift -> dx/dt


        dt = torch.ones(model_output.shape, device=sample.device) * self.dt

        if self._step_index == 0:
            # FIRST RUN
            self.initial_sample = sample
            img_next = sample - 0.5 * dt * drift_initial

            self.noise_predictions.append(noise_initial)
            self._step_index += 1

            self.initial_sample = sample
            self.initial_drift = drift_initial
            self.initial_noise = model_output

            return SchedulerOutput(prev_sample=img_next)
        elif self._step_index == 1:
            # SECOND RUN
            t_previous = torch.ones(model_output.shape, device=sample.device) * self.timesteps[0] / 1000
            drift_previous = self.drift_function(self.betas, self.config.num_train_timesteps, t_previous, self.initial_sample, self.noise_predictions[-1])

            img_next = sample - 0.75 * dt * drift_initial + 0.25 * dt * drift_previous

            self.noise_predictions.append(noise_initial)
            self._step_index += 1

            return SchedulerOutput(prev_sample=img_next)
        elif self._step_index == 2:
            h = 0.5 * dt
    
            noise_derivative = (3 * self.noise_predictions[0] - 4 * self.noise_predictions[1] + model_output) / (2 * h)
            noise_second_derivative = (self.noise_predictions[0] - 2 * self.noise_predictions[1] + model_output) / (h ** 2)
            noise_third_derivative = None

            model_output = self.initial_noise
            drift_initial = self.initial_drift
            sample = self.initial_sample

            t = self.timesteps[0] / total_step
            t_start = torch.ones(model_output.shape, device=sample.device) * t
            t_next = self.timesteps[2] / total_step
        elif self._step_index == 3:
            h = 0.5 * dt

            noise_derivative = (-3 * noise_initial + 4 * self.noise_predictions[-1] - self.noise_predictions[-2]) / (2 * h)
            noise_second_derivative = (noise_initial - 2 * self.noise_predictions[-1] + self.noise_predictions[-2]) / (h ** 2)
            noise_third_derivative = None

            self.noise_predictions.append(noise_initial)
        elif self._step_index == 4:
            h = dt

            noise_derivative = (-3 * noise_initial + 4 * self.noise_predictions[-1] - self.noise_predictions[-2]) / (2 * h)
            noise_second_derivative = (noise_initial - 2 * self.noise_predictions[-1] + self.noise_predictions[-2]) / (h ** 2)
            noise_third_derivative = None
            
            self.noise_predictions.append(noise_initial)
        else:
            # ALL ELSE
            h = dt
            
            noise_derivative = (2 * self.noise_predictions[-3] - 9 * self.noise_predictions[-2] + 18 * self.noise_predictions[-1] - 11 * noise_initial) / (6 * h)
            noise_second_derivative = (-self.noise_predictions[-3] + 4 * self.noise_predictions[-2] -5 * self.noise_predictions[-1] + 2 * noise_initial) / (h**2)
            noise_third_derivative = (self.noise_predictions[-3] - 3 * self.noise_predictions[-2] + 3 * self.noise_predictions[-1] - noise_initial) / (h**3)

            self.noise_predictions.append(noise_initial)


        Y_j_2 = sample
        Y_j_1 = sample
        Y_j = sample

        # Implementation of our Runge-Kutta-Gegenbauer second order method
        for j in range(1, self.s + 1):
            # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
            if j > 1:
                if j == 2:
                    fraction = 4 / (3 * (self.s**2 + self.s - 2))
                else:
                    fraction = ((j - 1)**2 + (j - 1) - 2) / (self.s**2 + self.s - 2)
            
            if j == 1:
                mu_tilde = 6 / ((self.s + 4) * (self.s - 1))
                dt = (t - t_next) * torch.ones(model_output.shape, device=sample.device)
                Y_j = Y_j_1 - dt * mu_tilde * model_output
            else:
                mu = (2 * j + 1) * self.b_coeff(j) / (j * self.b_coeff(j - 1))
                nu = -(j + 1) * self.b_coeff(j) / (j * self.b_coeff(j - 2))
                mu_tilde = mu * 6 / ((self.s + 4) * (self.s - 1))
                gamma_tilde = -mu_tilde * (1 - j * (j + 1) * self.b_coeff(j-1)/ 2)


                # Probability flow ODE update
                diff = -fraction * (t - t_next) * torch.ones(model_output.shape, device=sample.device)
                velocity = self.taylor_approximation(self.derivative_order, diff, model_output, noise_derivative, noise_second_derivative, noise_third_derivative)
                Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * sample - dt * mu_tilde * velocity - dt * gamma_tilde * model_output
                
            Y_j_2 = Y_j_1
            Y_j_1 = Y_j



        img_next = Y_j
        img_next = img_next.to(model_output.dtype)
        self._step_index += 1

        if not return_dict:
            return (img_next,)
        return STORKSchedulerOutput(prev_sample=img_next)


    def step_noise_4(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        '''
        One step of the STORK4 update for noise-based diffusion models.

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~schedulers.STORKSchedulerOutput`] instead of a plain tuple.

        Returns:
            `torch.FloatTensor`: The next sample in the diffusion chain.
        '''



        # Initialize the step index if it's the first step
        if self._step_index is None:
            self._step_index = 0
            self.initial_noise = model_output


        total_step = self.config.num_train_timesteps
        t = self.timesteps[self._step_index] / total_step

        beta_0, beta_1 = self.betas[0], self.betas[-1]
        t_start = torch.ones(model_output.shape, device=sample.device) * t
        beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step
        log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

        # Tweedie's trick
        if self._step_index == len(self.timesteps) - 1:
            noise_last = model_output
            img_next = sample - std * noise_last
            if not return_dict:
                return (img_next,)
            return STORKSchedulerOutput(prev_sample=img_next)
        
        t_next = self.timesteps[self._step_index + 1] / total_step

        # drift, diffusion -> f(x,t), g(t)
        drift_initial, diffusion_initial = -0.5 * beta_t * sample, torch.sqrt(beta_t) * torch.ones(sample.shape, device=sample.device)
        noise_initial = model_output
        score = -noise_initial / std  # score -> noise
        drift_initial = drift_initial - diffusion_initial ** 2 * score * 0.5 # drift -> dx/dt


        dt = torch.ones(model_output.shape, device=sample.device) * self.dt


        if self.derivative_order == 2:
            if self._step_index == 0:
                # Initial Euler update
                self.initial_sample = sample
                img_next = sample - dt * drift_initial

                self.noise_predictions.append(noise_initial)
                self._step_index += 1

                self.initial_drift = drift_initial
                
                if not return_dict:
                    return (img_next,)
                return SchedulerOutput(prev_sample=img_next)
            elif self._step_index == 1:
                # Initial 2-step Adams-Bashforth update
                drift_previous = self.initial_drift

                img_next = sample - 1.5 * dt * drift_initial + 0.5 * dt * drift_previous

                self.noise_predictions.append(noise_initial)
                self._step_index += 1

                if not return_dict:
                    return (img_next,)
                return SchedulerOutput(prev_sample=img_next)
            else:
                # STORK4 update
                h = dt

                # The first derivative is calculated using the three point approximation, 
                # and the second derivative is calculated using the standardtwo point approximation.
                noise_derivative = (-self.noise_predictions[-2] + 4 * self.noise_predictions[-1] - 3 * noise_initial) / (2 * h)
                noise_second_derivative = (self.noise_predictions[-2] - 2 * self.noise_predictions[-1] + noise_initial) / h**2
                noise_third_derivative = None

                self.noise_predictions.append(noise_initial)
                noise_approx_order = 2
        elif self.derivative_order == 1:
            if self._step_index == 0:
                # Initial Euler update
                self.initial_sample = sample
                img_next = sample - dt * drift_initial

                self.noise_predictions.append(noise_initial)
                self._step_index += 1

                self.initial_drift = drift_initial
                
                if not return_dict:
                    return (img_next,)
                return SchedulerOutput(prev_sample=img_next)
            else:
                # STORK4 update
                h = dt

                noise_derivative = (self.noise_predictions[-1] - noise_initial) / h
                noise_second_derivative = None
                noise_third_derivative = None

                self.noise_predictions.append(noise_initial)
                noise_approx_order = 1
        else:
            raise ValueError(f"Unknown derivative order: {self.derivative_order}")


        Y_j_2 = sample
        Y_j_1 = sample
        Y_j = sample

        ci1 = t_start
        ci2 = t_start
        ci3 = t_start

        # Coefficients of ROCK4
        ms, fpa, fpb, fpbe, recf = self.coeff_rock4()
        # Choose the degree that's in the precomputed table
        mdeg, mp = self.mdegr(self.s, ms)
        mz = int(mp[0])
        mr = int(mp[1])

        '''
        The first part of the STORK4 update
        '''
        for j in range(1, mdeg + 1):

            # First sub-step in the first part of the STORK4 update
            if j == 1:
                temp1 = -(t - t_next) * recf[mr] * torch.ones(model_output.shape, device=sample.device)
                ci1 = t_start + temp1
                ci2 = ci1
                Y_j_1 = sample + temp1 * model_output #subver
                
                # drift_approx = self.drift_function(self.betas, self.config.num_train_timesteps, t_start, Y_j, model_output)
                # Y_j_1 = sample + temp1 * drift_approx
                
            # Second and the following sub-steps in the first part of the STORK4 update
            else:
                diff = ci1 - t_start
                noise_approx = self.taylor_approximation(noise_approx_order, diff, model_output, noise_derivative, noise_second_derivative, noise_third_derivative)
                drift_approx = self.drift_function(self.betas, self.config.num_train_timesteps, ci1, Y_j_1, noise_approx)

                temp1 = -(t - t_next) * recf[mr + 2 * (j-2) + 1] * torch.ones(model_output.shape, device=sample.device)
                temp3 = -recf[mr + 2 * (j-2) + 2] * torch.ones(model_output.shape, device=sample.device)
                temp2 = torch.ones(model_output.shape, device=sample.device) - temp3

                ci1 = temp1 + temp2 * ci2 + temp3 * ci3
                Y_j = temp1 * drift_approx + temp2 * Y_j_1 + temp3 * Y_j_2

            # Update the intermediate variables
            Y_j_2 = Y_j_1
            Y_j_1 = Y_j

            ci3 = ci2
            ci2 = ci1

        '''
        The finishing four-step procedure as a composition method
        '''
        # First finishing step
        temp1 = -(t - t_next) * fpa[mz,0] * torch.ones(model_output.shape, device=sample.device)
        diff = ci1 - t_start
        noise_approx = self.taylor_approximation(noise_approx_order, diff, model_output, noise_derivative, noise_second_derivative, noise_third_derivative)
        drift_approx = self.drift_function(self.betas, self.config.num_train_timesteps, ci1, Y_j, noise_approx)
        Y_j_1 = drift_approx
        Y_j_3 = Y_j + temp1 * Y_j_1

        # Second finishing step
        ci2 = ci1 + temp1
        temp1 = -(t - t_next) * fpa[mz,1] * torch.ones(model_output.shape, device=sample.device)
        temp2 = -(t - t_next) * fpa[mz,2] * torch.ones(model_output.shape, device=sample.device)
        diff = ci2 - t_start
        noise_approx = self.taylor_approximation(noise_approx_order, diff, model_output, noise_derivative, noise_second_derivative, noise_third_derivative)
        drift_approx = self.drift_function(self.betas, self.config.num_train_timesteps, ci2, Y_j_3, noise_approx)
        Y_j_2 = drift_approx
        Y_j_4 = Y_j + temp1 * Y_j_1 + temp2 * Y_j_2

        # Third finishing step
        ci2 = ci1 + temp1 + temp2
        temp1 = -(t - t_next) * fpa[mz,3] * torch.ones(model_output.shape, device=sample.device)
        temp2 = -(t - t_next) * fpa[mz,4] * torch.ones(model_output.shape, device=sample.device)
        temp3 = -(t - t_next) * fpa[mz,5] * torch.ones(model_output.shape, device=sample.device)
        diff = ci2 - t_start
        noise_approx = self.taylor_approximation(noise_approx_order, diff, model_output, noise_derivative, noise_second_derivative, noise_third_derivative)
        drift_approx = self.drift_function(self.betas, self.config.num_train_timesteps, ci2, Y_j_4, noise_approx)
        Y_j_3 = drift_approx
        fnt = Y_j + temp1 * Y_j_1 + temp2 * Y_j_2 + temp3 * Y_j_3

        # Fourth finishing step
        ci2 = ci1 + temp1 + temp2 + temp3
        temp1 = -(t - t_next) * fpb[mz,0] * torch.ones(model_output.shape, device=sample.device)
        temp2 = -(t - t_next) * fpb[mz,1] * torch.ones(model_output.shape, device=sample.device)
        temp3 = -(t - t_next) * fpb[mz,2] * torch.ones(model_output.shape, device=sample.device)
        temp4 = -(t - t_next) * fpb[mz,3] * torch.ones(model_output.shape, device=sample.device)
        diff = ci2 - t_start
        noise_approx = self.taylor_approximation(noise_approx_order, diff, model_output, noise_derivative, noise_second_derivative, noise_third_derivative)
        drift_approx = self.drift_function(self.betas, self.config.num_train_timesteps, ci2, fnt, noise_approx)
        Y_j_4 = drift_approx
        Y_j = Y_j + temp1 * Y_j_1 + temp2 * Y_j_2 + temp3 * Y_j_3 + temp4 * Y_j_4



        img_next = Y_j
        self._step_index += 1

        if not return_dict:
            return (img_next,)
        return STORKSchedulerOutput(prev_sample=img_next)
        



    def __len__(self):
        return self.config.num_train_timesteps
    
    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Add noise to the original samples according to the noise magnitude at the given timestep.

        Args:
            original_samples (`torch.FloatTensor`):
                The original samples.
            noise (`torch.FloatTensor`):
                The noise to add.
            timesteps (`torch.IntTensor`):
                The timesteps for which to add noise.

        Returns:
            `torch.FloatTensor`:
                The noisy samples.
        """
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def get_velocity(
        self,
        sample: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Get the velocity (score) for the given sample, noise, and timesteps.

        Args:
            sample (`torch.FloatTensor`):
                The sample.
            noise (`torch.FloatTensor`):
                The noise.
            timesteps (`torch.IntTensor`):
                The timesteps.

        Returns:
            `torch.FloatTensor`:
                The velocity.
        """
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity
    
    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        if self.config.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        elif self.config.time_shift_type == "linear":
            return self._time_shift_linear(mu, sigma, t)

    def _time_shift_exponential(self, mu, sigma, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _time_shift_linear(self, mu, sigma, t):
        return mu / (mu + (1 / t - 1) ** sigma)
    
    def taylor_approximation(self, taylor_approx_order, diff, model_output, derivative, second_derivative, third_derivative=None):
        if taylor_approx_order == 1:
            approx_value = model_output + diff * derivative
        elif taylor_approx_order == 2:
            if third_derivative is not None:
                raise ValueError("The third derivative is computed but not used!")
            approx_value = model_output + diff * derivative + 0.5 * diff**2 * second_derivative
        elif taylor_approx_order == 3:
            if third_derivative is None:
                raise ValueError("The third derivative is not computed!")
            approx_value = model_output + diff * derivative + 0.5 * diff**2 * second_derivative \
                + diff**3 * third_derivative / 6
        else:
            print("The noise approximation order is not supported!")
            exit()

        return approx_value
    

    def drift_function(self, betas, total_step, t_eval, y_eval, noise):
        '''
        Drift function for the probability flow ODE in the noise-based diffusion model.

        Args:
            betas (`torch.FloatTensor`):
                The betas of the diffusion model.
            total_step (`int`):
                The total number of steps in the diffusion chain.
            t_eval (`torch.FloatTensor`):
                The timestep to be evaluated at in the diffusion chain.
            y_eval (`torch.FloatTensor`):
                The sample to be evaluated at in the diffusion chain.
            noise (`torch.FloatTensor`):
                The noise used at the current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                The drift term for the probability flow ODE in the diffusion model.
        '''
        beta_0, beta_1 = betas[0], betas[-1]
        beta_t = (beta_0 + t_eval * (beta_1 - beta_0)) * total_step
        beta_t = beta_t * torch.ones(y_eval.shape, device=y_eval.device)

        log_mean_coeff = (-0.25 * t_eval ** 2 * (beta_1 - beta_0) - 0.5 * t_eval * beta_0) * total_step
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

        # drift, diffusion -> f(x,t), g(t)
        drift, diffusion = -0.5 * beta_t * y_eval, torch.sqrt(beta_t) * torch.ones(y_eval.shape, device=y_eval.device)
        score = -noise / std  # score -> noise
        drift = drift - diffusion ** 2 * score * 0.5 # drift -> dx/dt

        return drift

    def b_coeff(self, j):
        '''
        Coefficients of STORK2. The are based on the second order Runge-Kutta-Gegenbauer method.
        Details of the coefficients can be found in https://www.sciencedirect.com/science/article/pii/S0021999120306537

        Args:
            j (`int`):
                The sub-step index of the coefficient.

        Returns:
            `float`:
                The coefficient of the STORK2.
        '''
        if j < 0:
            print("The b_j coefficient in the RKG method can't have j negative")
            return
        if j == 0:
            return 1
        if j == 1:
            return 1 / 3
        
        return 4 * (j - 1) * (j + 4) / (3 * j * (j + 1) * (j + 2) * (j + 3))

    def coeff_rock1(self, j):
        if j < 0:
            print("The b_j coefficient in the RKG method can't have j negative")
        return 2 / ((j + 1) * (j + 2))

    def coeff_rock4(self):
        '''
        Load pre-computed coefficients of STORK4. The are based on the fourth order orthogonal Runge-Kutta-Chebyshev (ROCK4) method.
        Details of the coefficients can be found in https://epubs.siam.org/doi/abs/10.1137/S1064827500379549.
        The pre-computed coefficients are based on the implementation https://www.mathworks.com/matlabcentral/fileexchange/12129-rock4.

        Args:
            j (`int`):
                The sub-step index of the coefficient.

        Returns:
            ms (`torch.FloatTensor`):
                The degrees that coefficients were pre-computed for STORK4.
            fpa, fpb, fpbe, recf (`torch.FloatTensor`):
                The parameters for the finishing procedure.
        '''
        # Degrees
        data = loadmat(f'{CONSTANTSFOLDER}/ms.mat')
        ms = data['ms'][0]

        # Parameters for the finishing procedure
        data = loadmat(f'{CONSTANTSFOLDER}/fpa.mat')
        fpa = data['fpa']

        data = loadmat(f'{CONSTANTSFOLDER}/fpb.mat')
        fpb = data['fpb']

        data = loadmat(f'{CONSTANTSFOLDER}/fpbe.mat')
        fpbe = data['fpbe']

        # Parameters for the recurrence procedure
        data = loadmat(f'{CONSTANTSFOLDER}/recf.mat')
        recf = data['recf'][0]

        return ms, fpa, fpb, fpbe, recf



    def mdegr(self, mdeg1, ms):
        '''
        Find the optimal degree in the pre-computed degree coefficients table for the STORK4 method.

        Args:
            mdeg1 (`int`):
                The degree to be evaluated.
            ms (`torch.FloatTensor`):
                The degrees that coefficients were pre-computed for STORK4.

        Returns:
            mdeg (`int`):
                The optimal degree in the pre-computed degree coefficients table for the STORK4 method.
            mp (`torch.FloatTensor`):
                The pointer which select the degree in ms[i], such that mdeg<=ms[i].
                mp[0] (`int`): The pointer which select the degree in ms[i], such that mdeg<=ms[i].
                mp[1] (`int`): The pointer which gives the corresponding position of a_1 in the data recf for the selected degree.
        '''           
        mp = torch.zeros(2)
        mp[1] = 1
        mdeg = mdeg1
        for i in range(len(ms)):
            if (ms[i]/mdeg) >= 1:
                mdeg = ms[i]
                mp[0] = i
                mp[1] = mp[1] - 1
                break
            else:   
                mp[1] = mp[1] + ms[i] * 2 - 1

        return mdeg, mp