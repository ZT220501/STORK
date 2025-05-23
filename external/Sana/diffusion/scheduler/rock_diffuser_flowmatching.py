# Copyright 2024 Zhejiang University Team and The HuggingFace Team. All rights reserved.
#
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

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

import math
import os
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from scipy.io import loadmat
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput
from diffusers.utils import is_scipy_available
from pathlib import Path

current_file = Path(__file__)
PARENTFOLDER = current_file.parent
#######################
# ROCK4-based methods #
#######################
'''
Coefficients of ROCK4
'''
def coeff_rock4():

    # Degrees
    data = loadmat(f'{PARENTFOLDER}/rock_constants/ms.mat')
    ms = data['ms'][0]

    # Parameters for the finishing procedure
    data = loadmat(f'{PARENTFOLDER}/rock_constants/fpa.mat')
    fpa = data['fpa']

    data = loadmat(f'{PARENTFOLDER}/rock_constants/fpb.mat')
    fpb = data['fpb']

    data = loadmat(f'{PARENTFOLDER}/rock_constants/fpbe.mat')
    fpbe = data['fpbe']

    # Parameters for the recurrence procedure
    data = loadmat(f'{PARENTFOLDER}/rock_constants/recf.mat')
    recf = data['recf'][0]


    return ms, fpa, fpb, fpbe, recf



'''
Find the optimal degree in the pre-computed degree coefficients table.
'''
def mdegr(mdeg1,ms):
    '''
    Find the optimal degree in the pre-computed degree coefficients table.
    MP(1): pointer which select the degree in ms(i),
        such that mdeg<=ms(i).
    MP(2): pointer which gives the corresponding position
        of a_1 in the data recf for the selected degree.
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




class FlowMatchROCKScheduler(SchedulerMixin, ConfigMixin):
    """
    `PNDMScheduler` uses pseudo numerical methods for diffusion models such as the Runge-Kutta and linear multi-step
    method.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        skip_prk_steps (`bool`, defaults to `False`):
            Allows the scheduler to skip the Runge-Kutta steps defined in the original paper as being required before
            PLMS steps.
        set_alpha_to_one (`bool`, defaults to `False`):
            Each diffusion step uses the alphas product value at that step and at the previous one. For the final step
            there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the alpha value at step 0.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process)
            or `v_prediction` (see section 2.4 of [Imagen Video](https://imagen.research.google/video/paper.pdf)
            paper).
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
    """
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        time_shift_type: str = "exponential",
        s: int = 50,
        precision = "float32",
    ):
        if self.config.use_beta_sigmas and not is_scipy_available():
           raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        if sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
            raise ValueError(
                "Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used."
            )
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError("`time_shift_type` must either be 'exponential' or 'linear'.")
 
        assert precision in ["float32", "float16"], f"precision {precision} not supported"


        if precision == "float32":
            self.np_dtype = np.float32
            self.dtype = torch.float32
        elif precision == "float16":   
            self.np_dtype = np.float16 
            self.dtype = torch.float16
            

        # timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        # timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

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
        self.velocity_predictions = []
        self.s = s

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

        # 1. Prepare default sigmas
        is_timesteps_provided = timesteps is not None

        if is_timesteps_provided:
            #timesteps = np.array(timesteps).astype(np.float32)
            timesteps = np.array(timesteps).astype(self.np_dtype)
        
        if sigmas is None:
            if timesteps is None:
                timesteps = np.linspace(
                    self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
                )
            sigmas = timesteps / self.config.num_train_timesteps
        else:
            #sigmas = np.array(sigmas).astype(np.float32)
            sigmas = np.array(sigmas).astype(self.np_dtype)
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
        #sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        sigmas = torch.from_numpy(sigmas).to(dtype=self.dtype, device=device)
        if not is_timesteps_provided:
            timesteps = sigmas * self.config.num_train_timesteps
        else:
            #timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)
            timesteps = torch.from_numpy(timesteps).to(dtype=self.dtype, device=device)

        # 6. Append the terminal sigma value.
        #    If a model requires inverted sigma schedule for denoising but timesteps without inversion, the
        #    `invert_sigmas` flag can be set to `True`. This case is only required in Mochi
        if self.config.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        else:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])


        self._step_index = None
        self._begin_index = None


        # Modify the timesteps to fit in the ROCK4 scheme
        self.timesteps = timesteps.tolist()
        self.timesteps = np.insert(self.timesteps, 1, self.timesteps[1])
        self.timesteps = np.insert(self.timesteps, 1, self.timesteps[0] + (self.timesteps[1] - self.timesteps[0]) / 2)
        self.timesteps = torch.tensor(self.timesteps)
        #self.timesteps = self.timesteps.to(dtype=torch.float32, device=device)
        self.timesteps = self.timesteps.to(dtype=self.dtype, device=device)

        # Modify the timesteps in order to become sigmas
        self.sigmas = self.timesteps.tolist()
        self.sigmas.append(0)
        self.sigmas = torch.tensor(self.sigmas)
        #self.sigmas = self.sigmas.to(dtype=torch.float32, device=device)
        self.sigmas = self.sigmas.to(dtype=self.dtype, device=device)
        self.sigmas = self.sigmas / self.config.num_train_timesteps

        self.dt_list = self.sigmas[:-1] - self.sigmas[1:]
        # Eliminate the zero values from the dt_list
        self.dt_list = self.dt_list[self.dt_list.nonzero()]
        self.dt_list = self.dt_list.reshape(-1)
        self.dt_list[0] = self.dt_list[0] * 2
        self.dt_list[1] = self.dt_list[1] * 2
        self.dt_list = self.dt_list.tolist()
        self.dt_list = np.insert(self.dt_list, 1, self.dt_list[0])
        self.dt_list = torch.tensor(self.dt_list).to(self.dtype)



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

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_shift(self, shift: float):
        self._shift = shift

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
            #schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            #timestep = timestep.to(sample.device, dtype=torch.float32)
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

    # This function is directly copied from the FlowMatchEulerDiscreteScheduler

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor = None,
        return_dict: bool = False
    ) -> torch.Tensor:
        # print("====================="*5)
        # print(f"{self._step_index}: model_output")
        # print(model_output)
        # print("====================="*5)
        # print("self.timesteps")
        # print(self.timesteps)
        # print("====================="*5)
        # print("self.sigmas")
        # print(self.sigmas)
        # exit()
        if self._step_index is None:
            self._step_index = 0
        n = sample.shape[0]
        total_step = self.config.num_train_timesteps

        # Upcast to avoid precision issues when computing prev_sample
        #sample = sample.to(torch.float32)
        sample = sample.to(self.dtype)
        sample = sample.to('cuda')



        velocity_approx_order = 0
        if self._step_index == 0:
            # FIRST RUN
            dt = self.dt_list[self._step_index]
            dt = torch.ones(n, device=sample.device) * dt #self.dt
            dt = dt.view(-1, 1, 1, 1)

            # Storing context
            self.initial_sample = sample
            self.first_velocity = model_output
            img_next = sample - 0.5 * dt * model_output
            self.velocity_predictions.append(model_output)
            self._step_index += 1
            img_next = img_next.to(model_output.dtype)
            return SchedulerOutput(prev_sample=img_next)
        elif self._step_index == 1:
            # SECOND RUN
            t = self.sigmas[self._step_index]
            img_next = sample - 0.75 * self.dt_list[self._step_index] * model_output + 0.25 * self.dt_list[self._step_index] * self.first_velocity
            self.velocity_predictions.append(model_output)
            self._step_index += 1
            img_next = img_next.to(model_output.dtype)
            return SchedulerOutput(prev_sample=img_next)
        elif self._step_index == 2:
            # THRID RUN 
            h = self.dt_list[self._step_index] / 2
            velocity_derivative = (3 * self.velocity_predictions[0] - 4 * self.velocity_predictions[1] + model_output) / (2 * h)
            velocity_second_derivative = (self.velocity_predictions[0] - 2 * self.velocity_predictions[1] + model_output) / (h ** 2)
            velocity_third_derivative = None
            velocity_approx_order = 2
            self._step_index += 1
            # Restore context
            t = self.sigmas[0]
            t_start = torch.ones(n, device=sample.device) * t
            t_next = self.sigmas[2]
            sample = self.initial_sample
        elif self._step_index == 3:
            # FOURTH RUN
            t = self.sigmas[self._step_index]
            t_start = torch.ones(n, device=sample.device) * t
            t_next = self.sigmas[self._step_index + 1]
            # dt = self.dt_list[self._step_index]
            # dt = torch.ones(n, device=sample.device) * dt #self.dt
            # dt = dt.view(-1, 1, 1, 1)


            h = self.dt_list[0] / 2
            velocity_derivative = (-3 * model_output + 4 * self.velocity_predictions[-1] - self.velocity_predictions[-2]) / (2 * h)
            velocity_second_derivative = (model_output - 2 * self.velocity_predictions[-1] + self.velocity_predictions[-2]) / (h ** 2)
            

            #print("====================="*5)
            #print(f"{self._step_index}: velocity_derivative")
            #print(velocity_derivative)
            #
            #print("====================="*5)
            #print(f"{self._step_index}:velocity_second_derivative")
            #print(velocity_second_derivative)            
#
            #print("====================="*5)
            #print(f"{self._step_index}:h")
            #print(h)
            

            velocity_third_derivative = None
            self.velocity_predictions.append(model_output)
            velocity_approx_order = 2
            self._step_index += 1


        elif self._step_index == 4:
            t = self.sigmas[self._step_index]
            t_start = torch.ones(n, device=sample.device) * t
            t_next = self.sigmas[self._step_index + 1]


            # dt = t - t_next
            # dt = torch.ones(n, device=sample.device) * dt #self.dt
            # dt = dt.view(-1, 1, 1, 1)

            h1 = self.dt_list[self._step_index-1]
            h2 = self.dt_list[self._step_index-2]

            velocity_derivative = (-self.velocity_predictions[-2] + 4 * self.velocity_predictions[-1] - 3 * model_output) / (2 * h1)
            velocity_second_derivative = 2 / (h1 * h2 * (h1 + h2)) * (self.velocity_predictions[-2] * h1 - self.velocity_predictions[-1] * (h1 + h2) + model_output * h2)
            velocity_third_derivative = None
            
            self.velocity_predictions.append(model_output)
            velocity_approx_order = 2
            self._step_index += 1
        else:
            t = self.sigmas[self._step_index]
            t_start = torch.ones(n, device=sample.device) * t
            t_next = self.sigmas[self._step_index + 1]

            # dt = t - t_next
            # dt = torch.ones(n, device=sample.device) * dt #self.dt
            # dt = dt.view(-1, 1, 1, 1)


            h1 = self.dt_list[self._step_index-1]
            h2 = h1 + self.dt_list[self._step_index-2]
            h3 = h2 + self.dt_list[self._step_index-3]

            velocity_derivative = ((h2 * h3) * (self.velocity_predictions[-1] - model_output) - (h1 * h3) * (self.velocity_predictions[-2] - model_output) + (h1 * h2) * (self.velocity_predictions[-3] - model_output)) / (h1 * h2 * h3)
            velocity_second_derivative = 2 * ((h2 + h3) * (self.velocity_predictions[-1] - model_output) - (h1 + h3) * (self.velocity_predictions[-2] - model_output) + (h1 + h2) * (self.velocity_predictions[-3] - model_output)) / (h1 * h2 * h3)
            velocity_third_derivative = 6 * ((h2 - h3) * (self.velocity_predictions[-1] - model_output) + (h3 - h1) * (self.velocity_predictions[-2] - model_output) + (h1 - h2) * (self.velocity_predictions[-3] - model_output)) / (h1 * h2 * h3)


            self.velocity_predictions.append(model_output)

            velocity_approx_order = 3
            self._step_index += 1


        # print("This is the timestep " + str(self._step_index))
        # print("t is " + str(self.sigmas[self._step_index]))
        # print("t_next is " + str(self.sigmas[self._step_index + 1]))



        Y_j_2 = sample
        Y_j_1 = sample
        Y_j = sample

        ci1 = t_start
        ci2 = t_start
        ci3 = t_start

        # Coefficients of ROCK4
        ms, fpa, fpb, fpbe, recf = coeff_rock4()
        # Choose the degree that's in the precomputed table
        mdeg, mp = mdegr(self.s, ms)
        mz = int(mp[0])
        mr = int(mp[1])

        '''
        ROCK4 Update
        '''
        # The first stage
        for j in range(1, mdeg + 1):
            # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
            if j == 1:
                temp1 = -(t - t_next) * recf[mr] * torch.ones(n, device=sample.device)
                ci1 = t_start + temp1
                ci2 = ci1
                Y_j_2 = sample
                Y_j_1 = sample + temp1.view(-1, 1, 1, 1) * model_output
            else:
                '''
                Approximate the noise using Taylor expansion
                '''
                diff = ci1 - t_start
                diff = diff.view(-1, 1, 1, 1)
                velocity = self.velocity_approximation(velocity_approx_order, diff, model_output, velocity_derivative, velocity_second_derivative, velocity_third_derivative)


                temp1 = -(t - t_next) * recf[mr + 2 * (j-2) + 1] * torch.ones(n, device=sample.device)
                temp3 = -recf[mr + 2 * (j-2) + 2] * torch.ones(n, device=sample.device)
                temp2 = torch.ones(n, device=sample.device) - temp3

                ci1 = temp1 + temp2 * ci2 + temp3 * ci3
                # print("Shape of ci1 is " + str(ci1.shape))
                Y_j = temp1.view(-1, 1, 1, 1) * velocity + temp2.view(-1, 1, 1, 1) * Y_j_1 + temp3.view(-1, 1, 1, 1) * Y_j_2

            # Update the intermediate variables
            Y_j_2 = Y_j_1
            Y_j_1 = Y_j

            ci3 = ci2
            ci2 = ci1


        # The finishing four-step procedure
        temp1 = -(t - t_next) * fpa[mz,0] * torch.ones(n, device=sample.device)
        diff = ci1 - t_start
        diff = diff.view(-1, 1, 1, 1)
        velocity = self.velocity_approximation(velocity_approx_order, diff, model_output, velocity_derivative, velocity_second_derivative, velocity_third_derivative)
        Y_j_1 = velocity
        Y_j_3 = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1

        ci2 = ci1 + temp1
        temp1 = -(t - t_next) * fpa[mz,1] * torch.ones(n, device=sample.device)
        temp2 = -(t - t_next) * fpa[mz,2] * torch.ones(n, device=sample.device)
        diff = ci2 - t_start
        diff = diff.view(-1, 1, 1, 1)
        velocity = self.velocity_approximation(velocity_approx_order, diff, model_output, velocity_derivative, velocity_second_derivative, velocity_third_derivative)
        Y_j_2 = velocity
        Y_j_4 = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2

        ci2 = ci1 + temp1 + temp2
        temp1 = -(t - t_next) * fpa[mz,3] * torch.ones(n, device=sample.device)
        temp2 = -(t - t_next) * fpa[mz,4] * torch.ones(n, device=sample.device)
        temp3 = -(t - t_next) * fpa[mz,5] * torch.ones(n, device=sample.device)
        diff = ci2 - t_start
        diff = diff.view(-1, 1, 1, 1)
        velocity = self.velocity_approximation(velocity_approx_order, diff, model_output, velocity_derivative, velocity_second_derivative, velocity_third_derivative)
        Y_j_3 = velocity
        fnt = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2 + temp3.view(-1, 1, 1, 1) * Y_j_3

        ci2 = ci1 + temp1 + temp2 + temp3
        temp1 = -(t - t_next) * fpb[mz,0] * torch.ones(n, device=sample.device)
        temp2 = -(t - t_next) * fpb[mz,1] * torch.ones(n, device=sample.device)
        temp3 = -(t - t_next) * fpb[mz,2] * torch.ones(n, device=sample.device)
        temp4 = -(t - t_next) * fpb[mz,3] * torch.ones(n, device=sample.device)
        diff = ci2 - t_start
        diff = diff.view(-1, 1, 1, 1)
        velocity = self.velocity_approximation(velocity_approx_order, diff, model_output, velocity_derivative, velocity_second_derivative, velocity_third_derivative)
        Y_j_4 = velocity
        Y_j = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2 + temp3.view(-1, 1, 1, 1) * Y_j_3 + temp4.view(-1, 1, 1, 1) * Y_j_4
        img_next = Y_j
        
        img_next = img_next.to(model_output.dtype)

        #print("======================"*5)
        #print(f"{self._step_index}: img_next has NaN")
        #print(img_next.isnan().any())
        #if self._step_index == 4:
        #    exit(0)
        return SchedulerOutput(prev_sample=img_next)
     
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
    
    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        if self.config.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        elif self.config.time_shift_type == "linear":
            return self._time_shift_linear(mu, sigma, t)

    def _time_shift_exponential(self, mu, sigma, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _time_shift_linear(self, mu, sigma, t):
        return mu / (mu + (1 / t - 1) ** sigma)
    
    def velocity_approximation(self, velocity_approx_order, diff, model_output, velocity_derivative, velocity_second_derivative, velocity_third_derivative=None):
        if velocity_approx_order == 2:
            if velocity_third_derivative is not None:
                raise ValueError("The third derivative is computed but not used!")
            velocity = model_output + diff * velocity_derivative + 0.5 * diff**2 * velocity_second_derivative
        elif velocity_approx_order == 3:
            if velocity_third_derivative is None:
                raise ValueError("The third derivative is not computed!")
            velocity = model_output + diff * velocity_derivative + 0.5 * diff**2 * velocity_second_derivative \
                + diff**3 * velocity_third_derivative / 6
        else:
            print("The noise approximation order is not supported!")
            exit()

        return velocity


from diffusers import DiffusionPipeline
import torch

class FlowSANAPipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)
        self.precision = next(model.parameters()).dtype

    @torch.no_grad()
    def __call__(self, batch_size=4, num_inference_steps=50, channel=3,  height=32, width=32, guidance_scale = 1,
                 uncondition=None, condition=None, model_kwargs=None, guidance_type="classifier-free"):
        device = next(self.model.parameters()).device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if guidance_type == "classifier-free":
            x = torch.randn((batch_size, channel, height, width), dtype=self.precision).to(device)
            c_in = torch.cat([uncondition, condition])
            for t in self.scheduler.timesteps:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t.unsqueeze(0)] * 2 * batch_size)
                try:
                    model_output_uncond, model_output = self.model(x_in, t_in, c_in, **model_kwargs).chunk(2)
                except:
                    model_output_uncond, model_output = self.model(x_in, t_in, c_in, **model_kwargs)[0].chunk(2)
                model_output = model_output_uncond + guidance_scale * (model_output - model_output_uncond)
                x = self.scheduler.step(model_output, t.item(), x).prev_sample
            return {"images": x}
        else: 
            image = torch.randn((batch_size, channel, height, width), dtype=self.precision).to(device)
            for t in self.scheduler.timesteps:
                t = t.unsqueeze(0)
                # Get the output from SANA model
                model_output = self.model(image, t, condition, **model_kwargs)
                image = self.scheduler.step(model_output, t.item(), image).prev_sample
            # Don't renormalize as we are working in the latent space
            #image = (image.clamp(-1, 1) + 1) / 2  # Normalize to [0,1]
            return {"images": image}
