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








'''
Drift function for the backward ODE in the diffusion model
'''
def drift_function(betas, total_step, t_eval, y_eval, noise):
    beta_0, beta_1 = betas[0], betas[-1]
    beta_t = (beta_0 + t_eval * (beta_1 - beta_0)) * total_step
    # beta_t = beta_0 + t_eval * (beta_1 - beta_0)

    log_mean_coeff = (-0.25 * t_eval ** 2 * (beta_1 - beta_0) - 0.5 * t_eval * beta_0) * total_step
    # log_mean_coeff = (-0.25 * t_eval ** 2 * (beta_1 - beta_0) - 0.5 * t_eval * beta_0)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

    # drift, diffusion -> f(x,t), g(t)
    drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * y_eval, torch.sqrt(beta_t) 
    score = -noise / std.view(-1, 1, 1, 1)  # score -> noise
    drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt

    return drift









from pathlib import Path
current_file = Path(__file__)
PARENTFOLDER = current_file.parent.parent
#######################
# ROCK4-based methods #
#######################
'''
Coefficients of ROCK4
'''
def coeff_rock4():

    # Degrees
    data = loadmat(f'{PARENTFOLDER}/ms.mat')
    ms = data['ms'][0]

    # Parameters for the finishing procedure
    data = loadmat(f'{PARENTFOLDER}/fpa.mat')
    fpa = data['fpa']

    data = loadmat(f'{PARENTFOLDER}/fpb.mat')
    fpb = data['fpb']

    data = loadmat(f'{PARENTFOLDER}/fpbe.mat')
    fpbe = data['fpbe']

    # Parameters for the recurrence procedure
    data = loadmat(f'{PARENTFOLDER}/recf.mat')
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






class ROCKScheduler(SchedulerMixin, ConfigMixin):
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
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        set_alpha_to_one: bool = False,
        shift: float = 0.63,
        sigma: float = 1.0,
        stopping_eps: float = 1e-3,
        s: int = 50,
    ):
        super().__init__()
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        #elif beta_schedule == "squaredcos_cap_v2":
        #   # Glide cosine schedule
        #    self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")
        
        self.betas = self.betas.to('cuda')

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        self.stopping_eps = stopping_eps
        self.shift = shift
        self.sigma = sigma
        self.s = s

        
       
        
    
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

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """



         # setable values
        self.num_inference_steps = None
        self._timesteps = None
        self.timesteps = None
        self.noise_predictions = []
        self.dt_list=[]
        self.velocity_predictions = []
        # Currently we only support constant step size
        self.initial_sample = None
        # Currently the prediction type only supports "epsilon", "v_prediction", or "flow_prediction"
        # Default is "epsilon", which corresponds to the noise prediction of the diffusion model
        # "v_prediction" is the velocity prediction of the diffusion model
        # "flow_prediction" is the flow prediction of the diffusion model
        
        self.first_dt = None
        


        self.num_inference_steps = num_inference_steps

        seq = np.linspace(0, 1, self.num_inference_steps+1)
        #seq = np.linspace(0, 1, num=self.num_inference_steps+1, endpoint=True)
        seq[0] = self.stopping_eps
        seq = seq[:-1]
        seq = seq[::-1]

        # Do exponential time shifting if we're using flow matching
        # if self.prediction_type == "flow_prediction":
        # seq = self._time_shift_exponential(self.shift, self.sigma, seq)

        # Add the intermediate step between the first step and the second step
        seq = np.insert(seq, 1, seq[1])
        seq = np.insert(seq, 1, seq[0] + (seq[1] - seq[0]) / 2)

        # The following lines are for the uniform timestepping case

        # FIXME I just modified this
        self.dt = -(seq[1] - seq[0]) * 2
        seq = seq * 1000
        self._timesteps = seq
 

        # if self.prediction_type == "flow_prediction":
        #     self._timesteps = seq
        # elif self.prediction_type == "epsilon":
        #     self._timesteps = seq * 1000
        # else:
        #     print("The prediction type is not supported!")
        #     exit()

        self.timesteps = torch.from_numpy(seq.copy()).to(device).type(torch.float32)
        print("self.timesteps is " + str(self.timesteps))

        self._step_index = None
        self._begin_index = None
        
                
    def old(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor = None,
    ) -> torch.Tensor:
        if self._step_index is None:
            self._step_index = 0
        # Get the noise schedule
        n = sample.shape[0]
        total_step = self.config.num_train_timesteps
        # FIXME I just modified this
        t = self.timesteps[self._step_index] / 999 #1000
        beta_0, beta_1 = self.betas[0], self.betas[-1]
        t_start = torch.ones(n, device=sample.device) * t
        beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step
        log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        if self._step_index == len(self.timesteps) - 1:
            # Tweedie's trick
            noise_last = model_output
            img_next = sample - std.view(-1, 1, 1, 1) * noise_last
            return SchedulerOutput(prev_sample=img_next)
        # FIXME I just modified this
        t_next = (self.timesteps[self._step_index + 1])/ 999 #1000
        # drift, diffusion -> f(x,t), g(t)
        drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * sample, torch.sqrt(beta_t)
        noise_initial = model_output
        score = -noise_initial / std.view(-1, 1, 1, 1)  # score -> noise
        drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt


        # FIXME in the original code, dt = t-t_next; However, here is just self.dt. Why?
        dt = t - t_next
        dt = torch.ones(n, device=sample.device) * dt #self.dt
        dt = dt.view(-1, 1, 1, 1)
        
        noise_approx_order = 0
        if self._step_index == 0:
            self.initial_sample = sample
            self.first_dt = dt
            self.first_t_start = t_start
            self.first_t_end = torch.ones(n, device=sample.device) * t_next
            self.first_drfit = drift_initial
            # FIRST RUN
            img_next = sample - 0.5 * dt * drift_initial
            self.noise_predictions.append(model_output)
            self._step_index += 1
            return SchedulerOutput(prev_sample=img_next)
        elif self._step_index == 1:
            # SECOND RUN


            t_intermediate = self.first_t_start - 0.5 * (self.first_t_start - self.first_t_end)
            beta_t = (beta_0 + t_intermediate * (beta_1 - beta_0)) * total_step
            log_mean_coeff = (-0.25 * t_intermediate ** 2 * (beta_1 - beta_0) - 0.5 * t_intermediate * beta_0) * total_step
            std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
            drift_next, diffusion_next = -0.5 * beta_t.view(-1, 1, 1, 1) * self.initial_sample, torch.sqrt(beta_t)
            score = -model_output/ std.view(-1, 1, 1, 1)
            drift_next = drift_next - diffusion_next.view(-1, 1, 1, 1) ** 2 * score * 0.5




            #drift_previous = self.first_drfit
            img_next = sample - 0.75 * self.first_dt * drift_next + 0.25 * self.first_dt * self.first_drfit
            self.noise_predictions.append(model_output)
            self._step_index += 1
            return SchedulerOutput(prev_sample=img_next)
        elif self._step_index == 2:
            # THRID RUN 
            h = 0.5 * self.first_dt
            noise_derivative = (3 * self.noise_predictions[0] - 4 * self.noise_predictions[1] + model_output) / (2 * h)
            noise_second_derivative = (self.noise_predictions[0] - 2 * self.noise_predictions[1] + model_output) / (h ** 2)
            self.dt_list.append(self.first_drfit)
            noise_approx_order = 2
            self._step_index += 1
        elif self._step_index == 3:
            # FOURTH RUN
            h = 0.5 * dt
            noise_derivative = (-3 * noise_initial + 4 * self.noise_predictions[-1] - self.noise_predictions[-2]) / (2 * h)
            noise_second_derivative = (noise_initial - 2 * self.noise_predictions[-1] + self.noise_predictions[-2]) / (h ** 2)
            self.noise_predictions.append(noise_initial)
            self.dt_list.append(dt)
            noise_approx_order = 2
            self._step_index += 1
        elif self._step_index == 4:
            # FIFTH RUN
            h1 = self.dt_list[-1]
            h2 = self.dt_list[-2]

            noise_derivative = (-self.noise_predictions[-2] + 4 * self.noise_predictions[-1] - 3 * noise_initial) / (2 * h1)
            noise_second_derivative = 2 / (h1 * h2 * (h1 + h2)) * (self.noise_predictions[-2] * h1 - self.noise_predictions[-1] * (h1 + h2) + noise_initial * h2)
            
            self.noise_predictions.append(noise_initial)
            self.dt_list.append(dt)
            noise_approx_order = 2
            self._step_index += 1
        else:
            # ALL ELSE
            h = self.dt_list[-1]
            noise_derivative = (2 * self.noise_predictions[-3] - 9 * self.noise_predictions[-2] + 18 * self.noise_predictions[-1] - 11 * noise_initial) / (6 * h)
            noise_second_derivative = (-self.noise_predictions[-3] + 4 * self.noise_predictions[-2] -5 * self.noise_predictions[-1] + 2 * noise_initial) / (h**2)
            noise_third_derivative = (self.noise_predictions[-3] - 3 * self.noise_predictions[-2] + 3 * self.noise_predictions[-1] - noise_initial) / (h**3)

            self.noise_predictions.append(noise_initial)
            self.dt_list.append(dt)

            noise_approx_order = 3
            self._step_index += 1




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
                Y_j_1 = sample + temp1.view(-1, 1, 1, 1) * drift_initial
            else:
                beta_0, beta_1 = self.betas[0], self.betas[-1]
                beta_t = (beta_0 + ci1 * (beta_1 - beta_0)) * total_step

                log_mean_coeff = (-0.25 * ci1 ** 2 * (beta_1 - beta_0) - 0.5 * ci1 * beta_0) * total_step
                std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

                # drift, diffusion -> f(x,t), g(t)
                drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * Y_j_1, torch.sqrt(beta_t)
                '''
                Approximate the noise using Taylor expansion
                '''
                diff = ci1 - t_start
                diff = diff.view(-1, 1, 1, 1)
                if noise_approx_order == 2:
                    noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative
                elif noise_approx_order == 3:
                    noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
                        + diff**3 * noise_third_derivative / 6
                else:
                    print("The noise approximation order is not supported!")
                    exit()
                score = -noise / std.view(-1, 1, 1, 1)  # score -> noise
                drift = drift - diffusion.view(-1, 1, 1, 1)**2 * score * 0.5  # drift -> dx/dt



                temp1 = -(t - t_next) * recf[mr + 2 * (j-2) + 1] * torch.ones(n, device=sample.device)
                temp3 = -recf[mr + 2 * (j-2) + 2] * torch.ones(n, device=sample.device)
                temp2 = torch.ones(n, device=sample.device) - temp3

                ci1 = temp1 + temp2 * ci2 + temp3 * ci3
                # print("Shape of ci1 is " + str(ci1.shape))
                Y_j = temp1.view(-1, 1, 1, 1) * drift + temp2.view(-1, 1, 1, 1) * Y_j_1 + temp3.view(-1, 1, 1, 1) * Y_j_2

            # Update the intermediate variables
            Y_j_2 = Y_j_1
            Y_j_1 = Y_j

            ci3 = ci2
            ci2 = ci1


        # The finishing four-step procedure
        temp1 = -(t - t_next) * fpa[mz,0] * torch.ones(n, device=sample.device)
        diff = ci1 - t_start
        diff = diff.view(-1, 1, 1, 1)
        if noise_approx_order == 2:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative
        elif noise_approx_order == 3:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
                + diff**3 * noise_third_derivative / 6
        else:
            print("The noise approximation order is not supported!")
            exit()
        drift = drift_function(self.betas, self.config.num_train_timesteps, ci1, Y_j, noise)
        Y_j_1 = drift
        Y_j_3 = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1

        ci2 = ci1 + temp1
        temp1 = -(t - t_next) * fpa[mz,1] * torch.ones(n, device=sample.device)
        temp2 = -(t - t_next) * fpa[mz,2] * torch.ones(n, device=sample.device)
        diff = ci2 - t_start
        diff = diff.view(-1, 1, 1, 1)
        if noise_approx_order == 2:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative
        elif noise_approx_order == 3:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
                + diff**3 * noise_third_derivative / 6
        else:
            print("The noise approximation order is not supported!")
            exit()        
        drift = drift_function(self.betas, self.config.num_train_timesteps, ci2, Y_j_3, noise)
        Y_j_2 = drift
        Y_j_4 = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2

        ci2 = ci1 + temp1 + temp2
        temp1 = -(t - t_next) * fpa[mz,3] * torch.ones(n, device=sample.device)
        temp2 = -(t - t_next) * fpa[mz,4] * torch.ones(n, device=sample.device)
        temp3 = -(t - t_next) * fpa[mz,5] * torch.ones(n, device=sample.device)
        diff = ci2 - t_start
        diff = diff.view(-1, 1, 1, 1)
        if noise_approx_order == 2:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative
        elif noise_approx_order == 3:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
                + diff**3 * noise_third_derivative / 6
        else:
            print("The noise approximation order is not supported!")
            exit()
        drift = drift_function(self.betas, self.config.num_train_timesteps, ci2, Y_j_4, noise)
        Y_j_3 = drift
        fnt = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2 + temp3.view(-1, 1, 1, 1) * Y_j_3

        ci2 = ci1 + temp1 + temp2 + temp3
        temp1 = -(t - t_next) * fpb[mz,0] * torch.ones(n, device=sample.device)
        temp2 = -(t - t_next) * fpb[mz,1] * torch.ones(n, device=sample.device)
        temp3 = -(t - t_next) * fpb[mz,2] * torch.ones(n, device=sample.device)
        temp4 = -(t - t_next) * fpb[mz,3] * torch.ones(n, device=sample.device)
        diff = ci2 - t_start
        diff = diff.view(-1, 1, 1, 1)
        if noise_approx_order == 2:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative
        elif noise_approx_order == 3:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
                + diff**3 * noise_third_derivative / 6
        else:
            print("The noise approximation order is not supported!")
            exit()
        drift = drift_function(self.betas, self.config.num_train_timesteps, ci2, fnt, noise)
        Y_j_4 = drift
        Y_j = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2 + temp3.view(-1, 1, 1, 1) * Y_j_3 + temp4.view(-1, 1, 1, 1) * Y_j_4
        img_next = Y_j
        return SchedulerOutput(prev_sample=img_next)


    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        if self._step_index is None:
            self._step_index = 0
        n = sample.shape[0]
        total_step = self.config.num_train_timesteps
        beta_0, beta_1 = self.betas[0], self.betas[-1]

        sample = sample.to('cuda')

        noise_approx_order = 0
        if self._step_index == 0:
            # FIRST RUN
            # t = self.timesteps[self._step_index] / 999 #1000
            t = self.timesteps[self._step_index] / 1000
            t_start = torch.ones(n, device=sample.device) * t
            beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step
            log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
            std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

            # t_next = (self.timesteps[self._step_index + 1])/ 999 #1000
            t_next = self.timesteps[self._step_index + 1] / 1000
            # drift, diffusion -> f(x,t), g(t)
            drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * sample, torch.sqrt(beta_t)
            noise_initial = model_output
            score = -noise_initial / std.view(-1, 1, 1, 1)  # score -> noise
            drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt
            dt = t - t_next
            dt = torch.ones(n, device=sample.device) * dt #self.dt
            dt = dt.view(-1, 1, 1, 1)
            self.initial_sample = sample

            # Storing context
            self.first_dt = dt
            self.first_t_start = t_start
            self.first_t_end = torch.ones(n, device=sample.device) * t_next
            self.first_drfit = drift_initial
            self.first_t_next = t_next
            self.first_t = t
            self.first_noise_initial = noise_initial
            
            img_next = sample - dt * drift_initial
            self.noise_predictions.append(model_output)
            self._step_index += 1
            return SchedulerOutput(prev_sample=img_next)
        elif self._step_index == 1:
            # SECOND RUN
            # t = self.timesteps[self._step_index] / 999 #1000
            t = self.timesteps[self._step_index] / 1000
            beta_t = (beta_0 + t * (beta_1 - beta_0)) * total_step
            log_mean_coeff = (-0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0) * total_step
            std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
            drift_next, diffusion_next = -0.5 * beta_t.view(-1, 1, 1, 1) * sample, torch.sqrt(beta_t)
            noise_prediction_intermediate = model_output
            score = -noise_prediction_intermediate / std.view(-1, 1, 1, 1)  # score -> noise
            drift_next = drift_next - diffusion_next.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt


            # img_next = sample - 0.75 * self.first_dt * drift_next + 0.25 * self.first_dt * self.first_drfit
            img_next = sample - 1.5 * self.first_dt * drift_next + 0.5 * self.first_dt * self.first_drfit
            self.noise_predictions.append(model_output)
            self._step_index += 1
            return SchedulerOutput(prev_sample=img_next)
        elif self._step_index == 2:
            # THRID RUN 
            # h = 0.5 * self.first_dt
            h = self.first_dt
            noise_derivative = (3 * self.noise_predictions[0] - 4 * self.noise_predictions[1] + model_output) / (2 * h)
            noise_second_derivative = (self.noise_predictions[0] - 2 * self.noise_predictions[1] + model_output) / (h ** 2)
            self.dt_list.append(self.first_dt)
            noise_approx_order = 2
            self._step_index += 1
            # Restore context
            t = self.first_t
            t_start = self.first_t_start
            log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
            std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

            t_next = self.first_t_next
            drift_initial = self.first_drfit
            noise_initial = self.first_noise_initial
            dt = self.first_dt

            sample = self.initial_sample

        elif self._step_index == 3:
            # FOURTH RUN
            # t = self.timesteps[self._step_index] / 999 #1000
            t = self.timesteps[self._step_index] / 1000
            beta_0, beta_1 = self.betas[0], self.betas[-1]
            t_start = torch.ones(n, device=sample.device) * t
            beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step
            log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
            std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

            # t_next = (self.timesteps[self._step_index + 1])/ 999 #1000
            t_next = self.timesteps[self._step_index + 1] / 1000
            # drift, diffusion -> f(x,t), g(t)
            drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * sample, torch.sqrt(beta_t)
            noise_initial = model_output
            score = -noise_initial / std.view(-1, 1, 1, 1)  # score -> noise
            drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt


            dt = t - t_next
            dt = torch.ones(n, device=sample.device) * dt #self.dt
            dt = dt.view(-1, 1, 1, 1)


            h = 0.5 * dt
            noise_derivative = (-3 * noise_initial + 4 * self.noise_predictions[-1] - self.noise_predictions[-2]) / (2 * h)
            noise_second_derivative = (noise_initial - 2 * self.noise_predictions[-1] + self.noise_predictions[-2]) / (h ** 2)
            self.noise_predictions.append(noise_initial)
            self.dt_list.append(dt)
            noise_approx_order = 2
            self._step_index += 1
        elif self._step_index == 4:
            # FIFTH RUN

            # t = self.timesteps[self._step_index] / 999 #1000
            t = self.timesteps[self._step_index] / 1000
            beta_0, beta_1 = self.betas[0], self.betas[-1]
            t_start = torch.ones(n, device=sample.device) * t
            beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step
            log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
            std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
            if self._step_index == len(self.timesteps) - 1:
                # Tweedie's trick
                noise_last = model_output
                img_next = sample - std.view(-1, 1, 1, 1) * noise_last
                return SchedulerOutput(prev_sample=img_next)
            # t_next = (self.timesteps[self._step_index + 1])/ 999 #1000
            t_next = self.timesteps[self._step_index + 1] / 1000
            # drift, diffusion -> f(x,t), g(t)
            drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * sample, torch.sqrt(beta_t)
            noise_initial = model_output
            score = -noise_initial / std.view(-1, 1, 1, 1)  # score -> noise
            drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt


            dt = t - t_next
            dt = torch.ones(n, device=sample.device) * dt #self.dt
            dt = dt.view(-1, 1, 1, 1)



            h1 = self.dt_list[-1]
            h2 = self.dt_list[-2]

            noise_derivative = (-self.noise_predictions[-2] + 4 * self.noise_predictions[-1] - 3 * noise_initial) / (2 * h1)
            noise_second_derivative = 2 / (h1 * h2 * (h1 + h2)) * (self.noise_predictions[-2] * h1 - self.noise_predictions[-1] * (h1 + h2) + noise_initial * h2)
            
            self.noise_predictions.append(noise_initial)
            self.dt_list.append(dt)
            noise_approx_order = 2
            self._step_index += 1
        # elif self._step_index == 5:
        else:
            # t = self.timesteps[self._step_index] / 999 #1000
            t = self.timesteps[self._step_index] / 1000
            beta_0, beta_1 = self.betas[0], self.betas[-1]
            t_start = torch.ones(n, device=sample.device) * t
            beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step
            log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
            std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
            if self._step_index == len(self.timesteps) - 1:
                # Tweedie's trick
                noise_last = model_output
                img_next = sample - std.view(-1, 1, 1, 1) * noise_last
                return SchedulerOutput(prev_sample=img_next)
            # t_next = (self.timesteps[self._step_index + 1])/ 999 #1000
            t_next = self.timesteps[self._step_index + 1] / 1000
            # drift, diffusion -> f(x,t), g(t)
            drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * sample, torch.sqrt(beta_t)
            noise_initial = model_output
            score = -noise_initial / std.view(-1, 1, 1, 1)  # score -> noise
            drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt


            dt = t - t_next
            dt = torch.ones(n, device=sample.device) * dt #self.dt
            dt = dt.view(-1, 1, 1, 1)


            h = self.dt_list[-1]
            noise_derivative = (2 * self.noise_predictions[-3] - 9 * self.noise_predictions[-2] + 18 * self.noise_predictions[-1] - 11 * noise_initial) / (6 * h)
            noise_second_derivative = (-self.noise_predictions[-3] + 4 * self.noise_predictions[-2] -5 * self.noise_predictions[-1] + 2 * noise_initial) / (h**2)
            noise_third_derivative = (self.noise_predictions[-3] - 3 * self.noise_predictions[-2] + 3 * self.noise_predictions[-1] - noise_initial) / (h**3)

            self.noise_predictions.append(noise_initial)
            self.dt_list.append(dt)

            noise_approx_order = 3
            self._step_index += 1



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
                Y_j_1 = sample + temp1.view(-1, 1, 1, 1) * drift_initial
            else:
                beta_0, beta_1 = self.betas[0], self.betas[-1]
                beta_t = (beta_0 + ci1 * (beta_1 - beta_0)) * total_step

                log_mean_coeff = (-0.25 * ci1 ** 2 * (beta_1 - beta_0) - 0.5 * ci1 * beta_0) * total_step
                std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

                # drift, diffusion -> f(x,t), g(t)
                drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * Y_j_1, torch.sqrt(beta_t)
                '''
                Approximate the noise using Taylor expansion
                '''
                diff = ci1 - t_start
                diff = diff.view(-1, 1, 1, 1)
                if noise_approx_order == 2:
                    noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative
                elif noise_approx_order == 3:
                    noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
                        + diff**3 * noise_third_derivative / 6
                else:
                    print("The noise approximation order is not supported!")
                    exit()
                score = -noise / std.view(-1, 1, 1, 1)  # score -> noise
                drift = drift - diffusion.view(-1, 1, 1, 1)**2 * score * 0.5  # drift -> dx/dt



                temp1 = -(t - t_next) * recf[mr + 2 * (j-2) + 1] * torch.ones(n, device=sample.device)
                temp3 = -recf[mr + 2 * (j-2) + 2] * torch.ones(n, device=sample.device)
                temp2 = torch.ones(n, device=sample.device) - temp3

                ci1 = temp1 + temp2 * ci2 + temp3 * ci3
                # print("Shape of ci1 is " + str(ci1.shape))
                Y_j = temp1.view(-1, 1, 1, 1) * drift + temp2.view(-1, 1, 1, 1) * Y_j_1 + temp3.view(-1, 1, 1, 1) * Y_j_2

            # Update the intermediate variables
            Y_j_2 = Y_j_1
            Y_j_1 = Y_j

            ci3 = ci2
            ci2 = ci1


        # The finishing four-step procedure
        temp1 = -(t - t_next) * fpa[mz,0] * torch.ones(n, device=sample.device)
        diff = ci1 - t_start
        diff = diff.view(-1, 1, 1, 1)
        if noise_approx_order == 2:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative
        elif noise_approx_order == 3:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
                + diff**3 * noise_third_derivative / 6
        else:
            print("The noise approximation order is not supported!")
            exit()
        drift = drift_function(self.betas, self.config.num_train_timesteps, ci1, Y_j, noise)
        Y_j_1 = drift
        Y_j_3 = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1

        ci2 = ci1 + temp1
        temp1 = -(t - t_next) * fpa[mz,1] * torch.ones(n, device=sample.device)
        temp2 = -(t - t_next) * fpa[mz,2] * torch.ones(n, device=sample.device)
        diff = ci2 - t_start
        diff = diff.view(-1, 1, 1, 1)
        if noise_approx_order == 2:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative
        elif noise_approx_order == 3:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
                + diff**3 * noise_third_derivative / 6
        else:
            print("The noise approximation order is not supported!")
            exit()        
        drift = drift_function(self.betas, self.config.num_train_timesteps, ci2, Y_j_3, noise)
        Y_j_2 = drift
        Y_j_4 = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2

        ci2 = ci1 + temp1 + temp2
        temp1 = -(t - t_next) * fpa[mz,3] * torch.ones(n, device=sample.device)
        temp2 = -(t - t_next) * fpa[mz,4] * torch.ones(n, device=sample.device)
        temp3 = -(t - t_next) * fpa[mz,5] * torch.ones(n, device=sample.device)
        diff = ci2 - t_start
        diff = diff.view(-1, 1, 1, 1)
        if noise_approx_order == 2:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative
        elif noise_approx_order == 3:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
                + diff**3 * noise_third_derivative / 6
        else:
            print("The noise approximation order is not supported!")
            exit()
        drift = drift_function(self.betas, self.config.num_train_timesteps, ci2, Y_j_4, noise)
        Y_j_3 = drift
        fnt = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2 + temp3.view(-1, 1, 1, 1) * Y_j_3

        ci2 = ci1 + temp1 + temp2 + temp3
        temp1 = -(t - t_next) * fpb[mz,0] * torch.ones(n, device=sample.device)
        temp2 = -(t - t_next) * fpb[mz,1] * torch.ones(n, device=sample.device)
        temp3 = -(t - t_next) * fpb[mz,2] * torch.ones(n, device=sample.device)
        temp4 = -(t - t_next) * fpb[mz,3] * torch.ones(n, device=sample.device)
        diff = ci2 - t_start
        diff = diff.view(-1, 1, 1, 1)
        if noise_approx_order == 2:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative
        elif noise_approx_order == 3:
            noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
                + diff**3 * noise_third_derivative / 6
        else:
            print("The noise approximation order is not supported!")
            exit()
        drift = drift_function(self.betas, self.config.num_train_timesteps, ci2, fnt, noise)
        Y_j_4 = drift
        Y_j = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2 + temp3.view(-1, 1, 1, 1) * Y_j_3 + temp4.view(-1, 1, 1, 1) * Y_j_4
        img_next = Y_j
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