# Copyright 2022 Luping Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# AAAAC3NzaC1lZDI1NTE5AAAAIPF25Y78AoRvJe+afFENhsopMEFkO6E5sB7UOAPFFmeL
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import math
import torch as th
import torch.nn as nn
import numpy as np
import os

import runner.method as mtd


def get_schedule(args, config):
    if config['type'] == "quad":
        betas = (np.linspace(config['beta_start'] ** 0.5, config['beta_end'] ** 0.5, config['diffusion_step'], dtype=np.float64) ** 2)
    elif config['type'] == "linear":
        betas = np.linspace(config['beta_start'], config['beta_end'], config['diffusion_step'], dtype=np.float64)
    elif config['type'] == 'cosine':
        betas = betas_for_alpha_bar(config['diffusion_step'], lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    else:
        betas = None

    betas = th.from_numpy(betas).float()
    alphas = 1.0 - betas
    # \Bar{alpha} in the paper; culmulative product of all the alpha_t's
    alphas_cump = alphas.cumprod(dim=0)

    return betas, alphas_cump


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class Schedule(object):
    def __init__(self, args, config):
        device = th.device(args.device)
        betas, alphas_cump = get_schedule(args, config)

        self.betas, self.alphas_cump = betas.to(device), alphas_cump.to(device)
        self.alphas_cump_pre = th.cat([th.ones(1).to(device), self.alphas_cump[:-1]], dim=0)
        self.total_step = config['diffusion_step']

        self.method_name = args.method

        self.method = mtd.choose_method(args.method)  # add pflow
        self.ets = None

        self.noise_predictions = None

    def diffusion(self, img, t_end, t_start=0, noise=None):
        if noise is None:
            noise = th.randn_like(img)
        alpha = self.alphas_cump.index_select(0, t_end).view(-1, 1, 1, 1)
        img_n = img * alpha.sqrt() + noise * (1 - alpha).sqrt()

        return img_n, noise

    def denoising(self, img_n, t_end, t_start, model, first_step=False, pflow=False, img_initial=None):
        if pflow:
            drift = self.method(img_n, t_start, t_end, model, self.betas, self.total_step)

            return drift



        else:
            if self.method_name == "SRKL2" or self.method_name == "SRKG2" or self.method_name == "SROCK" \
                or self.method_name == "RKL2" or self.method_name == "RKG2" or self.method_name == "ROCK":
                '''
                IMPORTANT: If the method used is <SOMETHING>_APPROX, do NOT change the s here since it won't work!!!!1
                change the s below if you want to use <SOMETHING>_APPROX
                '''
                s = int(os.getenv("INTRA_S", 5))

                #print("This is " + str(self.method_name) + " method.")
                img_next = self.method(img_n, t_start, t_end, model, self.betas, self.total_step, s)
            elif self.method_name == "RKL2_APPROX" or self.method_name == "SRKL2_APPROX" or self.method_name == "RKG2_APPROX" \
                or self.method_name == "ROCK4":
                '''
                IMPORTANT: If the method used is <SOMETHING>_APPROX or ROCK4, change s here
                '''
                #print("Betas",  len(self.betas), self.betas)
                s = int(os.getenv("INTRA_S", 50))
                #print("s is " + str(s))
                if first_step:
                    self.noise_predictions = []
                    self.dt_list = []
                img_next = self.method(img_n, t_start, t_end, model, self.betas, self.total_step, s, self.noise_predictions, self.dt_list)                
            elif self.method_name == "RKL2_EXTRAPOLATION" or self.method_name == "RKL2_EXTRAPOLATION_ADAPTIVE"\
                or self.method_name == "ROCK4_EXTRAPOLATION":
                '''
                IMPORTANT: If the method used is <SOMETHING>_EXTRAPOLATION, change s here
                '''
                s = int(os.getenv("INTRA_S", 50))
                if first_step:
                    self.noise_predictions_fine = []
                    self.noise_predictions_coarse = []
                    self.dt_list_fine = []
                    self.dt_list_coarse = []
                img_next = self.method(img_n, t_start, t_end, model, self.betas, self.total_step, s, self.noise_predictions_fine, \
                                       self.noise_predictions_coarse, self.dt_list_fine, self.dt_list_coarse)
            elif self.method_name == "ROCK4_ADAPTIVE_TIMESTEP":
                '''
                IMPORTANT: If the method used is <SOMETHING>_ADAPTIVE_TIMESTEP, change s here
                '''
                s = int(os.getenv("INTRA_S", 50))
                if first_step:
                    self.noise_predictions = []
                    self.dt_list = []
                img_next = self.method(img_n, t_start, t_end, model, self.betas, self.total_step, s, self.noise_predictions, self.dt_list) 
            else:
                if first_step:
                    self.ets = []
                img_next = self.method(img_n, t_start, t_end, model, self.alphas_cump, self.ets)


            return img_next
        

