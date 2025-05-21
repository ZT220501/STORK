# Copyright 2022 Luping Liu
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

import sys
import copy
import torch as th

from scipy.io import loadmat



def choose_method(name):
    if name == 'DDIM':
        return gen_order_1
    elif name == 'S-PNDM':
        return gen_order_2
    elif name == 'F-PNDM':
        return gen_order_4
    elif name == 'FON':
        return gen_fon
    elif name == 'PF':
        return gen_pflow
    # Starting from now, methods below are our new methods
    elif name == 'SRKL2':
        return stochastic_rkl2
    elif name == 'SRKG2':
        return stochastic_rkg2
    elif name == 'RKL2_APPROX':
        return rkl2_approx
    elif name == 'SRKL2_APPROX':
        return srkl2_approx
    elif name == "RKG2_APPROX":
        return rkg2_approx
    elif name == 'RKL2':
        return rkl2
    elif name == 'RKG2':
        return rkg2
    elif name == 'RKL2_EXTRAPOLATION':
        return rkl2_extrapolation
    elif name == 'RKL2_EXTRAPOLATION_ADAPTIVE':
        return rkl2_extrapolation_adaptive
    # The ROCK4-based methods are the most successful ones up to 04/12/2025
    elif name == 'ROCK4':
        return rock4
    elif name == 'ROCK4_EXTRAPOLATION':
        return rock4_extrapolation
    elif name == 'ROCK4_ADAPTIVE_TIMESTEP':
        return rock4_adaptive_timestep
    elif name == 'DPM-Solver':
        return dpm_solver_3rd_order
    else:
        print("The method " + str(name) + " is not implemented; check your spelling.")
        return None



# This code is to enforce the terminal SNR to be 0, adopted from the paper https://arxiv.org/pdf/2305.08891
def enforce_zero_terminal_snr(betas):
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
        alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = th.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas


# Functions for the DPM-Solver method
def marginal_log_mean_coeff(t, beta_0, beta_1):
    """
    Compute log(alpha_t) of a given continuous-time label t in [0, T].
    """
    return (-0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0) * 1000


def marginal_lambda(t, beta_0, beta_1):
    """
    Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
    """
    log_mean_coeff = marginal_log_mean_coeff(t, beta_0, beta_1)
    log_std = 0.5 * th.log(1. - th.exp(2. * log_mean_coeff))
    return log_mean_coeff - log_std


def inverse_lambda(lamb, beta_0, beta_1):
    """
    Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
    """
    tmp = 2. * (beta_1 - beta_0) * th.logaddexp(-2. * lamb, th.zeros((1,)).to(lamb))
    Delta = beta_0**2 + tmp
    return tmp / (th.sqrt(Delta) + beta_0) / (beta_1 - beta_0)



'''
Implementation of the DPM-Solver 3rd order method in the PNDM setting
This is like an ablation study.
'''
def dpm_solver_3rd_order(img, t, t_next, model, betas, total_step):

    t = t / total_step
    t_next = t_next / total_step

    beta_0, beta_1 = betas[0], betas[-1]
    t_start = th.ones(img.shape[0], device=img.device) * t

    lambda_s, lambda_t = marginal_lambda(t, beta_0, beta_1), marginal_lambda(t_next, beta_0, beta_1)
    h = lambda_t - lambda_s
    lambda_s1 = lambda_s + 1. / 3. * h
    lambda_s2 = lambda_s + 2. / 3. * h
    s1 = inverse_lambda(lambda_s1, beta_0, beta_1)
    s2 = inverse_lambda(lambda_s2, beta_0, beta_1)
    log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = marginal_log_mean_coeff(t, beta_0, beta_1), marginal_log_mean_coeff(s1, beta_0, beta_1), marginal_log_mean_coeff(s2, beta_0, beta_1), marginal_log_mean_coeff(t_next, beta_0, beta_1)
    sigma_s, sigma_s1, sigma_s2, sigma_t = th.sqrt(1. - th.exp(2. * log_alpha_s)), th.sqrt(1. - th.exp(2. * log_alpha_s1)), th.sqrt(1. - th.exp(2. * log_alpha_s2)), th.sqrt(1. - th.exp(2. * log_alpha_t))
    alpha_s1, alpha_s2, alpha_t = th.exp(log_alpha_s1), th.exp(log_alpha_s2), th.exp(log_alpha_t)

    phi_11 = th.expm1(-1. / 3. * h)
    phi_12 = th.expm1(-2. / 3. * h)
    phi_1 = th.expm1(-h)
    phi_22 = th.expm1(-2. / 3. * h) / (2. / 3. * h) + 1.
    phi_2 = phi_1 / h + 1.
    phi_3 = phi_2 / h - 0.5

    model_s = model(img, t_start * (total_step - 1))
    x_s1 = (
        (sigma_s1 / sigma_s).view(-1, 1, 1, 1) * img
        - (alpha_s1 * phi_11).view(-1, 1, 1, 1) * model_s
    )
    model_s1 = model(x_s1, s1 * (total_step - 1))
    x_s2 = (
        (sigma_s2 / sigma_s).view(-1, 1, 1, 1) * img
        - (alpha_s2 * phi_12).view(-1, 1, 1, 1) * model_s
        + 2 * (alpha_s2 * phi_22).view(-1, 1, 1, 1) * (model_s1 - model_s)
    )
    model_s2 = model(x_s2, s2 * (total_step - 1))

    img_next = (
        (sigma_t / sigma_s).view(-1, 1, 1, 1) * img
        - (alpha_t * phi_1).view(-1, 1, 1, 1) * model_s
        + 1.5 * (alpha_t * phi_2).view(-1, 1, 1, 1) * (model_s2 - model_s)
    )


    print(img_next[0])
    return img_next



#######################
# ROCK4-based methods #
#######################
'''
Coefficients of ROCK4
'''
def coeff_rock4():

    # Degrees
    data = loadmat('ms.mat')
    ms = data['ms'][0]

    # Parameters for the finishing procedure
    data = loadmat('fpa.mat')
    fpa = data['fpa']

    data = loadmat('fpb.mat')
    fpb = data['fpb']

    data = loadmat('fpbe.mat')
    fpbe = data['fpbe']

    # Parameters for the recurrence procedure
    data = loadmat('recf.mat')
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
    mp = th.zeros(2)
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


'''
Drift function for the backward ODE in the diffusion model
'''
def drift_function(img, t, t_next, betas, total_step, t_eval, y_eval, noise):
    n = img.shape[0]

    beta_0, beta_1 = betas[0], betas[-1]
    beta_t = (beta_0 + t_eval * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_eval ** 2 * (beta_1 - beta_0) - 0.5 * t_eval * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

    # drift, diffusion -> f(x,t), g(t)
    drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * y_eval, th.sqrt(beta_t) 
    score = -noise / std.view(-1, 1, 1, 1)  # score -> noise
    drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt

    return drift












'''
ROCK4 extrapolation method
'''
def rock4_extrapolation(img, t, t_next, model, betas, total_step, s, noise_predictions_coarse, dt_list_coarse, noise_predictions_fine, dt_list_fine):
    '''
    img: image in the backward process at time t
    t: starting time
    t_next: ending time
    model: model used to the noise prediction/score
    betas: beta_t in the DDPM/DDIM training process. Use linear scheduling to get the corresponding beta_t
        at each intermediate timestep
    total_step: total super-steps in the backward process
    s: stage number for the stochatic RKL2
    '''

    print("")
    print("I'm using the RKL2 extrapolation method.")
    print("t current is " + str(t[0]))
    print("t next is " + str(t_next[0]))

    is_last = False
    # Change the last step a bit away from zero to avoid nan noise prediction 
    if t[0] == 0:
        is_last = True

    t_original = t
    t_next_original = t_next


    # Normalize t and t_next into the interval [0, 1]
    t = t / total_step
    t_next = t_next / total_step

    # NEW: Adaptive s; use large s at the very beginning while use small s afterward
    s = max(int(s * t[0]), 5)

    # Batch size
    n = img.shape[0]

    beta_0, beta_1 = betas[0], betas[-1]
    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))


    # Tweedie's trick
    if is_last:
        noise_last = model(img, th.zeros(n, device=img.device))
        print("The final noise is " + str(noise_last))
        img_next = img + std.view(-1, 1, 1, 1)**2 * noise_last
        return img_next
    

    gamma = 0.5
    t_intermediate = t_original - gamma * (t_original - t_next_original)

    # Avoid the blow-up during the last positive step
    # since the composition method will overshoot a little bit
    # if t_intermediate1[0] < 0 or t_intermediate2[0] < 0:
    if t_intermediate[0] < 0:
        img_next = rock4(img, t_original, t_next_original, model, betas, total_step, s, noise_predictions_coarse, dt_list_coarse)
        return img_next
    

    # TODO: Maybe change the 2s here to be some other value in order to guarantee the stability?
    image_coarse = rock4(img, t_original, t_next_original, model, betas, total_step, s, noise_predictions_coarse, dt_list_coarse)

    img_intermediate1 = rock4(img, t_original, t_intermediate, model, betas, total_step, s, noise_predictions_fine, dt_list_fine, noise_initial=noise_predictions_coarse[-1])
    img_fine = rock4(img_intermediate1, t_intermediate, t_next_original, model, betas, total_step, s, noise_predictions_fine, dt_list_fine)


    # TODO: Try the following three cases and see which one is the best
    # Setting A
    img_next = (8 * img_fine - image_coarse) / 7 
    # Setting B
    # img_next = (16 * img_fine - image_coarse) / 15
    # Setting C
    # img_next = (32 * img_fine - image_coarse) / 31


    print(img_next[0])
    return img_next



import os
'''
ROCK4 method; the is our main method and the most successful one up to 04/12/2025.
In ROCK4, Taylor expansion up to third order is used to approximate the noise.
In this method, 1 step=1 NFE except for the first step.
'''
def rock4(img, t, t_next, model, betas, total_step, s, noise_predictions, dt_list):
    '''
    img: image in the backward process at time t
    t: starting time
    t_next: ending time
    model: model used to the noise prediction/score
    betas: beta_t in the DDPM/DDIM training process. Use linear scheduling to get the corresponding beta_t
        at each intermediate timestep
    total_step: total super-steps in the backward process
    s: stage number for the ROCK4 method
    '''

    #print("I'm using the ODE ROCK4 approximation method.")



    n = img.shape[0]


    t = t[0].item()
    t_next = t_next[0].item()

    # # TODO: Check if this will largely affect the performance.
    # t = round(t, 5)
    # t_next = round(t_next, 5)

    #print(type(t))


    t_start = th.ones(n, device=img.device) * t
    t_end = th.ones(n, device=img.device) * t_next



    #print("t current is " + str(t))
    #print("t next is " + str(t_next))


    

    dt = t - t_next
    dt = th.ones(n, device=img.device) * dt
    dt = dt.view(-1, 1, 1, 1)


    beta_0, beta_1 = betas[0], betas[-1]
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))


    if t_next == -1:
        # Tweedie's trick
        # print("t start is " + str(t_start))
        # FIXME why is this better
        use_tweedie = os.getenv('USE_TWEEDIE', True)
        #print("Using tweedie: ", use_tweedie)
        if bool(use_tweedie):
            noise_last = model(img, t_start * (total_step - 1))
        else:
            noise_last =  model(img, t_start)
        #noise_last = model(img, t_start)
        #noise_last = model(img, t_start * (total_step - 1))
        img_next = img - std.view(-1, 1, 1, 1) * noise_last
        return img_next

    #FIRST RUN

    # drift, diffusion -> f(x,t), g(t)
    drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    noise_initial = model(img, t_start * (total_step - 1))

    score = -noise_initial / std.view(-1, 1, 1, 1)  # score -> noise
    drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt

    ###########################################################
    # This is the old way of calculating the noise derivative #
    # for the first two steps.                                #
    # If things are not working well, turn back to this.      #
    ###########################################################

    noise_approx_order = 0
    if len(noise_predictions) == 0:
        #SECOND RUN
        img_intermediate_prediction = img - 0.5 * dt * drift_initial

        # t_intermediate = t - 0.5 * (t - t_next)
        t_intermediate = t_start - 0.5 * (t_start - t_end)
        beta_t = (beta_0 + t_intermediate * (beta_1 - beta_0)) * total_step
        log_mean_coeff = (-0.25 * t_intermediate ** 2 * (beta_1 - beta_0) - 0.5 * t_intermediate * beta_0) * total_step
        std = th.sqrt(1. - th.exp(2. * log_mean_coeff))
        drift_next, diffusion_next = -0.5 * beta_t.view(-1, 1, 1, 1) * img_intermediate_prediction, th.sqrt(beta_t)

        noise_prediction_intermediate = model(img_intermediate_prediction, t_intermediate * (total_step - 1))
        score = -noise_prediction_intermediate / std.view(-1, 1, 1, 1)  # score -> noise
        drift_next = drift_next - diffusion_next.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt

        #THRID RUN
        # img_next_prediction = img_intermediate_prediction - 0.5 * dt * drift_next
        img_next_prediction = img_intermediate_prediction - 0.75 * dt * drift_next + 0.25 * dt * drift_initial
        noise_next_prediction = model(img_next_prediction, t_end * (total_step - 1))

        h = 0.5 * dt
        noise_derivative = (3 * noise_initial - 4 * noise_prediction_intermediate + noise_next_prediction) / (2 * h)
        noise_second_derivative = (noise_initial - 2 * noise_prediction_intermediate + noise_next_prediction) / (h ** 2)

        # We append the noise at T and T-1/2
        # The noise at T-1 is rather computed and stored for the next step.
        noise_predictions.append(noise_initial)
        noise_predictions.append(noise_prediction_intermediate)
        dt_list.append(dt)

        noise_approx_order = 2
    elif len(noise_predictions) == 2:
        # FOURTH RUN
        h = dt / 2

        noise_derivative = (-3 * noise_initial + 4 * noise_predictions[-1] - noise_predictions[-2]) / (2 * h)
        noise_second_derivative = (noise_initial - 2 * noise_predictions[-1] + noise_predictions[-2]) / (h ** 2)

        noise_predictions.append(noise_initial)
        dt_list.append(dt)
        noise_approx_order = 2
    elif len(noise_predictions) == 3:
        # FIFTH RUN
        h1 = dt_list[-1]
        h2 = dt_list[-2]

        noise_derivative = (-noise_predictions[-2] + 4 * noise_predictions[-1] - 3 * noise_initial) / (2 * h1)
        noise_second_derivative = 2 / (h1 * h2 * (h1 + h2)) * (noise_predictions[-2] * h1 - noise_predictions[-1] * (h1 + h2) + noise_initial * h2)

        noise_predictions.append(noise_initial)
        dt_list.append(dt)
        noise_approx_order = 2
    else:
        # ALL ELSE
        h = dt_list[-1]
        noise_derivative = (2 * noise_predictions[-3] - 9 * noise_predictions[-2] + 18 * noise_predictions[-1] - 11 * noise_initial) / (6 * h)
        noise_second_derivative = (-noise_predictions[-3] + 4 * noise_predictions[-2] - 5 * noise_predictions[-1] + 2 * noise_initial) / (h**2)
        noise_third_derivative = (noise_predictions[-3] - 3 * noise_predictions[-2] + 3 * noise_predictions[-1] - noise_initial) / (h**3)
        

        noise_predictions.append(noise_initial)
        dt_list.append(dt)

        noise_approx_order = 3


    Y_j_2 = img
    Y_j_1 = img
    Y_j = img

    ci1 = t_start
    ci2 = t_start
    ci3 = t_start

    # Coefficients of ROCK4
    ms, fpa, fpb, fpbe, recf = coeff_rock4()
    # Choose the degree that's in the precomputed table
    mdeg, mp = mdegr(s, ms)
    mz = int(mp[0])
    mr = int(mp[1])

    '''
    ROCK4 Update
    '''
    # The first stage
    for j in range(1, mdeg + 1):
        # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
        if j == 1:
            temp1 = -(t - t_next) * recf[mr] * th.ones(n, device=img.device)
            ci1 = t_start + temp1
            ci2 = ci1
            Y_j_2 = img
            Y_j_1 = img + temp1.view(-1, 1, 1, 1) * drift_initial
        else:
            beta_0, beta_1 = betas[0], betas[-1]
            beta_t = (beta_0 + ci1 * (beta_1 - beta_0)) * total_step

            log_mean_coeff = (-0.25 * ci1 ** 2 * (beta_1 - beta_0) - 0.5 * ci1 * beta_0) * total_step
            std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

            # drift, diffusion -> f(x,t), g(t)
            drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * Y_j_1, th.sqrt(beta_t)
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



            temp1 = -(t - t_next) * recf[mr + 2 * (j-2) + 1] * th.ones(n, device=img.device)
            temp3 = -recf[mr + 2 * (j-2) + 2] * th.ones(n, device=img.device)
            temp2 = th.ones(n, device=img.device) - temp3

            ci1 = temp1 + temp2 * ci2 + temp3 * ci3
            # print("Shape of ci1 is " + str(ci1.shape))
            Y_j = temp1.view(-1, 1, 1, 1) * drift + temp2.view(-1, 1, 1, 1) * Y_j_1 + temp3.view(-1, 1, 1, 1) * Y_j_2

        # Update the intermediate variables
        Y_j_2 = Y_j_1
        Y_j_1 = Y_j

        ci3 = ci2
        ci2 = ci1


    # The finishing four-step procedure
    temp1 = -(t - t_next) * fpa[mz,0] * th.ones(n, device=img.device)
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
    drift = drift_function(img, t, t_next, betas, total_step, ci1, Y_j, noise)
    Y_j_1 = drift
    Y_j_3 = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1

    ci2 = ci1 + temp1
    temp1 = -(t - t_next) * fpa[mz,1] * th.ones(n, device=img.device)
    temp2 = -(t - t_next) * fpa[mz,2] * th.ones(n, device=img.device)
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
    drift = drift_function(img, t, t_next, betas, total_step, ci2, Y_j_3, noise)
    Y_j_2 = drift
    Y_j_4 = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2

    ci2 = ci1 + temp1 + temp2
    temp1 = -(t - t_next) * fpa[mz,3] * th.ones(n, device=img.device)
    temp2 = -(t - t_next) * fpa[mz,4] * th.ones(n, device=img.device)
    temp3 = -(t - t_next) * fpa[mz,5] * th.ones(n, device=img.device)
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
    drift = drift_function(img, t, t_next, betas, total_step, ci2, Y_j_4, noise)
    Y_j_3 = drift
    fnt = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2 + temp3.view(-1, 1, 1, 1) * Y_j_3

    ci2 = ci1 + temp1 + temp2 + temp3
    temp1 = -(t - t_next) * fpb[mz,0] * th.ones(n, device=img.device)
    temp2 = -(t - t_next) * fpb[mz,1] * th.ones(n, device=img.device)
    temp3 = -(t - t_next) * fpb[mz,2] * th.ones(n, device=img.device)
    temp4 = -(t - t_next) * fpb[mz,3] * th.ones(n, device=img.device)
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
    drift = drift_function(img, t, t_next, betas, total_step, ci2, fnt, noise)
    Y_j_4 = drift

    Y_j = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2 + temp3.view(-1, 1, 1, 1) * Y_j_3 + temp4.view(-1, 1, 1, 1) * Y_j_4



    img_next = Y_j
    #print(img_next[0])
    return img_next




'''
ROCK4 method, with adaptive time stepping.
In ROCK4, Taylor expansion up to third order is used to approximate the noise.
In this method, 1 step=1 NFE
'''
def rock4_adaptive_timestep(img, t, t_next, model, betas, total_step, s, noise_predictions, dt_list, noise_initial=None, disp=True):
    '''
    img: image in the backward process at time t
    t: starting time
    t_next: ending time
    model: model used to the noise prediction/score
    betas: beta_t in the DDPM/DDIM training process. Use linear scheduling to get the corresponding beta_t
        at each intermediate timestep
    total_step: total super-steps in the backward process
    s: stage number for the ROCK4 method
    '''

    print("I'm using the ODE ROCK4 approximation method.")
    print("Length of noise predictions is " + str(len(noise_predictions)))



    n = img.shape[0]

    print("t current is " + str(t[0]))
    print("t next is " + str(t_next[0]))


    dt = t - t_next
    dt = th.ones(n, device=img.device) * dt
    dt = dt.view(-1, 1, 1, 1)


    beta_0, beta_1 = betas[0], betas[-1]
    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))


    # drift, diffusion -> f(x,t), g(t)
    drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    if noise_initial is None:
        noise_initial = model(img, t_start * (total_step - 1))
    
    score = -noise_initial / std.view(-1, 1, 1, 1)  # score -> noise
    drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt





    if len(noise_predictions) <= 1:
        img_temp = img - dt * drift_initial

        beta_0, beta_1 = betas[0], betas[-1]
        t_start = th.ones(n, device=img.device) * t
        t_end = th.ones(n, device=img.device) * t_next
        beta_t = (beta_0 + t_end * (beta_1 - beta_0)) * total_step

        log_mean_coeff = (-0.25 * t_end ** 2 * (beta_1 - beta_0) - 0.5 * t_end * beta_0) * total_step
        std = th.sqrt(1. - th.exp(2. * log_mean_coeff))


        # drift, diffusion -> f(x,t), g(t)
        drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * img_temp, th.sqrt(beta_t)        
        score = -model(img_temp, t_end * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
        drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5

        img_next = img - 0.5 * dt * (drift_initial + drift)

        noise_predictions.append(noise_initial)
        dt_list.append(dt)
        return img_next
    elif len(noise_predictions) == 2:
        h1 = dt_list[-1]
        h2 = dt_list[-2]

        noise_derivative = (-noise_predictions[-2] + 4 * noise_predictions[-1] - 3 * noise_initial) / (2 * h1)
        noise_second_derivative = 2 / (h1 * h2 * (h1 + h2)) * (noise_predictions[-2] * h1 - noise_predictions[-1] * (h1 + h2) + noise_initial * h2)
    else:
        h1 = dt_list[-1]
        h2 = h1 + dt_list[-2]
        h3 = h2 + dt_list[-3]

        noise_derivative = ((h2 * h3) * (noise_predictions[-1] - noise_initial) - (h1 * h3) * (noise_predictions[-2] - noise_initial) + (h1 * h2) * (noise_predictions[-3] - noise_initial)) / (h1 * h2 * h3)
        noise_second_derivative = 2 * ((h2 + h3) * (noise_predictions[-1] - noise_initial) - (h1 + h3) * (noise_predictions[-2] - noise_initial) + (h1 + h2) * (noise_predictions[-3] - noise_initial)) / (h1 * h2 * h3)
        noise_third_derivative = 6 * ((h2 - h3) * (noise_predictions[-1] - noise_initial) + (h3 - h1) * (noise_predictions[-2] - noise_initial) + (h1 - h2) * (noise_predictions[-3] - noise_initial)) / (h1 * h2 * h3)
    


    Y_j_2 = img
    Y_j_1 = img
    Y_j = img

    ci1 = t_start
    ci2 = t_start
    ci3 = t_start

    # Coefficients of ROCK4
    ms, fpa, fpb, fpbe, recf = coeff_rock4()
    # Choose the degree that's in the precomputed table
    mdeg, mp = mdegr(s, ms)
    mz = int(mp[0])
    mr = int(mp[1])

    '''
    ROCK4 Update
    secrets: AAAAC3NzaC1lZDI1NTE5AAAAIPF25Y78AoRvJe+afFENhsopMEFkO6E5sB7UOAPFFmeL
    '''
    # The first stage
    for j in range(1, mdeg + 1):
        # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
        if j == 1:
            temp1 = -(t - t_next) * recf[mr] * th.ones(n, device=img.device)
            ci1 = t_start + temp1
            ci2 = ci1
            Y_j_2 = img
            Y_j_1 = img + temp1.view(-1, 1, 1, 1) * drift_initial
        else:
            beta_0, beta_1 = betas[0], betas[-1]
            beta_t = (beta_0 + ci1 * (beta_1 - beta_0)) * total_step

            log_mean_coeff = (-0.25 * ci1 ** 2 * (beta_1 - beta_0) - 0.5 * ci1 * beta_0) * total_step
            std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

            # drift, diffusion -> f(x,t), g(t)
            drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * Y_j_1, th.sqrt(beta_t)
            '''
            Approximate the noise using Taylor expansion
            '''
            diff = ci1 - t_start
            diff = diff.view(-1, 1, 1, 1)
            if len(noise_predictions) <= 1:
                noise = noise_initial + diff * noise_derivative
            elif len(noise_predictions) == 2:
                noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative
            # elif len(noise_predictions) == 3:
            else:
                noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
                    + diff**3 * noise_third_derivative / 6
            # else:
            #     noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
            #         + diff**3 * noise_third_derivative / 6 + diff**4 * noise_fourth_derivative / 24
            score = -noise / std.view(-1, 1, 1, 1)  # score -> noise
            drift = drift - diffusion.view(-1, 1, 1, 1)**2 * score * 0.5  # drift -> dx/dt



            temp1 = -(t - t_next) * recf[mr + 2 * (j-2) + 1] * th.ones(n, device=img.device)
            temp3 = -recf[mr + 2 * (j-2) + 2] * th.ones(n, device=img.device)
            temp2 = th.ones(n, device=img.device) - temp3

            ci1 = temp1 + temp2 * ci2 + temp3 * ci3
            # print("Shape of ci1 is " + str(ci1.shape))
            Y_j = temp1.view(-1, 1, 1, 1) * drift + temp2.view(-1, 1, 1, 1) * Y_j_1 + temp3.view(-1, 1, 1, 1) * Y_j_2

        # Update the intermediate variables
        Y_j_2 = Y_j_1
        Y_j_1 = Y_j

        ci3 = ci2
        ci2 = ci1


    # The finishing four-step procedure
    temp1 = -(t - t_next) * fpa[mz,0] * th.ones(n, device=img.device)
    diff = ci1 - t_start
    diff = diff.view(-1, 1, 1, 1)
    if len(noise_predictions) <= 1:
        noise = noise_initial + diff * noise_derivative
    elif len(noise_predictions) == 2:
        noise = noise_initial + diff * noise_derivative + diff**2 * noise_second_derivative
    # elif len(noise_predictions) == 3:
    else:
        noise = noise_initial + diff * noise_derivative + diff**2 * noise_second_derivative \
            + diff**3 * noise_third_derivative
    # else:
    #     noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
    #         + diff**3 * noise_third_derivative / 6 + diff**4 * noise_fourth_derivative / 24
    drift = drift_function(img, t, t_next, betas, total_step, ci1, Y_j, noise)
    Y_j_1 = drift
    Y_j_3 = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1

    ci2 = ci1 + temp1
    temp1 = -(t - t_next) * fpa[mz,1] * th.ones(n, device=img.device)
    temp2 = -(t - t_next) * fpa[mz,2] * th.ones(n, device=img.device)
    diff = ci2 - t_start
    diff = diff.view(-1, 1, 1, 1)
    if len(noise_predictions) <= 1:
        noise = noise_initial + diff * noise_derivative
    elif len(noise_predictions) == 2:
        noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative
    # elif len(noise_predictions) == 3:
    else:
        noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
            + diff**3 * noise_third_derivative / 6
    # else:
    #     noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
    #         + diff**3 * noise_third_derivative / 6 + diff**4 * noise_fourth_derivative / 24
    drift = drift_function(img, t, t_next, betas, total_step, ci2, Y_j_3, noise)
    Y_j_2 = drift
    Y_j_4 = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2

    ci2 = ci1 + temp1 + temp2
    temp1 = -(t - t_next) * fpa[mz,3] * th.ones(n, device=img.device)
    temp2 = -(t - t_next) * fpa[mz,4] * th.ones(n, device=img.device)
    temp3 = -(t - t_next) * fpa[mz,5] * th.ones(n, device=img.device)
    diff = ci2 - t_start
    diff = diff.view(-1, 1, 1, 1)
    if len(noise_predictions) <= 1:
        noise = noise_initial + diff * noise_derivative
    elif len(noise_predictions) == 2:
        noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative
    # elif len(noise_predictions) == 3:
    else:
        noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
            + diff**3 * noise_third_derivative / 6
    # else:
    #     noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
    #         + diff**3 * noise_third_derivative / 6 + diff**4 * noise_fourth_derivative / 24
    drift = drift_function(img, t, t_next, betas, total_step, ci2, Y_j_4, noise)
    Y_j_3 = drift
    fnt = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2 + temp3.view(-1, 1, 1, 1) * Y_j_3

    ci2 = ci1 + temp1 + temp2 + temp3
    temp1 = -(t - t_next) * fpb[mz,0] * th.ones(n, device=img.device)
    temp2 = -(t - t_next) * fpb[mz,1] * th.ones(n, device=img.device)
    temp3 = -(t - t_next) * fpb[mz,2] * th.ones(n, device=img.device)
    temp4 = -(t - t_next) * fpb[mz,3] * th.ones(n, device=img.device)
    diff = ci2 - t_start
    diff = diff.view(-1, 1, 1, 1)
    if len(noise_predictions) <= 1:
        noise = noise_initial + diff * noise_derivative
    elif len(noise_predictions) == 2:
        noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative
    # elif len(noise_predictions) == 3:
    else:
        noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
            + diff**3 * noise_third_derivative / 6
    # else:
    #     noise = noise_initial + diff * noise_derivative + 0.5 * diff**2 * noise_second_derivative \
    #         + diff**3 * noise_third_derivative / 6 + diff**4 * noise_fourth_derivative / 24
    drift = drift_function(img, t, t_next, betas, total_step, ci2, fnt, noise)
    Y_j_4 = drift

    Y_j = Y_j + temp1.view(-1, 1, 1, 1) * Y_j_1 + temp2.view(-1, 1, 1, 1) * Y_j_2 + temp3.view(-1, 1, 1, 1) * Y_j_3 + temp4.view(-1, 1, 1, 1) * Y_j_4



    img_next = Y_j

    noise_predictions.append(noise_initial)
    dt_list.append(dt)

    print(img_next[0])
    return img_next
    















#######################################################################################################
# Methods above are our new methods, which are stochastic RKL2, RKG2, and ROCK.
# All of them are stabilized Runge-Kutta methods of second order, and solve the SDE (NOT ODE!!!!!)

# Methods below are the existing methods for solving the backward ODE. They can be used as a comparison
# to show that SDE-based methods achieve lower (better) FID compared to the ODE-based methods.
#######################################################################################################



def gen_pflow(img, t, t_next, model, betas, total_step):

    # print("This is PFlow running.")
    print("t is " + str(t))
    print("t next is " + str(t_next))
    print()

    n = img.shape[0]
    beta_0, beta_1 = betas[0], betas[-1]

    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step


    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

    # drift, diffusion -> f(x,t), g(t)
    drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    score = -model(img, t_start * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
    drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5  # drift -> dx/dt

    return drift


def gen_fon(img, t, t_next, model, alphas_cump, ets):
    print("This is FON running.")
    print("t is " + str(t[0]))
    print("t next is " + str(t_next[0]))

    t_list = [t, (t + t_next) / 2.0, t_next]

    if len(ets) > 2:
        noise = model(img, t)
        img_next = transfer(img, t, t-1, noise, alphas_cump)
        delta1 = img_next - img
        ets.append(delta1)
        delta = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise = model(img, t_list[0])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_1 = img_ - img
        ets.append(delta_1)

        img_2 = img + delta_1 * (t - t_next).view(-1, 1, 1, 1) / 2.0
        noise = model(img_2, t_list[1])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_2 = img_ - img

        img_3 = img + delta_2 * (t - t_next).view(-1, 1, 1, 1) / 2.0
        noise = model(img_3, t_list[1])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_3 = img_ - img

        img_4 = img + delta_3 * (t - t_next).view(-1, 1, 1, 1)
        noise = model(img_4, t_list[2])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_4 = img_ - img
        delta = (1 / 6.0) * (delta_1 + 2*delta_2 + 2*delta_3 + delta_4)

    img_next = img + delta * (t - t_next).view(-1, 1, 1, 1)
    return img_next


def gen_order_4(img, t, t_next, model, alphas_cump, ets):

    #print("t is " + str(t[0]))
    #print("t next is " + str(t_next[0]))
    
    t_list = [t, (t+t_next)/2, t_next]
    if len(ets) > 2:
        noise_ = model(img, t)
        ets.append(noise_)
        noise = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise = runge_kutta(img, t_list, model, alphas_cump, ets)

    img_next = transfer(img, t, t_next, noise, alphas_cump)

    return img_next


def runge_kutta(x, t_list, model, alphas_cump, ets):
    e_1 = model(x, t_list[0])
    ets.append(e_1)
    x_2 = transfer(x, t_list[0], t_list[1], e_1, alphas_cump)

    e_2 = model(x_2, t_list[1])
    x_3 = transfer(x, t_list[0], t_list[1], e_2, alphas_cump)

    e_3 = model(x_3, t_list[1])
    x_4 = transfer(x, t_list[0], t_list[2], e_3, alphas_cump)

    e_4 = model(x_4, t_list[2])
    et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

    return et


def gen_order_2(img, t, t_next, model, alphas_cump, ets):
    if len(ets) > 0:
        noise_ = model(img, t)
        ets.append(noise_)
        noise = 0.5 * (3 * ets[-1] - ets[-2])
    else:
        noise = improved_euler(img, t, t_next, model, alphas_cump, ets)

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


def improved_euler(x, t, t_next, model, alphas_cump, ets):
    e_1 = model(x, t)
    ets.append(e_1)
    x_2 = transfer(x, t, t_next, e_1, alphas_cump)

    e_2 = model(x_2, t_next)
    et = (e_1 + e_2) / 2
    # x_next = transfer(x, t, t_next, et, alphas_cump)

    return et


def gen_order_1(img, t, t_next, model, alphas_cump, ets):
    noise = model(img, t)
    ets.append(noise)
    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


def transfer(x, t, t_next, et, alphas_cump):
    at = alphas_cump[t.long() + 1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1, 1)

    x_delta = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x - \
                                1 / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et)

    x_next = x + x_delta
    return x_next



def transfer_stochastic(x, t, t_next, et, alphas_cump):

    dt = t - t_next
    dt = th.ones(x.shape[0], device=x.device) * dt
    dt = dt.view(-1, 1, 1, 1)

    at = alphas_cump[t.long() + 1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1, 1)

    sigma_square = (1 - at_next) * (1 - at / at_next) / (1 - at)
    sigma_square = sigma_square.view(-1, 1, 1, 1)

    x_delta = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x - \
                                1 / (at.sqrt() * (((1 - at_next - sigma_square) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et) \
                                + th.sqrt(sigma_square) * th.randn_like(x) * th.sqrt(dt)

    x_next = x + x_delta
    return x_next



def transfer_dev(x, t, t_next, et, alphas_cump):
    at = alphas_cump[t.long()+1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long()+1].view(-1, 1, 1, 1)

    x_start = th.sqrt(1.0 / at) * x - th.sqrt(1.0 / at - 1) * et
    x_start = x_start.clamp(-1.0, 1.0)

    x_next = x_start * th.sqrt(at_next) + th.sqrt(1 - at_next) * et

    return x_next










#########################################################
# Some of the implemented but not so successful methods #
#########################################################







# Auxiliary function to calculate the coefficients needed for the stochastic RKG2 method
def b_coeff(j):
    if j < 0:
        print("The b_j coefficient in the RKG method can't have j negative")
        return
    if j == 0:
        return 1
    if j == 1:
        return 1 / 3
    
    return 4 * (j - 1) * (j + 4) / (3 * j * (j + 1) * (j + 2) * (j + 3))

    

'''
Stochastic Runge-Kutta-Legendre second order method (SRKL2), using Taylor expansion of noises.
This is our new method.
'''
def rkl2_extrapolation_adaptive(img, t, t_next, model, betas, total_step, s, noise_predictions_fine, noise_predictions_coarse, dt_list_fine, dt_list_coarse):
    '''
    img: image in the backward process at time t
    t: starting time
    t_next: ending time
    model: model used to the noise prediction/score
    betas: beta_t in the DDPM/DDIM training process. Use linear scheduling to get the corresponding beta_t
        at each intermediate timestep
    total_step: total super-steps in the backward process
    s: stage number for the stochatic RKL2
    '''

    print("")
    print("I'm using the RKL2 composition method.")
    print("t current is " + str(t[0]))
    print("t next is " + str(t_next[0]))

    is_last = False
    # Change the last step a bit away from zero to avoid nan noise prediction 
    if t[0] == 0:
        is_last = True

    t_original = t
    t_next_original = t_next


    # Normalize t and t_next into the interval [0, 1]
    t = t / total_step
    t_next = t_next / total_step

    # NEW: Adaptive s; use large s at the very beginning while use small s afterward
    s = max(int(s * t[0]), 5)

    # Batch size
    n = img.shape[0]

    beta_0, beta_1 = betas[0], betas[-1]
    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))


    # Tweedie's trick
    if is_last:
        noise_last = model(img, th.zeros(n, device=img.device))
        print("The final noise is " + str(noise_last))
        img_next = img + std.view(-1, 1, 1, 1)**2 * noise_last
        return img_next
    

    gamma = 0.5
    t_intermediate = t_original - gamma * (t_original - t_next_original)

    # Avoid the blow-up during the last positive step
    # since the composition method will overshoot a little bit
    # if t_intermediate1[0] < 0 or t_intermediate2[0] < 0:
    if t_intermediate[0] < 0:
        img_next = rkl2_approx(img, t_original, t_next_original, model, betas, total_step, s, noise_predictions_coarse, dt_list_coarse)
        return img_next
    

    # TODO: Maybe change the 2s here to be some other value in order to guarantee the stability?
    image_coarse = rkl2_approx(img, t_original, t_next_original, model, betas, total_step, s, noise_predictions_coarse, dt_list_coarse, disp=False)

    img_intermediate1 = rkl2_approx(img, t_original, t_intermediate, model, betas, total_step, s, noise_predictions_fine, dt_list_fine, noise_initial=noise_predictions_coarse[-1], disp=False)
    img_fine = rkl2_approx(img_intermediate1, t_intermediate, t_next_original, model, betas, total_step, s, noise_predictions_fine, dt_list_fine, disp=False)



    img_next = (8 * img_fine - image_coarse) / 7


    print(img_next[0])
    return img_next





'''
Stochastic Runge-Kutta-Legendre second order method (SRKL2), using Taylor expansion of noises.
This is our new method.
'''
def rkl2_extrapolation(img, t, t_next, model, betas, total_step, s, noise_predictions_fine, noise_predictions_coarse, dt_list_fine, dt_list_coarse):
    '''
    img: image in the backward process at time t
    t: starting time
    t_next: ending time
    model: model used to the noise prediction/score
    betas: beta_t in the DDPM/DDIM training process. Use linear scheduling to get the corresponding beta_t
        at each intermediate timestep
    total_step: total super-steps in the backward process
    s: stage number for the stochatic RKL2
    '''

    print("")
    print("I'm using the RKL2 composition method.")
    print("t current is " + str(t[0]))
    print("t next is " + str(t_next[0]))

    is_last = False
    # Change the last step a bit away from zero to avoid nan noise prediction 
    if t[0] == 0:
        is_last = True

    t_original = t
    t_next_original = t_next


    # Normalize t and t_next into the interval [0, 1]
    t = t / total_step
    t_next = t_next / total_step

    # Batch size
    n = img.shape[0]

    beta_0, beta_1 = betas[0], betas[-1]
    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))


    # Tweedie's trick
    if is_last:
        noise_last = model(img, th.zeros(n, device=img.device))
        print("The final noise is " + str(noise_last))
        img_next = img + std.view(-1, 1, 1, 1)**2 * noise_last
        return img_next
    

    gamma = 0.5
    t_intermediate = t_original - gamma * (t_original - t_next_original)

    # Avoid the blow-up during the last positive step
    # since the composition method will overshoot a little bit
    # if t_intermediate1[0] < 0 or t_intermediate2[0] < 0:
    if t_intermediate[0] < 0:
        img_next = rkl2_approx(img, t_original, t_next_original, model, betas, total_step, s, noise_predictions_coarse, dt_list_coarse)
        return img_next
    

    # TODO: Maybe change the 2s here to be some other value in order to guarantee the stability?
    image_coarse = rkl2_approx(img, t_original, t_next_original, model, betas, total_step, s, noise_predictions_coarse, dt_list_coarse, disp=False)

    img_intermediate1 = rkl2_approx(img, t_original, t_intermediate, model, betas, total_step, s, noise_predictions_fine, dt_list_fine, noise_initial=noise_predictions_coarse[-1], disp=False)
    img_fine = rkl2_approx(img_intermediate1, t_intermediate, t_next_original, model, betas, total_step, s, noise_predictions_fine, dt_list_fine, disp=False)



    img_next = (8 * img_fine - image_coarse) / 7


    print(img_next[0])
    return img_next









'''
Stochastic Runge-Kutta-Legendre second order method (SRKL2), using Taylor expansion of noises.
This is our new method.
'''
def rkl2_approx(img, t, t_next, model, betas, total_step, s, noise_predictions, dt_list, noise_initial=None, disp=True):
    '''
    img: image in the backward process at time t
    t: starting time
    t_next: ending time
    model: model used to the noise prediction/score
    betas: beta_t in the DDPM/DDIM training process. Use linear scheduling to get the corresponding beta_t
        at each intermediate timestep
    total_step: total super-steps in the backward process
    s: stage number for the stochatic RKL2
    '''

    if disp:
        print("I'm using the ODE RKL2 approximation method.")
        print("t current is " + str(t[0]))
        print("t next is " + str(t_next[0]))

    is_last = False
    # Change the last step a bit away from zero to avoid nan noise prediction 
    if t[0] == 0:
        is_last = True


    # Normalize t and t_next into the interval [0, 1]
    t = t / total_step
    t_next = t_next / total_step

    # Batch size
    n = img.shape[0]

    dt = t - t_next
    dt = th.ones(n, device=img.device) * dt
    dt = dt.view(-1, 1, 1, 1)


    beta_0, beta_1 = betas[0], betas[-1]
    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))


    # Tweedie's trick
    if is_last:
        noise_last = model(img, th.zeros(n, device=img.device))
        print("The final noise is " + str(noise_last))
        img_next = img + std.view(-1, 1, 1, 1)**2 * noise_last
        return img_next


    # drift, diffusion -> f(x,t), g(t)
    drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    if noise_initial == None:
        noise_initial = model(img, t_start * (total_step - 1))
    score = -noise_initial / std.view(-1, 1, 1, 1)  # score -> noise
    drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt



    if noise_predictions == []:
        img_next_prediction = img - dt * drift_initial
        noise_next_prediction = model(img_next_prediction, t_next * th.ones(n, device=img.device) * (total_step - 1))
        noise_derivative = (noise_initial - noise_next_prediction) / dt
    elif len(noise_predictions) == 1:
        noise_derivative = (noise_predictions[-1] - noise_initial) / dt
    elif len(noise_predictions) <= 3:
        h1 = dt_list[-1]
        h2 = dt_list[-2]

        noise_derivative = (-noise_predictions[-2] + 4 * noise_predictions[-1] - 3 * noise_initial) / (2 * h1)

        # noise_derivative = (noise_predictions[-1] - noise_initial) / h1
        noise_second_derivative = 2 / (h1 * h2 * (h1 + h2)) * (noise_predictions[-2] * h1 - noise_predictions[-1] * (h1 + h2) + noise_initial * h2)
    else:
        h1 = dt_list[-1]
        h2 = dt_list[-2]
        h3 = dt_list[-3]
        h4 = dt_list[-4]

        # For now, assume that all the grids are uniform, so that here it's a O(h^3) approximation to the first derivative under uniform grid
        noise_derivative = (2 * noise_predictions[-3] - 9 * noise_predictions[-2] + 18 * noise_predictions[-1] - 11 * noise_initial) / (6 * h1)

        # noise_derivative = (noise_predictions[-1] - noise_initial) / h1
        noise_second_derivative = 2 / (h1 * h2 * h3 * (h1 + h2 + h3)) * (noise_predictions[-3] * h1 * h2 + noise_predictions[-2] * h1 * h3 - noise_predictions[-1] * (h1 + h2 + h3) * h3 + noise_initial * h2 * h3)
        noise_third_derivative = 2 / (h1 * h2 * h3 * h4 * (h1 + h2 + h3 + h4))\
              * (noise_predictions[-4] * h1 * h2 * h3 - noise_predictions[-3] * h1 * h2 * h4 \
                 + noise_predictions[-2] * h1 * h3 * h4 - noise_predictions[-1] * (h1 + h2 + h3 + h4) * h3 * h4 \
                    + noise_initial * h2 * h3 * h4)


    Y_j_2 = img
    Y_j_1 = img
    Y_j = img

    # Implementation of our stochastic Runge-Kutta-Legendre second order method
    for j in range(1, s + 1):

        # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
        if j > 1:
            if j == 2:
                fraction = 4 / (3 * (s**2 + s - 2))
            else:
                fraction = ((j - 1)**2 + (j - 1) - 2) / (s**2 + s - 2)
            t_intermediate = t_start - (t - t_next) * fraction * th.ones(n, device=img.device)

            beta_0, beta_1 = betas[0], betas[-1]
            beta_t = (beta_0 + t_intermediate * (beta_1 - beta_0)) * total_step

            log_mean_coeff = (-0.25 * t_intermediate ** 2 * (beta_1 - beta_0) - 0.5 * t_intermediate * beta_0) * total_step
            std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

            # drift, diffusion -> f(x,t), g(t)
            drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * Y_j_1, th.sqrt(beta_t)
            '''
            Approximate the noise using Taylor expansion
            '''
            # if noise_predictions == []:
            #     noise = model(Y_j_1, t_intermediate * (total_step - 1))
            if len(noise_predictions) <= 1:
                noise = noise_initial - fraction * dt * noise_derivative
            # if len(noise_predictions) <= 1:
            #     noise = model(Y_j_1, t_intermediate * (total_step - 1))
            elif len(noise_predictions) <= 3:
                noise = noise_initial - fraction * dt * noise_derivative + 0.5 * (fraction * dt)**2 * noise_second_derivative
            else:
                noise = noise_initial - fraction * dt * noise_derivative + 0.5 * (fraction * dt)**2 * noise_second_derivative \
                - (fraction * dt)**3 * noise_third_derivative / 6
            score = -noise / std.view(-1, 1, 1, 1)  # score -> noise
            drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5  # drift -> dx/dt

        if j == 1:
            mu_tilde = 4 / (3 * (s**2 + s - 2))
            # Y_j = Y_j_1 - dt * mu_tilde * drift_initial + mu_tilde * diffusion_initial.view(-1, 1, 1, 1) * gaussian_noise * th.sqrt(dt)
            Y_j = Y_j_1 - dt * mu_tilde * drift_initial
        else:
            if j == 2:
                mu = (2 * j - 1) / j
                nu = -(j - 1) / j
                mu_tilde = mu * 4 / (s**2 + s - 2)
                gamma_tilde = -2 / 3 * mu_tilde
            elif j == 3:
                mu = 3 * (2 * j - 1) / j * (j**2 + j - 2) / (2 * j * (j+1))
                nu = - 3 * (j - 1) / j * (j**2 + j - 2) / (2 * j * (j+1))
                mu_tilde = mu * 4 / (s**2 + s - 2)
                gamma_tilde = - 2 / 3 * mu_tilde
            else:
                mu = (2 * j - 1) * (j + 2) * (j - 1)**2 / (j * (j - 2) * (j + 1)**2)
                nu = -(j - 1)**3 * (j**2 - 4) / (j**3 * (j + 1) * (j - 3))
                mu_tilde = (2 * j - 1) * (j + 2) * (j - 1)**2 / (j * (j - 2) * (j + 1)**2) * 4 / (s**2 + s - 2)
                gamma_tilde = (((j - 1)**2 + j - 3)/(2 * j * (j - 1)) - 1) * mu_tilde



            # Deterministic case
            Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img - dt * mu_tilde * drift - dt * gamma_tilde * drift_initial


            # if is_last:
            #     print("This is the last step.")
            # if j == s:
            #     print("Drift max is " + str(th.max(drift)))
            #     print("Drift min is " + str(th.min(drift)))
            #     print("Diffusion max is " + str(th.max(diffusion)))
            #     print("Diffusion min is " + str(th.min(diffusion )))

            #     print("Number of stages is " + str(s))

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j



    img_next = Y_j

    noise_predictions.append(noise_initial)
    dt_list.append(dt)

    if disp:
        print(img_next[0])
    return img_next



'''
Stochastic Runge-Kutta-Gegenbauer second order method (SRKL2), using Taylor expansion of noises.
This is our new method.
'''
def rkg2_approx(img, t, t_next, model, betas, total_step, s, noise_predictions, dt_list, disp=True):
    '''
    img: image in the backward process at time t
    t: starting time
    t_next: ending time
    model: model used to the noise prediction/score
    betas: beta_t in the DDPM/DDIM training process. Use linear scheduling to get the corresponding beta_t
        at each intermediate timestep
    total_step: total super-steps in the backward process
    s: stage number for the stochatic RKL2
    '''

    if disp:
        print("I'm using the ODE RKG2 approximation method.")
        print("t current is " + str(t[0]))
        print("t next is " + str(t_next[0]))

    is_last = False
    # Change the last step a bit away from zero to avoid nan noise prediction 
    if t[0] == 0:
        is_last = True

    # Normalize t and t_next into the interval [0, 1]
    t = t / total_step
    t_next = t_next / total_step

    # Batch size
    n = img.shape[0]

    dt = t - t_next
    dt = th.ones(n, device=img.device) * dt
    dt = dt.view(-1, 1, 1, 1)


    beta_0, beta_1 = betas[0], betas[-1]
    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))


    # Tweedie's trick
    if is_last:
        noise_last = model(img, th.zeros(n, device=img.device))
        print("The final noise is " + str(noise_last))
        img_next = img + std.view(-1, 1, 1, 1)**2 * noise_last
        return img_next


    # drift, diffusion -> f(x,t), g(t)
    drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    noise_initial = model(img, t_start * (total_step - 1))
    score = -noise_initial / std.view(-1, 1, 1, 1)  # score -> noise
    drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt


    # TODO: Also try to use higher order one-sided derivative approximation to the higher order derivatives?
    img_tilde = img - dt * drift_initial
    if noise_predictions == []:
        noise_next_step_approx = model(img_tilde, t_next * (total_step - 1))
        noise_derivative = (noise_initial - noise_next_step_approx) / dt
    elif len(noise_predictions) == 1:
        h1 = dt_list[-1]
        noise_derivative = (noise_predictions[-1] - noise_initial) / h1
    elif len(noise_predictions) <= 3:
        h1 = dt_list[-1]
        h2 = dt_list[-2]
        noise_derivative = (noise_predictions[-1] - noise_initial) / h1
        noise_second_derivative = 2 / (h1 * h2 * (h1 + h2)) * (noise_predictions[-2] * h1 - noise_predictions[-1] * (h1 + h2) + noise_initial * h2)
    else:
        h1 = dt_list[-1]
        h2 = dt_list[-2]
        h3 = dt_list[-3]
        h4 = dt_list[-4]
        noise_derivative = (noise_predictions[-1] - noise_initial) / h1
        noise_second_derivative = 2 / (h1 * h2 * h3 * (h1 + h2 + h3)) * (noise_predictions[-3] * h1 * h2 + noise_predictions[-2] * h1 * h3 - noise_predictions[-1] * (h1 + h2 + h3) * h3 + noise_initial * h2 * h3)
        noise_third_derivative = 2 / (h1 * h2 * h3 * h4 * (h1 + h2 + h3 + h4))\
              * (noise_predictions[-4] * h1 * h2 * h3 - noise_predictions[-3] * h1 * h2 * h4 \
                 + noise_predictions[-2] * h1 * h3 * h4 - noise_predictions[-1] * (h1 + h2 + h3 + h4) * h3 * h4 \
                    + noise_initial * h2 * h3 * h4)



    # if noise_predictions == []:
    #     noise_next_step_approx = model(img_tilde, t_next * (total_step - 1))
    #     noise_derivative = (noise_initial - noise_next_step_approx) / dt
    # elif len(noise_predictions) == 1:
    #     noise_derivative = (noise_predictions[-1] - noise_initial) / dt
    # elif len(noise_predictions) <= 3:
    #     noise_derivative = (noise_predictions[-1] - noise_initial) / dt
    #     noise_second_derivative = (noise_predictions[-2] - 2 * noise_predictions[-1] + noise_initial) / dt**2
    # else:
    #     noise_derivative = (noise_predictions[-1] - noise_initial) / dt
    #     noise_second_derivative = (-noise_predictions[-3] + 4 * noise_predictions[-2] - 5 * noise_predictions[-1] + 2 * noise_initial) / dt**2
    #     noise_third_derivative = (-5 * noise_predictions[-4] + 18 * noise_predictions[-3] - 24 * noise_predictions[-2] + 14 * noise_predictions[-1] - 3 * noise_initial) / (2 * dt**3)
    


    Y_j_2 = img
    Y_j_1 = img
    Y_j = img

    # Implementation of our stochastic Runge-Kutta-Legendre second order method
    for j in range(1, s + 1):
        # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
        if j > 1:
            if j == 2:
                fraction = 4 / (3 * (s**2 + s - 2))
            else:
                fraction = ((j - 1)**2 + (j - 1) - 2) / (s**2 + s - 2)
            t_intermediate = t_start - (t - t_next) * fraction * th.ones(n, device=img.device)

            beta_0, beta_1 = betas[0], betas[-1]
            beta_t = (beta_0 + t_intermediate * (beta_1 - beta_0)) * total_step

            log_mean_coeff = (-0.25 * t_intermediate ** 2 * (beta_1 - beta_0) - 0.5 * t_intermediate * beta_0) * total_step
            std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

            # drift, diffusion -> f(x,t), g(t)
            drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * Y_j_1, th.sqrt(beta_t)
            '''
            Approximate the noise using Taylor expansion
            '''
            if len(noise_predictions) <= 1:
                noise = noise_initial - fraction * dt * noise_derivative
            elif len(noise_predictions) <= 3:
                noise = noise_initial - fraction * dt * noise_derivative + (fraction * dt)**2 * noise_second_derivative
            else:
                noise = noise_initial - fraction * dt * noise_derivative + (fraction * dt)**2 * noise_second_derivative \
                - (fraction * dt)**3 * noise_third_derivative
            score = -noise / std.view(-1, 1, 1, 1)  # score -> noise
            drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5  # drift -> dx/dt

        
        if j == 1:
            mu_tilde = 6 / ((s + 4) * (s - 1))
            Y_j = Y_j_1 - dt * mu_tilde * drift_initial
        else:
            mu = (2 * j + 1) * b_coeff(j) / (j * b_coeff(j - 1))
            nu = -(j + 1) * b_coeff(j) / (j * b_coeff(j - 2))
            mu_tilde = mu * 6 / ((s + 4) * (s - 1))
            gamma_tilde = -mu_tilde * (1 - j * (j + 1) * b_coeff(j-1)/ 2)


            # Probability flow ODE update
            Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img - dt * mu_tilde * drift - dt * gamma_tilde * drift_initial

            if j == s and disp:
                print("Drift max is " + str(th.max(drift)))
                print("Drift min is " + str(th.min(drift)))
                print("Diffusion max is " + str(th.max(diffusion)))
                print("Diffusion min is " + str(th.min(diffusion )))

                print("Number of stages is " + str(s))

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j



    img_next = Y_j

    noise_predictions.append(noise_initial)
    dt_list.append(dt)

    if disp:
        print(img_next[0])
    return img_next




'''
Stochastic Runge-Kutta-Legendre second order method (SRKL2), using Taylor expansion of noises.
This is our new method.
TODO: Currently it still seems that the srkl2_approx method does NOT work; perhaps
it's because the higher order Taylor expansion totally messed up the noise
'''
def srkl2_approx(img, t, t_next, model, betas, total_step, s, noise_predictions):
    '''
    img: image in the backward process at time t
    t: starting time
    t_next: ending time
    model: model used to the noise prediction/score
    betas: beta_t in the DDPM/DDIM training process. Use linear scheduling to get the corresponding beta_t
        at each intermediate timestep
    total_step: total super-steps in the backward process
    s: stage number for the stochatic RKL2
    '''

    print("I'm using the SDE RKL2 approximation method.")
    print("t current is " + str(t[0]))
    print("t next is " + str(t_next[0]))

    is_last = False
    # Change the last step a bit away from zero to avoid nan noise prediction 
    if t[0] == 0:
        is_last = True

    t_original = t
    t_next_original = t_next

    # Normalize t and t_next into the interval [0, 1]
    t = t / total_step
    t_next = t_next / total_step

    # Batch size
    n = img.shape[0]

    dt = t - t_next
    dt = th.ones(n, device=img.device) * dt
    dt = dt.view(-1, 1, 1, 1)


    beta_0, beta_1 = betas[0], betas[-1]
    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step


    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))


    # Tweedie's trick
    if is_last:
        noise_last = model(img, th.zeros(n, device=img.device))
        print("The final noise is " + str(noise_last))
        img_next = img + std.view(-1, 1, 1, 1)**2 * noise_last
        return img_next


    # drift, diffusion -> f(x,t), g(t)
    drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    noise_initial = model(img, t_start * (total_step - 1))
    score = -noise_initial / std.view(-1, 1, 1, 1)  # score -> noise
    drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score # drift -> dx/dt



    # TODO: Also try to use higher order one-sided derivative approximation to the higher order derivatives?
    # img_tilde = img - dt * drift_initial

    # Do a deterministic approximation to the next step using deterministic RKL? 
    # HOPEFULLY this would be able to work...
    img_tilde = rkl2_approx(img, t_original, (t_next_original + t_original) / 2, model, betas, total_step, s, noise_predictions)


    noise_next_step_approx = model(img_tilde, t_next * (total_step - 1))
    if noise_predictions == []:
        noise_derivative = (noise_initial - noise_next_step_approx) / dt
    else:
        noise_derivative = (noise_predictions[-1] - noise_next_step_approx) / (2 * dt)
    if len(noise_predictions) >= 2:
        noise_second_derivative = (noise_predictions[-1] - 2 * noise_initial + noise_next_step_approx) / dt**2


    # if noise_predictions == []:
    #     noise_next_step_approx = model(img_tilde, t_next * (total_step - 1))
    #     noise_derivative = (noise_initial - noise_next_step_approx) / dt
    # elif len(noise_predictions) == 1:
    #     noise_derivative = (noise_predictions[-1] - noise_initial) / dt
    # else:
    #     noise_derivative = (noise_predictions[-1] - noise_initial) / dt
    #     noise_second_derivative = (noise_predictions[-2] - 2 * noise_predictions[-1] + noise_initial) / dt**2
    # noise_next_step_approx = model(img_tilde, t_next * (total_step - 1))
    # noise_derivative = (noise_initial - noise_next_step_approx) / dt



    gaussian_noise = th.randn_like(img)

    Y_j_2 = img
    Y_j_1 = img
    Y_j = img

    # Implementation of our stochastic Runge-Kutta-Legendre second order method
    for j in range(1, s + 1):
        # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
        if j > 1:
            if j == 2:
                fraction = 4 / (3 * (s**2 + s - 2))
            else:
                fraction = ((j - 1)**2 + (j - 1) - 2) / (s**2 + s - 2)
            t_intermediate = t_start - (t - t_next) * fraction * th.ones(n, device=img.device)

            beta_0, beta_1 = betas[0], betas[-1]
            beta_t = (beta_0 + t_intermediate * (beta_1 - beta_0)) * total_step

            log_mean_coeff = (-0.25 * t_intermediate ** 2 * (beta_1 - beta_0) - 0.5 * t_intermediate * beta_0) * total_step
            std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

            # drift, diffusion -> f(x,t), g(t)
            drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * Y_j_1, th.sqrt(beta_t)
            '''
            Approximate the noise using Taylor expansion
            '''
            if len(noise_predictions) <= 1:
                noise = noise_initial - fraction * dt * noise_derivative
            else:
                noise = noise_initial - fraction * dt * noise_derivative + (fraction * dt)**2 * noise_second_derivative

            # print("The Taylor expansion noise max is " + str(noise))

            # noise = noise_initial - fraction * dt * noise_derivative
            score = -noise / std.view(-1, 1, 1, 1)  # score -> noise
            drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score  # drift -> dx/dt

        if j == 1:
            mu_tilde = 4 / (3 * (s**2 + s - 2))
            Y_j = Y_j_1 - dt * mu_tilde * drift_initial + mu_tilde * diffusion_initial.view(-1, 1, 1, 1) * gaussian_noise * th.sqrt(dt)        
        else:
            if j == 2:
                mu = (2 * j - 1) / j
                nu = -(j - 1) / j
                mu_tilde = mu * 4 / (s**2 + s - 2)
                gamma_tilde = -2 / 3 * mu_tilde
            elif j == 3:
                mu = 3 * (2 * j - 1) / j * (j**2 + j - 2) / (2 * j * (j+1))
                nu = - 3 * (j - 1) / j * (j**2 + j - 2) / (2 * j * (j+1))
                mu_tilde = mu * 4 / (s**2 + s - 2)
                gamma_tilde = - 2 / 3 * mu_tilde
            else:
                mu = (2 * j - 1) * (j + 2) * (j - 1)**2 / (j * (j - 2) * (j + 1)**2)
                nu = -(j - 1)**3 * (j**2 - 4) / (j**3 * (j + 1) * (j - 3))
                mu_tilde = (2 * j - 1) * (j + 2) * (j - 1)**2 / (j * (j - 2) * (j + 1)**2) * 4 / (s**2 + s - 2)
                gamma_tilde = (((j - 1)**2 + j - 3)/(2 * j * (j - 1)) - 1) * mu_tilde
                
            # # Stochastic case implementation 1
            # if j == s - 1:
            #     Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img \
            #         - dt * mu_tilde * drift - dt * gamma_tilde * drift_initial \
            #         + gaussian_noise * th.sqrt(dt) * 0.5 * diffusion.view(-1, 1, 1, 1)
            #     diffusion_s_2 = diffusion
            # elif j == s:
            #     Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img \
            #         - dt * mu_tilde * drift - dt * gamma_tilde * drift_initial \
            #         + gaussian_noise * th.sqrt(dt) * (diffusion.view(-1, 1, 1, 1) - diffusion_s_2.view(-1, 1, 1, 1))
            # else:
            #     Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img \
            #         - dt * mu_tilde * drift - dt * gamma_tilde * drift_initial



            # Stochastic case implementation 2
            # This seems to be the one that's working now
            Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img \
                - dt * mu_tilde * drift + mu_tilde * diffusion.view(-1, 1, 1, 1) * gaussian_noise * th.sqrt(dt) \
                - dt * gamma_tilde * drift_initial + gamma_tilde * diffusion_initial.view(-1, 1, 1, 1) * gaussian_noise * th.sqrt(dt)

            if j == s:
                print("Drift max is " + str(th.max(drift)))
                print("Drift min is " + str(th.min(drift)))
                print("Diffusion max is " + str(th.max(diffusion)))
                print("Diffusion min is " + str(th.min(diffusion )))

                print("Number of stages is " + str(s))

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j



    img_next = Y_j

    noise_predictions.append(noise_initial)

    print(img_next[0])
    return img_next





'''
Deterministic Runge-Kutta-Legendre second order method (RKL2).
This is our new method for solving the backward probability flow ODE
'''
def rkl2(img, t, t_next, model, betas, total_step, s):
    '''
    img: image in the backward process at time t
    t: starting time
    t_next: ending time
    model: model used to the noise prediction/score
    betas: beta_t in the DDPM/DDIM training process. Use linear scheduling to get the corresponding beta_t
        at each intermediate timestep
    total_step: total super-steps in the backward process
    s: stage number for the stochatic RKL2
    '''

    print("t current is " + str(t[0]))
    print("t next is " + str(t_next[0]))

    is_last = False
    # Change the last step a bit away from zero to avoid nan noise prediction 
    if t[0] == 0:
        is_last = True
        # t_next = total_step * 1e-3 * th.ones(t_next.shape, device=img.device)

    # Normalize t and t_next into the interval [0, 1]
    t = t / total_step
    t_next = t_next / total_step

    # Batch size
    n = img.shape[0]

    dt = t - t_next
    dt = th.ones(n, device=img.device) * dt
    dt = dt.view(-1, 1, 1, 1)


    beta_0, beta_1 = betas[0], betas[-1]
    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))


    if is_last:
        noise_last = model(img, th.zeros(n, device=img.device))
        print("The final noise is " + str(noise_last))
        img_next = img + std.view(-1, 1, 1, 1)**2 * noise_last
        return img_next

    # drift, diffusion -> f(x,t), g(t)
    drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    score = -model(img, t_start * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
    drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score * 0.5 # deterministic drift



    Y_j_2 = img
    Y_j_1 = img
    Y_j = img

    # Implementation of our stochastic Runge-Kutta-Legendre second order method
    for j in range(1, s + 1):

        # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
        if j > 1:
            if j == 2:
                t_intermediate = t_start - (t - t_next) * 4 / (3 * (s**2 + s - 2)) * th.ones(n, device=img.device)
            else:
                t_intermediate = t_start - (t - t_next) * ((j - 1)**2 + (j - 1) - 2) / (s**2 + s - 2) * th.ones(n, device=img.device)
            # t_intermediate = t_start - (t - t_next) * (j**2 + j - 2) / (s**2 + s - 2) * th.ones(n, device=img.device)

            beta_0, beta_1 = betas[0], betas[-1]
            beta_t = (beta_0 + t_intermediate * (beta_1 - beta_0)) * total_step

            log_mean_coeff = (-0.25 * t_intermediate ** 2 * (beta_1 - beta_0) - 0.5 * t_intermediate * beta_0) * total_step
            std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

            # drift, diffusion -> f(x,t), g(t)
            drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * Y_j_1, th.sqrt(beta_t)
            score = -model(Y_j_1, t_intermediate * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
            drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5 # deterministic drift

        if j == 1:
            mu_tilde = 4 / (3 * (s**2 + s - 2))
            Y_j = Y_j_1 - dt * mu_tilde * drift_initial
        else:
            if j == 2:
                mu = (2 * j - 1) / j
                nu = -(j - 1) / j
                mu_tilde = mu * 4 / (s**2 + s - 2)
                gamma_tilde = -2 / 3 * mu_tilde
            elif j == 3:
                mu = 3 * (2 * j - 1) / j * (j**2 + j - 2) / (2 * j * (j+1))
                nu = - 3 * (j - 1) / j * (j**2 + j - 2) / (2 * j * (j+1))
                mu_tilde = mu * 4 / (s**2 + s - 2)
                gamma_tilde = - 2 / 3 * mu_tilde
            else:
                mu = (2 * j - 1) * (j + 2) * (j - 1)**2 / (j * (j - 2) * (j + 1)**2)
                nu = -(j - 1)**3 * (j**2 - 4) / (j**3 * (j + 1) * (j - 3))
                mu_tilde = (2 * j - 1) * (j + 2) * (j - 1)**2 / (j * (j - 2) * (j + 1)**2) * 4 / (s**2 + s - 2)
                gamma_tilde = (((j - 1)**2 + j - 3)/(2 * j * (j - 1)) - 1) * mu_tilde


            # Check for the deterministic case
            Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img - dt * mu_tilde * drift - dt * gamma_tilde * drift_initial

            if j == s:
                print("Drift max is " + str(th.max(drift)))
                print("Drift min is " + str(th.min(drift)))
                print("Diffusion max is " + str(th.max(diffusion)))
                print("Diffusion min is " + str(th.min(diffusion)))

                print("Number of stages is " + str(s))

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j



    img_next = Y_j

    # if is_last:
    #     print("This is supposed to be the final image.")
    # print(img_next[0])
    return img_next




'''
Stochastic Runge-Kutta-Legendre second order method (SRKL2).
This is our new method.
'''
def stochastic_rkl2(img, t, t_next, model, betas, total_step, s):
    '''
    img: image in the backward process at time t
    t: starting time
    t_next: ending time
    model: model used to the noise prediction/score
    betas: beta_t in the DDPM/DDIM training process. Use linear scheduling to get the corresponding beta_t
        at each intermediate timestep
    total_step: total super-steps in the backward process
    s: stage number for the stochatic RKL2
    '''

    print("t current is " + str(t[0]))
    print("t next is " + str(t_next[0]))

    is_last = False
    # Change the last step a bit away from zero to avoid nan noise prediction 
    if t[0] == 0:
        is_last = True
        # t_next = total_step * 1e-3 * th.ones(t_next.shape, device=img.device)

    # Normalize t and t_next into the interval [0, 1]
    t = t / total_step
    t_next = t_next / total_step

    # Batch size
    n = img.shape[0]

    dt = t - t_next
    dt = th.ones(n, device=img.device) * dt
    dt = dt.view(-1, 1, 1, 1)


    beta_0, beta_1 = betas[0], betas[-1]
    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

    # Tweedie's trick
    if is_last:
        noise_last = model(img, th.zeros(n, device=img.device))
        print("The final noise is " + str(noise_last))
        img_next = img + std.view(-1, 1, 1, 1)**2 * noise_last
        return img_next

    # drift, diffusion -> f(x,t), g(t)
    drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    score = -model(img, t_start * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
    drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score # drift -> dx/dt




    gaussian_noise = th.randn_like(img)


    Y_j_2 = img
    Y_j_1 = img
    Y_j = img

    diffusion_s_2 = diffusion_initial
    # Implementation of our stochastic Runge-Kutta-Legendre second order method
    for j in range(1, s + 1):

        # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
        if j > 1:
            # t_intermediate = t_start - (t - t_next) * ((j - 1)**2 + (j - 1) - 2) / (s**2 + s - 2) * th.ones(n, device=img.device)
            if j == 2:
                t_intermediate = t_start - (t - t_next) * 4 / (3 * (s**2 + s - 2)) * th.ones(n, device=img.device)
            else:
                t_intermediate = t_start - (t - t_next) * ((j - 1)**2 + (j - 1) - 2) / (s**2 + s - 2) * th.ones(n, device=img.device)

            beta_0, beta_1 = betas[0], betas[-1]
            beta_t = (beta_0 + t_intermediate * (beta_1 - beta_0)) * total_step

            log_mean_coeff = (-0.25 * t_intermediate ** 2 * (beta_1 - beta_0) - 0.5 * t_intermediate * beta_0) * total_step
            std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

            # drift, diffusion -> f(x,t), g(t)
            drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * Y_j_1, th.sqrt(beta_t)
            score = -model(Y_j_1, t_intermediate * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
            drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score  # drift -> dx/dt

        if j == 1:
            mu_tilde = 4 / (3 * (s**2 + s - 2))
            Y_j = Y_j_1 - dt * mu_tilde * drift_initial - mu_tilde * diffusion_initial.view(-1, 1, 1, 1) * gaussian_noise * th.sqrt(dt)
            # Y_j = Y_j_1 - dt * mu_tilde * drift_initial
        else:
            if j == 2:
                mu = (2 * j - 1) / j
                nu = -(j - 1) / j
                mu_tilde = mu * 4 / (s**2 + s - 2)
                gamma_tilde = -2 / 3 * mu_tilde
            elif j == 3:
                mu = 3 * (2 * j - 1) / j * (j**2 + j - 2) / (2 * j * (j+1))
                nu = - 3 * (j - 1) / j * (j**2 + j - 2) / (2 * j * (j+1))
                mu_tilde = mu * 4 / (s**2 + s - 2)
                gamma_tilde = - 2 / 3 * mu_tilde
            else:
                mu = (2 * j - 1) * (j + 2) * (j - 1)**2 / (j * (j - 2) * (j + 1)**2)
                nu = -(j - 1)**3 * (j**2 - 4) / (j**3 * (j + 1) * (j - 3))
                mu_tilde = (2 * j - 1) * (j + 2) * (j - 1)**2 / (j * (j - 2) * (j + 1)**2) * 4 / (s**2 + s - 2)
                gamma_tilde = (((j - 1)**2 + j - 3)/(2 * j * (j - 1)) - 1) * mu_tilde

            # # Stochastic case implementation 1
            # if j == s - 1:
            #     Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img \
            #         - dt * mu_tilde * drift - dt * gamma_tilde * drift_initial \
            #         + gaussian_noise * th.sqrt(dt) * 0.5 * diffusion.view(-1, 1, 1, 1)
            #     diffusion_s_2 = diffusion
            # elif j == s:
            #     Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img \
            #         - dt * mu_tilde * drift - dt * gamma_tilde * drift_initial \
            #         + gaussian_noise * th.sqrt(dt) * (diffusion.view(-1, 1, 1, 1) - diffusion_s_2.view(-1, 1, 1, 1))
            # else:
            #     Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img \
            #         - dt * mu_tilde * drift - dt * gamma_tilde * drift_initial \


            # Stochastic case implementation 2
            # This seems to be the one that's working now
            Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img \
                - dt * mu_tilde * drift - mu_tilde * diffusion.view(-1, 1, 1, 1) * gaussian_noise * th.sqrt(dt) \
                - dt * gamma_tilde * drift_initial - gamma_tilde * diffusion_initial.view(-1, 1, 1, 1) * gaussian_noise * th.sqrt(dt)




            if is_last:
                print("This is the last step.")
            if j == s:
                print("Drift max is " + str(th.max(drift)))
                print("Drift min is " + str(th.min(drift)))
                print("Diffusion max is " + str(th.max(diffusion)))
                print("Diffusion min is " + str(th.min(diffusion)))

                print("Number of stages is " + str(s))

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j



    img_next = Y_j

    # if is_last:
    #     print("This is supposed to be the final image.")
    # print(img_next[0])
    return img_next




'''
Deterministic Runge-Kutta-Gegenbauer second order method (RKG2).
This is our new method.
'''
def rkg2(img, t, t_next, model, betas, total_step, s):
    '''
    img: image in the backward process at time t
    t: starting time
    t_next: ending time
    model: model used to the noise prediction/score
    betas: beta_t in the DDPM/DDIM training process. Use linear scheduling to get the corresponding beta_t
        at each intermediate timestep
    total_step: total super-steps in the backward process
    s: stage number for the stochatic RKL2
    '''

    print("t current is " + str(t[0]))
    print("t next is " + str(t_next[0]))

    is_last = False
    # Prepare for the Tweedie's trick
    if t[0] == 0:
        is_last = True

    # Renormalize t and t_next into the interval [0, 1]
    t = t / total_step
    t_next = t_next / total_step

    # Batch size
    n = img.shape[0]


    # Calculate the drift and diffusion terms in the backward SDE
    beta_0, beta_1 = betas[0], betas[-1]
    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

    # Tweedie's trick
    if is_last:
        noise_last = model(img, th.zeros(n, device=img.device))
        print("The final noise is " + str(noise_last))
        img_next = img + std.view(-1, 1, 1, 1)**2 * noise_last
        return img_next

    # drift, diffusion -> f(x,t), g(t)
    drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    score = -model(img, t_start * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
    drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt


    dt = t - t_next
    dt = th.ones(n, device=img.device) * dt
    dt = dt.view(-1, 1, 1, 1)


    # Implementation of our stochastic Runge-Kutta-Gegenbauer second order method
    Y_j_2 = img
    Y_j_1 = img
    Y_j = img

    for j in range(1, s + 1):
        # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
        if j > 1:
            if j == 2:
                t_intermediate = t_start - (t - t_next) * 6 / ((s + 4) * (s - 1)) * th.ones(n, device=img.device)
            else:
                t_intermediate = t_start - (t - t_next) * ((j + 3) * (j - 2)) / ((s + 4) * (s - 1)) * th.ones(n, device=img.device)

            beta_0, beta_1 = betas[0], betas[-1]
            beta_t = (beta_0 + t_intermediate * (beta_1 - beta_0)) * total_step

            log_mean_coeff = (-0.25 * t_intermediate ** 2 * (beta_1 - beta_0) - 0.5 * t_intermediate * beta_0) * total_step
            std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

            # drift, diffusion -> f(x,t), g(t)
            drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * Y_j_1, th.sqrt(beta_t)
            score = -model(Y_j_1, t_intermediate * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
            drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt

        if j == 1:
            mu_tilde = 6 / ((s + 4) * (s - 1))
            Y_j = Y_j_1 - dt * mu_tilde * drift_initial
        else:
            mu = (2 * j + 1) * b_coeff(j) / (j * b_coeff(j - 1))
            nu = -(j + 1) * b_coeff(j) / (j * b_coeff(j - 2))
            mu_tilde = mu * 6 / ((s + 4) * (s - 1))
            gamma_tilde = -mu_tilde * (1 - j * (j + 1) * b_coeff(j-1)/ 2)


            # Probability flow ODE update
            Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img - dt * mu_tilde * drift - dt * gamma_tilde * drift_initial


            
            if j == s:
                print("beta_t is " + str((beta_0 + t_intermediate* (beta_1 - beta_0) * total_step)[0]))
                print("Noise predictor max is " + str(th.max(model(Y_j_1, t - (t - t_next) * ((j - 1)**2 + (j - 1) - 2) / (s**2 + s - 2)))))
                print("Noise predictor min is " + str(th.min(model(Y_j_1, t - (t - t_next) * ((j - 1)**2 + (j - 1) - 2) / (s**2 + s - 2)))))
        Y_j_2 = Y_j_1
        Y_j_1 = Y_j



    img_next = Y_j

    if is_last:
        print("This is supposed to be the final image.")
    print(img_next[0])
    return img_next





'''
Stochastic Runge-Kutta-Gegenbauer second order method (SRKG2).
This is our new method.
'''
def stochastic_rkg2(img, t, t_next, model, betas, total_step, s):
    '''
    img: image in the backward process at time t
    t: starting time
    t_next: ending time
    model: model used to the noise prediction/score
    betas: beta_t in the DDPM/DDIM training process. Use linear scheduling to get the corresponding beta_t
        at each intermediate timestep
    total_step: total super-steps in the backward process
    s: stage number for the stochatic RKL2
    '''

    print("t current is " + str(t[0]))
    print("t next is " + str(t_next[0]))

    is_last = False
    # Change the last step a bit away from zero to avoid nan noise prediction 
    if t[0] == 0:
        is_last = True

    # Renormalize t and t_next into the interval [0, 1]
    t = t / total_step
    t_next = t_next / total_step

    # Batch size
    n = img.shape[0]


    # Calculate the drift and diffusion terms in the backward SDE
    beta_0, beta_1 = betas[0], betas[-1]
    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

    # Tweedie's trick
    if is_last:
        noise_last = model(img, th.zeros(n, device=img.device))
        print("The final noise is " + str(noise_last))
        img_next = img + std.view(-1, 1, 1, 1)**2 * noise_last
        return img_next

    # drift, diffusion -> f(x,t), g(t)
    drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    score = -model(img, t_start * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
    drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score  # drift -> dx/dt


    dt = t - t_next
    dt = th.ones(n, device=img.device) * dt
    dt = dt.view(-1, 1, 1, 1)

    gaussian_noise = th.randn_like(img)


    # Implementation of our stochastic Runge-Kutta-Gegenbauer second order method
    Y_j_2 = img
    Y_j_1 = img
    Y_j = img

    for j in range(1, s + 1):
        # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
        if j > 1:
            if j == 2:
                t_intermediate = t_start - (t - t_next) * 6 / ((s + 4) * (s - 1)) * th.ones(n, device=img.device)
            else:
                t_intermediate = t_start - (t - t_next) * ((j + 3) * (j - 2)) / ((s + 4) * (s - 1)) * th.ones(n, device=img.device)

            beta_0, beta_1 = betas[0], betas[-1]
            beta_t = (beta_0 + t_intermediate * (beta_1 - beta_0)) * total_step

            log_mean_coeff = (-0.25 * t_intermediate ** 2 * (beta_1 - beta_0) - 0.5 * t_intermediate * beta_0) * total_step
            std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

            # drift, diffusion -> f(x,t), g(t)
            drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * Y_j_1, th.sqrt(beta_t)
            score = -model(Y_j_1, t_intermediate * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
            drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score  # drift -> dx/dt

        if j == 1:
            mu_tilde = 6 / ((s + 4) * (s - 1))
            Y_j = Y_j_1 - dt * mu_tilde * drift_initial - mu_tilde * diffusion_initial.view(-1, 1, 1, 1) * gaussian_noise * th.sqrt(dt)
        else:
            mu = (2 * j + 1) * b_coeff(j) / (j * b_coeff(j - 1))
            nu = -(j + 1) * b_coeff(j) / (j * b_coeff(j - 2))
            mu_tilde = mu * 6 / ((s + 4) * (s - 1))
            gamma_tilde = -mu_tilde * (1 - j * (j + 1) * b_coeff(j-1)/ 2)



            Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img \
                - dt * mu_tilde * drift - mu_tilde * diffusion.view(-1, 1, 1, 1) * gaussian_noise * th.sqrt(dt) \
                - dt * gamma_tilde * drift_initial - gamma_tilde * diffusion_initial.view(-1, 1, 1, 1) * gaussian_noise * th.sqrt(dt)


            
            if j == s:
                print("beta_t is " + str((beta_0 + t_intermediate* (beta_1 - beta_0) * total_step)[0]))
                print("Noise predictor max is " + str(th.max(model(Y_j_1, t - (t - t_next) * ((j - 1)**2 + (j - 1) - 2) / (s**2 + s - 2)))))
                print("Noise predictor min is " + str(th.min(model(Y_j_1, t - (t - t_next) * ((j - 1)**2 + (j - 1) - 2) / (s**2 + s - 2)))))
        Y_j_2 = Y_j_1
        Y_j_1 = Y_j



    img_next = Y_j

    if is_last:
        print("This is supposed to be the final image.")
    print(img_next[0])
    return img_next





#######################################################################################################
# Code that might be useful but more than likely won't
#######################################################################################################


# '''
# Stochastic Runge-Kutta-Legendre second order method (SRKL2), using Taylor expansion of noises.
# This is our new method.
# '''
# def rkl2_composition(img, t, t_next, model, betas, total_step, s, noise_predictions):
#     '''
#     img: image in the backward process at time t
#     t: starting time
#     t_next: ending time
#     model: model used to the noise prediction/score
#     betas: beta_t in the DDPM/DDIM training process. Use linear scheduling to get the corresponding beta_t
#         at each intermediate timestep
#     total_step: total super-steps in the backward process
#     s: stage number for the stochatic RKL2
#     '''

#     print("I'm using the composition RKL2 approximation method.")
#     print("t current is " + str(t[0]))
#     print("t next is " + str(t_next[0]))



#     is_last = False
#     # Change the last step a bit away from zero to avoid nan noise prediction 
#     if t[0] == 0:
#         is_last = True

#     # Normalize t and t_next into the interval [0, 1]
#     t = t / total_step
#     t_next = t_next / total_step

#     # Batch size
#     n = img.shape[0]


#     '''
#     Composition method with symmetric compisition steps
#     The coefficients in the composition are gamma1=1.35121, gamma2=-1.70242, gamma3=1.35121, which is symmetric
#     '''
#     dt = t - t_next
#     dt = th.ones(n, device=img.device) * dt
#     dt = dt.view(-1, 1, 1, 1)


#     beta_0, beta_1 = betas[0], betas[-1]
#     t_start = th.ones(n, device=img.device) * t
#     beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

#     log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
#     std = th.sqrt(1. - th.exp(2. * log_mean_coeff))


#     # Tweedie's trick
#     if is_last:
#         noise_last = model(img, th.zeros(n, device=img.device))
#         print("The final noise is " + str(noise_last))
#         img_next = img + std.view(-1, 1, 1, 1)**2 * noise_last
#         return img_next


#     # drift, diffusion -> f(x,t), g(t)
#     drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
#     noise_initial = model(img, t_start * (total_step - 1))
#     score = -noise_initial / std.view(-1, 1, 1, 1)  # score -> noise
#     drift_initial = drift_initial - diffusion_initial.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt



#     # Taylor expansion approximation of noise
#     img_tilde = img - dt * drift_initial
#     if noise_predictions == []:
#         noise_next_step_approx = model(img_tilde, t_next * (total_step - 1))
#         noise_derivative = (noise_initial - noise_next_step_approx) / dt
#     elif len(noise_predictions) == 1:
#         noise_derivative = (noise_predictions[-1] - noise_initial) / dt
#     elif len(noise_predictions) <= 3:
#         noise_derivative = (noise_predictions[-1] - noise_initial) / dt
#         noise_second_derivative = (noise_predictions[-2] - 2 * noise_predictions[-1] + noise_initial) / dt**2
#     else:
#         noise_derivative = (noise_predictions[-1] - noise_initial) / dt
#         noise_second_derivative = (-noise_predictions[-3] + 4 * noise_predictions[-2] - 5 * noise_predictions[-1] + 2 * noise_initial) / dt**2
#         noise_third_derivative = (-5 * noise_predictions[-4] + 18 * noise_predictions[-3] - 24 * noise_predictions[-2] + 14 * noise_predictions[-1] - 3 * noise_initial) / (2 * dt**3)


#     gamma1 = 1 / (2 - 2 ** (1/3))
#     gamma2 = 1 - 2 * gamma1
#     gamma3 = gamma1



#     '''
#     First step in the composition method
#     '''
#     dt = gamma1 * (t - t_next)
#     dt = th.ones(n, device=img.device) * dt
#     dt = dt.view(-1, 1, 1, 1)


#     beta_0, beta_1 = betas[0], betas[-1]
#     t_start = th.ones(n, device=img.device) * t
#     beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

#     log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
#     std = th.sqrt(1. - th.exp(2. * log_mean_coeff))


#     Y_j_2 = img
#     Y_j_1 = img
#     Y_j = img

#     # Implementation of our Runge-Kutta-Legendre second order method
#     for j in range(1, s + 1):

#         # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
#         if j > 1:
#             if j == 2:
#                 fraction = 4 / (3 * (s**2 + s - 2))
#             else:
#                 fraction = ((j - 1)**2 + (j - 1) - 2) / (s**2 + s - 2)
#             t_intermediate = t_start - gamma1 * (t - t_next) * fraction * th.ones(n, device=img.device)

#             beta_0, beta_1 = betas[0], betas[-1]
#             beta_t = (beta_0 + t_intermediate * (beta_1 - beta_0)) * total_step

#             log_mean_coeff = (-0.25 * t_intermediate ** 2 * (beta_1 - beta_0) - 0.5 * t_intermediate * beta_0) * total_step
#             std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

#             # drift, diffusion -> f(x,t), g(t)
#             drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * Y_j_1, th.sqrt(beta_t)
#             '''
#             Approximate the noise using Taylor expansion
#             '''
#             if len(noise_predictions) <= 1:
#                 noise = noise_initial - fraction * dt * noise_derivative
#             elif len(noise_predictions) <= 3:
#                 noise = noise_initial - fraction * dt * noise_derivative + (fraction * dt)**2 * noise_second_derivative
#             else:
#                 noise = noise_initial - fraction * dt * noise_derivative + (fraction * dt)**2 * noise_second_derivative \
#                 - (fraction * dt)**3 * noise_third_derivative
#             score = -noise / std.view(-1, 1, 1, 1)  # score -> noise
#             drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5  # drift -> dx/dt

#         if j == 1:
#             mu_tilde = 4 / (3 * (s**2 + s - 2))
#             # Y_j = Y_j_1 - dt * mu_tilde * drift_initial + mu_tilde * diffusion_initial.view(-1, 1, 1, 1) * gaussian_noise * th.sqrt(dt)
#             Y_j = Y_j_1 - dt * mu_tilde * drift_initial
#         else:
#             if j == 2:
#                 mu = (2 * j - 1) / j
#                 nu = -(j - 1) / j
#                 mu_tilde = mu * 4 / (s**2 + s - 2)
#                 gamma_tilde = -2 / 3 * mu_tilde
#             elif j == 3:
#                 mu = 3 * (2 * j - 1) / j * (j**2 + j - 2) / (2 * j * (j+1))
#                 nu = - 3 * (j - 1) / j * (j**2 + j - 2) / (2 * j * (j+1))
#                 mu_tilde = mu * 4 / (s**2 + s - 2)
#                 gamma_tilde = - 2 / 3 * mu_tilde
#             else:
#                 mu = (2 * j - 1) * (j + 2) * (j - 1)**2 / (j * (j - 2) * (j + 1)**2)
#                 nu = -(j - 1)**3 * (j**2 - 4) / (j**3 * (j + 1) * (j - 3))
#                 mu_tilde = (2 * j - 1) * (j + 2) * (j - 1)**2 / (j * (j - 2) * (j + 1)**2) * 4 / (s**2 + s - 2)
#                 gamma_tilde = (((j - 1)**2 + j - 3)/(2 * j * (j - 1)) - 1) * mu_tilde



#             # Deterministic case
#             Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img - dt * mu_tilde * drift - dt * gamma_tilde * drift_initial

#         Y_j_2 = Y_j_1
#         Y_j_1 = Y_j

#     # Result of the 1st composition
#     img_intermediate1 = Y_j

#     noise_predictions.append(noise_initial)



#     '''
#     Second step in the composition method
#     '''
#     t_diff1 = gamma1 * (t - t_next)
#     t_diff1 = t_diff1.view(-1, 1, 1, 1)

#     dt = gamma2 * (t - t_next)
#     dt = th.ones(n, device=img.device) * dt
#     dt = dt.view(-1, 1, 1, 1)




#     Y_j_2 = img_intermediate1
#     Y_j_1 = img_intermediate1
#     Y_j = img_intermediate1

#     # Implementation of our Runge-Kutta-Legendre second order method
#     for j in range(1, s + 1):

#         # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
#         if j > 1:
#             if j == 2:
#                 fraction = 4 / (3 * (s**2 + s - 2))
#             else:
#                 fraction = ((j - 1)**2 + (j - 1) - 2) / (s**2 + s - 2)
#             t_intermediate = t_start - gamma1 * (t - t_next) - gamma2 * (t - t_next) * fraction * th.ones(n, device=img.device)

#             beta_0, beta_1 = betas[0], betas[-1]
#             beta_t = (beta_0 + t_intermediate * (beta_1 - beta_0)) * total_step

#             log_mean_coeff = (-0.25 * t_intermediate ** 2 * (beta_1 - beta_0) - 0.5 * t_intermediate * beta_0) * total_step
#             std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

#             # drift, diffusion -> f(x,t), g(t)
#             drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * Y_j_1, th.sqrt(beta_t)
#             '''
#             Approximate the noise using Taylor expansion
#             '''
#             if len(noise_predictions) <= 1:
#                 noise = noise_initial - (t_diff1 + fraction * dt) * noise_derivative
#             elif len(noise_predictions) <= 3:
#                 noise = noise_initial - (t_diff1 + fraction * dt) * noise_derivative + (t_diff1 + fraction * dt)**2 * noise_second_derivative
#             else:
#                 noise = noise_initial - (t_diff1 + fraction * dt) * noise_derivative + (t_diff1 + fraction * dt)**2 * noise_second_derivative \
#                 - (t_diff1 + fraction * dt)**3 * noise_third_derivative
#             score = -noise / std.view(-1, 1, 1, 1)  # score -> noise
#             drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5  # drift -> dx/dt

#         if j == 1:
#             mu_tilde = 4 / (3 * (s**2 + s - 2))
#             Y_j = Y_j_1 - dt * mu_tilde * drift_initial
#         else:
#             if j == 2:
#                 mu = (2 * j - 1) / j
#                 nu = -(j - 1) / j
#                 mu_tilde = mu * 4 / (s**2 + s - 2)
#                 gamma_tilde = -2 / 3 * mu_tilde
#             elif j == 3:
#                 mu = 3 * (2 * j - 1) / j * (j**2 + j - 2) / (2 * j * (j+1))
#                 nu = - 3 * (j - 1) / j * (j**2 + j - 2) / (2 * j * (j+1))
#                 mu_tilde = mu * 4 / (s**2 + s - 2)
#                 gamma_tilde = - 2 / 3 * mu_tilde
#             else:
#                 mu = (2 * j - 1) * (j + 2) * (j - 1)**2 / (j * (j - 2) * (j + 1)**2)
#                 nu = -(j - 1)**3 * (j**2 - 4) / (j**3 * (j + 1) * (j - 3))
#                 mu_tilde = (2 * j - 1) * (j + 2) * (j - 1)**2 / (j * (j - 2) * (j + 1)**2) * 4 / (s**2 + s - 2)
#                 gamma_tilde = (((j - 1)**2 + j - 3)/(2 * j * (j - 1)) - 1) * mu_tilde



#             # Deterministic case
#             Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img - dt * mu_tilde * drift - dt * gamma_tilde * drift_initial

#         Y_j_2 = Y_j_1
#         Y_j_1 = Y_j

#     img_intermediate2 = Y_j



#     '''
#     Third step in the composition method
#     '''
#     t_diff2 = gamma1 * (t - t_next) + gamma2 * (t - t_next)
#     t_diff2 = t_diff2.view(-1, 1, 1, 1)

#     dt = gamma3 * (t - t_next)
#     dt = th.ones(n, device=img.device) * dt
#     dt = dt.view(-1, 1, 1, 1)




#     Y_j_2 = img_intermediate2
#     Y_j_1 = img_intermediate2
#     Y_j = img_intermediate2

#     # Implementation of our Runge-Kutta-Legendre second order method
#     for j in range(1, s + 1):

#         # Calculate the corresponding \bar{alpha}_t and beta_t that aligns with the correct timestep
#         if j > 1:
#             if j == 2:
#                 fraction = 4 / (3 * (s**2 + s - 2))
#             else:
#                 fraction = ((j - 1)**2 + (j - 1) - 2) / (s**2 + s - 2)
#             t_intermediate = t_start - gamma1 * (t - t_next) - gamma2 * (t - t_next) - gamma3 * (t - t_next) * fraction * th.ones(n, device=img.device)

#             beta_0, beta_1 = betas[0], betas[-1]
#             beta_t = (beta_0 + t_intermediate * (beta_1 - beta_0)) * total_step

#             log_mean_coeff = (-0.25 * t_intermediate ** 2 * (beta_1 - beta_0) - 0.5 * t_intermediate * beta_0) * total_step
#             std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

#             # drift, diffusion -> f(x,t), g(t)
#             drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * Y_j_1, th.sqrt(beta_t)
#             '''
#             Approximate the noise using Taylor expansion
#             '''
#             if len(noise_predictions) <= 1:
#                 noise = noise_initial - (t_diff2 + fraction * dt) * noise_derivative
#             elif len(noise_predictions) <= 3:
#                 noise = noise_initial - (t_diff2 + fraction * dt) * noise_derivative + (t_diff2 + fraction * dt)**2 * noise_second_derivative
#             else:
#                 noise = noise_initial - (t_diff2 + fraction * dt) * noise_derivative + (t_diff2 + fraction * dt)**2 * noise_second_derivative \
#                 - (t_diff2 + fraction * dt)**3 * noise_third_derivative
#             score = -noise / std.view(-1, 1, 1, 1)  # score -> noise
#             drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5  # drift -> dx/dt

#         if j == 1:
#             mu_tilde = 4 / (3 * (s**2 + s - 2))
#             Y_j = Y_j_1 - dt * mu_tilde * drift_initial
#         else:
#             if j == 2:
#                 mu = (2 * j - 1) / j
#                 nu = -(j - 1) / j
#                 mu_tilde = mu * 4 / (s**2 + s - 2)
#                 gamma_tilde = -2 / 3 * mu_tilde
#             elif j == 3:
#                 mu = 3 * (2 * j - 1) / j * (j**2 + j - 2) / (2 * j * (j+1))
#                 nu = - 3 * (j - 1) / j * (j**2 + j - 2) / (2 * j * (j+1))
#                 mu_tilde = mu * 4 / (s**2 + s - 2)
#                 gamma_tilde = - 2 / 3 * mu_tilde
#             else:
#                 mu = (2 * j - 1) * (j + 2) * (j - 1)**2 / (j * (j - 2) * (j + 1)**2)
#                 nu = -(j - 1)**3 * (j**2 - 4) / (j**3 * (j + 1) * (j - 3))
#                 mu_tilde = (2 * j - 1) * (j + 2) * (j - 1)**2 / (j * (j - 2) * (j + 1)**2) * 4 / (s**2 + s - 2)
#                 gamma_tilde = (((j - 1)**2 + j - 3)/(2 * j * (j - 1)) - 1) * mu_tilde



#             # Deterministic case
#             Y_j = mu * Y_j_1 + nu * Y_j_2 + (1 - mu - nu) * img - dt * mu_tilde * drift - dt * gamma_tilde * drift_initial

#         Y_j_2 = Y_j_1
#         Y_j_1 = Y_j

#     img_next = Y_j



#     noise_predictions.append(noise_initial)

#     print(img_next[0])
#     return img_next