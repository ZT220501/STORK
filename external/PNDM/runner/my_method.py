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
    n = img.shape[0]
    t = t[0].item()
    t_next = t_next[0].item()
    t_start = th.ones(n, device=img.device) * t
    t_end = th.ones(n, device=img.device) * t_next
    dt = t - t_next
    dt = th.ones(n, device=img.device) * dt
    dt = dt.view(-1, 1, 1, 1)
    beta_0, beta_1 = betas[0], betas[-1]
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step
    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))
    # FIXME this is also problematic, why is it -1????
    if t_next == -1:
        # Tweedie's trick
        # FIXME why is this better
        #noise_last = model(img, t_start)
        noise_last = model(img, t_start * (total_step - 1))
        img_next = img - std.view(-1, 1, 1, 1) * noise_last
        return img_next


    # drift, diffusion -> f(x,t), g(t)
    drift_initial, diffusion_initial = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    #FIRST NOISE
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
        # FIRST IMAGE RETURN
        img_intermediate_prediction = img - 0.5 * dt * drift_initial
        # SECOND RUN STARTED
        t_intermediate = t_start - 0.5 * (t_start - t_end)
        beta_t = (beta_0 + t_intermediate * (beta_1 - beta_0)) * total_step
        log_mean_coeff = (-0.25 * t_intermediate ** 2 * (beta_1 - beta_0) - 0.5 * t_intermediate * beta_0) * total_step
        std = th.sqrt(1. - th.exp(2. * log_mean_coeff))
        drift_next, diffusion_next = -0.5 * beta_t.view(-1, 1, 1, 1) * img_intermediate_prediction, th.sqrt(beta_t)
        #SECOND NOISE
        noise_prediction_intermediate = model(img_intermediate_prediction, t_intermediate * (total_step - 1))
        score = -noise_prediction_intermediate / std.view(-1, 1, 1, 1)  # score -> noise
        drift_next = drift_next - diffusion_next.view(-1, 1, 1, 1) ** 2 * score * 0.5 # drift -> dx/dt
        #SECOND IMAGE RETURN
        img_next_prediction = img_intermediate_prediction - 0.75 * dt * drift_next + 0.25 * dt * drift_initial
        # THIRD NOISE
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
    return img_next




































