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

import os
import sys
import time
import torch as th
import numpy as np
import torch.optim as optimi
import torch.utils.data as data
import torchvision.utils as tvu
import torch.utils.tensorboard as tb
from scipy import integrate
# from torchdiffeq import odeint
from tqdm.auto import tqdm

from dataset import get_dataset, inverse_data_transform
from model.ema import EMAHelper


def get_optim(params, config):
    if config['optimizer'] == 'adam':
        optim = optimi.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'],
                            betas=(config['beta1'], 0.999), amsgrad=config['amsgrad'],
                            eps=config['eps'])
    elif config['optimizer'] == 'sgd':
        optim = optimi.SGD(params, lr=config['lr'], momentum=0.9)
    else:
        optim = None

    return optim


class Runner(object):
    def __init__(self, args, config, schedule, model):
        self.args = args
        self.config = config
        self.diffusion_step = config['Schedule']['diffusion_step']
        self.sample_speed = args.sample_speed
        self.device = th.device(args.device)

        self.schedule = schedule
        self.model = model

    def train(self):
        schedule = self.schedule
        model = self.model
        model = th.nn.DataParallel(model)

        optim = get_optim(model.parameters(), self.config['Optim'])

        config = self.config['Dataset']
        dataset, test_dataset = get_dataset(self.args, config)
        train_loader = data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                                       num_workers=config['num_workers'])

        config = self.config['Train']
        if config['ema']:
            ema = EMAHelper(mu=config['ema_rate'])
            ema.register(model)
        else:
            ema = None

        tb_logger = tb.SummaryWriter(f'temp/tensorboard/{time.strftime("%m%d-%H%M")}')
        epoch, step = 0, 0

        if self.args.restart:
            train_state = th.load(os.path.join(self.args.train_path, 'train.ckpt'), map_location=self.device)
            model.load_state_dict(train_state[0])
            optim.load_state_dict(train_state[1])
            epoch, step = train_state[2:4]
            if ema is not None:
                ema_state = th.load(os.path.join(self.args.train_path, 'ema.ckpt'), map_location=self.device)
                ema.load_state_dict(ema_state)

        for epoch in range(epoch, config['epoch']):
            for i, (img, y) in enumerate(train_loader):
                n = img.shape[0]
                model.train()
                step += 1
                t = th.randint(low=0, high=self.diffusion_step, size=(n // 2 + 1,))
                t = th.cat([t, self.diffusion_step - t - 1], dim=0)[:n].to(self.device)
                img = img.to(self.device) * 2.0 - 1.0

                img_n, noise = schedule.diffusion(img, t)
                noise_p = model(img_n, t)

                if config['loss_type'] == 'linear':
                    loss = (noise_p - noise).abs().sum(dim=(1, 2, 3)).mean(dim=0)
                elif config['loss_type'] == 'square':
                    loss = (noise_p - noise).square().sum(dim=(1, 2, 3)).mean(dim=0)
                else:
                    loss = None

                optim.zero_grad()
                loss.backward()
                try:
                    th.nn.utils.clip_grad_norm_(model.parameters(), self.config['Optim']['grad_clip'])
                except Exception:
                    pass
                optim.step()

                if ema is not None:
                    ema.update(model)

                if step % 10 == 0:
                    tb_logger.add_scalar('loss', loss, global_step=step)
                if step % 50 == 0:
                    print(step, loss.item())
                if step % 500 == 0:
                    config = self.config['Dataset']
                    model.eval()
                    skip = self.diffusion_step // self.sample_speed
                    seq = range(0, self.diffusion_step, skip)
                    noise = th.randn(16, config['channels'], config['image_size'],
                                     config['image_size'], device=self.device)
                    img = self.sample_image(noise, seq, model)
                    img = th.clamp(img * 0.5 + 0.5, 0.0, 1.0)
                    tb_logger.add_images('sample', img, global_step=step)
                    config = self.config['Train']
                    model.train()

                if step % 5000 == 0:
                    train_state = [model.state_dict(), optim.state_dict(), epoch, step]
                    th.save(train_state, os.path.join(self.args.train_path, 'train.ckpt'))
                    if ema is not None:
                        th.save(ema.state_dict(), os.path.join(self.args.train_path, 'ema.ckpt'))

    def sample_fid(self):
        config = self.config['Sample']
        mpi_rank = 0
        # if config['mpi4py']:
        #     from mpi4py import MPI
        #     comm = MPI.COMM_WORLD
        #     mpi_rank = comm.Get_rank()

        model = self.model
        device = self.device
        pflow = True if self.args.method == 'PF' else False

        model.load_state_dict(th.load(self.args.model_path, map_location=device, weights_only=True), strict=True)
        model.eval()

        n = config['batch_size']
        total_num = config['total_num']

        skip = self.diffusion_step // self.sample_speed
        
        #print("Total diffusion step is " + str(self.diffusion_step))
        #print("Skip is " + str(skip))

        '''
        This seems to be where the problem happens. Before people are only using 'leading' way; however as pointed out by
        https://arxiv.org/pdf/2305.08891, the 'trailing' way is actually better.
        TODO: Try to change different time stepping settings; try case one and case three below.
        '''
        ####################
        # This is case one #
        ####################
        # seq = range(0, self.diffusion_step, skip)                 # This is the original setting


        ###########################################
        # This is case two, the pf time stepping #
        ###########################################
        # TODO: Test stopping_eps = 1e-3, 1e-4, and 1e-5
        stopping_eps = float(os.getenv("EPS", "1e-3"))          #1e-4
        #print("stopping_eps is " + str(stopping_eps))
        #print("Type of self.sample_speed is " + str(type(self.sample_speed)))
        seq = np.linspace(0, 1, self.sample_speed+1)
        seq[0] = stopping_eps
        seq = seq[:-1]
        #print("Case 4 seq is " + str(seq))



        ###########################################
        # This is case three, the pf time stepping #
        ###########################################
        # # TODO: Test stopping_eps = 1e-3, 1e-4, and 1e-5; make this adaptive time stepping correct
        # stopping_eps = float(os.getenv("EPS", "1e-4"))          #1e-3, 1e-5
        # initial_time_step = 1 / (self.sample_speed * 4)
        # seq = np.linspace(0, 1 - 1 / self.sample_speed - 2 * initial_time_step, self.sample_speed-1)
        # seq[0] = stopping_eps
        # # seq = seq[:-1]
        # seq = np.append(seq, 1 - 1 / self.sample_speed - initial_time_step)
        # seq = np.append(seq, 1 - 1 / self.sample_speed)
        # print("Case 4 seq is " + str(seq))


        image_num = 0

        config = self.config['Dataset']
        if mpi_rank == 0:
            # my_iter = tqdm(range(total_num // n + 1), ncols=120)
            my_iter = tqdm(range(total_num // n), ncols=120)
        else:
            my_iter = range(total_num // n + 1)

        for _ in my_iter:
            noise = th.randn(n, config['channels'], config['image_size'],
                             config['image_size'], device=self.device)

            if self.args.method == "ROCK4_ADAPTIVE_TIMESTEP":
                # TODO: Try two different starting times for the adaptive time stepping\
                # Shall we start at self.diffusion_step or self.diffusion_step - skip?
                img = self.sample_image_nonuniform(noise, self.diffusion_step, model, skip)
            else:
                img = self.sample_image(noise, seq, model, pflow)

            img = inverse_data_transform(config, img)
            for i in range(img.shape[0]):
                if image_num+i > total_num:
                    break
                tvu.save_image(img[i], os.path.join(self.args.image_path, f"{self.args.method}-{image_num+i}.png"))


            image_num += n

    def sample_image(self, noise, seq, model, pflow=False):
        with th.no_grad():
            if pflow:
                shape = noise.shape
                device = self.device
                tol = 1e-5 if self.sample_speed > 1 else self.sample_speed

                def drift_func(t, x):
                    x = th.from_numpy(x.reshape(shape)).to(device).type(th.float32)
                    drift = self.schedule.denoising(x, None, t, model, pflow=pflow)
                    drift = drift.cpu().numpy().reshape((-1,))
                    return drift

                solution = integrate.solve_ivp(drift_func, (1, 1e-3), noise.cpu().numpy().reshape((-1,)),
                                               rtol=tol, atol=tol, method='RK45')
                img = th.tensor(solution.y[:, -1]).reshape(shape).type(th.float32)

            else:
                imgs = [noise]


                # print("seq is " + str(seq))

                # This is the original setting
                seq_next = [-1] + list(seq[:-1])

                start = True
                n = noise.shape[0]


                iter = 0
                for i, j in zip(reversed(seq), reversed(seq_next)):

                    iter += 1
                    #print("This is iteration " + str(iter))
                    t = (th.ones(n) * i).to(self.device)
                    t_next = (th.ones(n) * j).to(self.device)

                    img_t = imgs[-1].to(self.device)
                    img_next = self.schedule.denoising(img_t, t_next, t, model, first_step=start, pflow=pflow)
                    start = False

                    imgs.append(img_next.to('cpu'))

                img = imgs[-1]


            return img
        


    '''
    Sampling with non-uniform timestep; currently exclusively used for SRKL method
    '''
    def sample_image_nonuniform(self, noise, diffusion_step, model, dt):        
        with th.no_grad():
            imgs = [noise]

            start = True
            n = noise.shape[0]

            t_current = diffusion_step * th.ones(n)
            skip = self.diffusion_step // (self.sample_speed - 2)
            iter = 0
            while t_current[0] > 100:

                iter += 1
                #print("This is iteration " + str(iter))
                #print("t_current is " + str(t_current[0]))

                # The first two iterations are small time steppings; 
                # starting from the third iteration, we use large time steppings
                if iter <= 2:
                    dt = th.ones(n) * (skip / 8)

                t = (th.ones(n) * t_current).to(self.device)
                t_next = (th.ones(n) * (t_current - dt)).to(self.device)

                img_t = imgs[-1].to(self.device)
                img_next = self.schedule.denoising(img_t, t_next, t, model, start, pflow=False)
                start = False
                imgs.append(img_next)


                t_current = t_current - dt
                # dt = min(skip, t_current[0]) * th.ones(n)
                dt = min(dt[0] * 2, skip) * th.ones(n)

            img = imgs[-1]

            return img
