# We borrowed the code from the PNDM codebase.

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

import argparse
import yaml
import os
import numpy as np
import torch as th
from runner.schedule import Schedule
from runner.runner import Runner


def args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=str, default='sample',
                        help="Choose the mode of runner")
    parser.add_argument("--config", type=str, default='ddim_cifar10.yml',
                        help="Choose the config file")
    parser.add_argument("--model", type=str, default='DDIM',
                        help="Choose the model's structure (DDIM, iDDPM, PF)")
    parser.add_argument("--method", type=str, default='F-PNDM',
                        help="Choose the numerical methods (DDIM, FON, S-PNDM, F-PNDM, PF)")
    parser.add_argument("--sample_speed", type=int, default=50,
                        help="Control the total generation step")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Choose the device to use")
    parser.add_argument("--image_path", type=str, default='temp/sample',
                        help="Choose the path to save images")
    parser.add_argument("--model_path", type=str, default='temp/models/ddim/ema_cifar10.ckpt',
                        help="Choose the path of model")
    parser.add_argument("--restart", action="store_true",
                        help="Restart a previous training process")
    parser.add_argument("--train_path", type=str, default='temp/train',
                        help="Choose the path to save training status")
    args = parser.parse_args()
    work_dir = os.getcwd()
    with open(f'{work_dir}/config/{args.config}', 'r') as f:
        config = yaml.safe_load(f)
    return args, config


if __name__ == "__main__":
    args, config = args_and_config()
    seed = config['Sample']['seed']
    # Seed randomness for reproducibility
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    generator = th.Generator('cpu').manual_seed(seed)
    assert not config['Sample']['mpi4py'], "MPI4PY is not supported in our codebase"
    device = th.device(args.device)
    schedule = Schedule(args, config['Schedule'])
    if config['Model']['struc'] == 'DDIM':
        from model.ddim import Model
        model = Model(args, config['Model']).to(device)
    else:
        model = None
        raise NotImplementedError("Model structure not benchmarked yet")
    runner = Runner(args, config, schedule, model)
    if args.runner == 'train':
        runner.train()
    elif args.runner == 'sample':
        runner.sample_fid(generator)

