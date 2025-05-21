from diffusers import DiffusionPipeline, PNDMScheduler, PNDMScheduler, \
    DPMSolverMultistepScheduler, UniPCMultistepScheduler, DPMSolverSinglestepScheduler, DEISMultistepScheduler
#from runner.rock_diffusers_old import ROCKScheduler
from runner.rock_diffuser_sorted import ROCKScheduler_sorted
from model.ddim import Model
import yaml, os, torch, argparse
from torchvision.transforms.functional import to_pil_image





def build_scheduler(name, **kwargs):
    if name == "DPM-Solver++":
        scheduler = DPMSolverMultistepScheduler(**kwargs)
    elif name == "DPM-Solver":
        scheduler = DPMSolverMultistepScheduler(**kwargs)
    elif name == "PNDM":
        scheduler = PNDMScheduler(**kwargs)
    elif name == "UniPC":
        scheduler = UniPCMultistepScheduler(**kwargs)
    elif name == "PNDM":
        scheduler = PNDMScheduler(**kwargs)
    elif name == "DPM++Single":
        scheduler = DPMSolverSinglestepScheduler(**kwargs)
    elif name == "ROCK4":
        scheduler = ROCKScheduler_sorted(**kwargs)
    elif name == "DEIS":
        scheduler = DEISMultistepScheduler(**kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
    return scheduler




def args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=str, default='sample',
                        help="Choose the mode of runner")
    parser.add_argument("--config", type=str, default='ddim_cifar10.yml',
                        help="Choose the config file")
    parser.add_argument("--model", type=str, default='DDIM',
                        help="Choose the model's structure (DDIM, iDDPM, PF)")
    
    parser.add_argument("--method", type=str, default='DPM-Solver++',
                        help="Choose the scheduler (DPM-Solver++, DPM-Solver, UniPC)")

    parser.add_argument("--sample_speed", type=int, default=50,
                        help="Control the total generation step")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Choose the device to use")
    parser.add_argument("--image_path", type=str, default='temp/sample',
                        help="Choose the path to save images")
    parser.add_argument("--model_path", type=str, default='temp/models/ddim/ema_cifar10.ckpt',
                        help="Choose the path of model")
    


    parser.add_argument("--eps", type=float, default=None,
                        help="Stopping epsilon for ROCK4, enforced if the method is ROCK4")    
    parser.add_argument("--s", type=int, default=None,
                        help="INTRA-S for ROCK4, enforced if the method is ROCK4") 
    



    parser.add_argument("--restart", action="store_true",
                        help="Restart a previous training process")
    parser.add_argument("--train_path", type=str, default='temp/train',
                        help="Choose the path to save training status")
    args = parser.parse_args()

    work_dir = os.getcwd()
    with open(f'{work_dir}/config/{args.config}', 'r') as f:
        config = yaml.safe_load(f)

    if args.method == "ROCK4":
        assert args.eps is not None, "Stopping epsilon must be specified for ROCK4"
        assert args.s is not None, "INTRA-S must be specified for ROCK4"

    return args, config



class MyCustomPipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, batch_size=4, num_inference_steps=50, height=32, width=32):
        device = next(model.parameters()).device
        noise = torch.randn((batch_size, self.model.in_channels, height, width)).to(device)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        image = noise
        for t in self.scheduler.timesteps:
            t = t.unsqueeze(0)
            model_output = self.model(image, t)
            image = self.scheduler.step(model_output, t.item(), image).prev_sample
        image = (image.clamp(-1, 1) + 1) / 2  # Normalize to [0,1]
        return {"images": image}

from tqdm import tqdm

if __name__ == "__main__":
    args, config = args_and_config()
    device = torch.device(args.device)
    # Load the ddim model
    model = Model(args, config["Model"])
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    
    # Adding Hook
    # forward_count = dict(count=0)
    # def NFE_hook(model, input, output):
    #     forward_count['count'] += 1
    # Register the hook to the model
    # hook_handle = model.register_forward_hook(NFE_hook)



    # Make scheuler configs
    if args.method == "DPM-Solver++":
        scheduler_config = dict(
            beta_schedule = "linear", solver_order = 3, algorithm_type = "dpmsolver++", use_lu_lambdas = True,
        )
        scheduler = build_scheduler(args.method, **scheduler_config)
    elif args.method == "DPM-Solver":
        scheduler_config = dict(
            beta_schedule = "linear", solver_order = 3, algorithm_type = "dpmsolver", use_lu_lambdas = True,
        )
        scheduler = build_scheduler(args.method, **scheduler_config)
    elif args.method == "UniPC":
        # Recommended solver_order is 3 for UniPC unconditional sampling
        scheduler_config = dict(
            beta_schedule = "linear", solver_order = 3
        )
        scheduler = build_scheduler(args.method, **scheduler_config)
    elif args.method == "PNDM":
        scheduler_config = dict(
            beta_schedule = "linear"
        )
        scheduler = build_scheduler(args.method, **scheduler_config)
    elif args.method == "DPM++Single":
        scheduler_config = dict(
            beta_schedule = "linear", solver_order = 3, algorithm_type = "dpmsolver++",
        )
        scheduler = build_scheduler(args.method, **scheduler_config)
    elif args.method == "ROCK4":
        scheduler_config = dict(
            stopping_eps=args.eps, s=args.s,
        )
        print(scheduler_config)
        #exit(0)
        scheduler = build_scheduler(args.method, **scheduler_config)
    elif args.method == "DEIS":
        scheduler_config = dict(
            beta_schedule = "linear", solver_order = 3, algorithm_type = "deis",
        )
        scheduler = build_scheduler(args.method, **scheduler_config)
    else:
        raise ValueError(f"Unknown scheduler: {args.method}")
    pipeline = MyCustomPipeline(model, scheduler).to(device)
    # Generate images

    pipeline.enable_xformers_memory_efficient_attention()
    # Create output directory if it doesn't exist
    save_dir = args.image_path
    batch_size = config["Sample"]["batch_size"]
    num_samples = config["Sample"]["total_num"]
    num_inference_steps = args.sample_speed
    os.makedirs(save_dir, exist_ok=True)
    h = w = config["Model"]["image_size"]
    with torch.inference_mode():
        for start_idx in tqdm(range(0, num_samples, batch_size), unit="Batch"):
            end_idx = min(start_idx + batch_size, num_samples)
            actual_size = end_idx - start_idx
            #print(f"Processing samples {start_idx} to {end_idx} (b={actual_size})...")
            images = pipeline(
                batch_size=actual_size, 
                num_inference_steps=num_inference_steps,
                height=h, width=w,
            )["images"]
            for i, image in enumerate(images):
                pil_image = to_pil_image(image.cpu())
                pil_image.save(f"{save_dir}/{start_idx + i}.png")
    # Unregister the hook after training/sampling. Not really needed, but good practice.
    # hook_handle.remove()
    # print("Total NFE is " + str(forward_count['count']))