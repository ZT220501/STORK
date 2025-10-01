from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline, \
    UniPCMultistepScheduler, DEISMultistepScheduler, PNDMScheduler, DDIMScheduler, PixArtAlphaPipeline
from tqdm import tqdm
import torch
import os
import argparse
import json
import torch.multiprocessing as mp
from STORKScheduler import STORKScheduler


MODELS = [
    "PixArt-alpha/PixArt-XL-2-512x512"
]

SCHEDULERS = {
    "STORK-4-1st-noise": STORKScheduler,
    "STORK-4-2nd-noise": STORKScheduler,
    "DPM-Solver++": DPMSolverMultistepScheduler,
    "UniPC": UniPCMultistepScheduler,
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
    "DEIS": DEISMultistepScheduler,
}


def load_jobs(job_path, save_folder, overwrite=True):
    if os.path.exists(save_folder):
        generated_contents = os.listdir(save_folder)
        generated_keys = [x.split(".")[0] for x in generated_contents]
        if overwrite:
            print(f"Overwriting {len(generated_keys)} generated images.")
        else:
            print(f"Skipping {len(generated_keys)} generated images.")
    else:
        print('No saved images found. Generating all images.')
        generated_keys = []
    tasks = json.load(open(job_path, "r"))
    jobs = []
    for key in tasks.keys():
        if not overwrite:
            if key in generated_keys:
                continue
        jobs.append(
                (
                    tasks[key]["prompt"], 
                    key
                )
            )
    return jobs

def load_jobs_hpsv2(job_path):
    styles = ['anime', 'concept-art', 'paintings', 'photo']
    res = {}
    for sty in styles:
        with open(f"{job_path}/{sty}.json") as f:
            prompts = json.load(f)
        jobs = []
        for i, prompt in enumerate(prompts):
            jobs.append(
                (
                    prompt, 
                    i
                )
            )
        res[sty] = jobs
    return res


def divide_chunks(lst, n_chunks):
    length = len(lst)
    k, m = divmod(length, n_chunks)
    chunks = []
    start = 0
    for i in range(n_chunks):
        end = start + k + (1 if i < m else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks

def build_scheduler(method, kwargs):
    assert method in list(SCHEDULERS.keys()), f"Unknown scheduler: {method}"
    return SCHEDULERS[method](**kwargs)
        

def sample(rank, jobs, model_id, device, num_inference_steps, scheduler, precision, height=512, width=512,
           guidance_scale=3.5, batch_size=10, save_dir="output", seed=0, stopping_eps=1e-2):
    if precision == "bfloat16":
        torch_dtype = torch.bfloat16
    elif precision == "float16":
        torch_dtype = torch.float16
    elif precision == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {precision}")
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    
    # Load the model
    if model_id == "PixArt-alpha/PixArt-XL-2-512x512":
        pipeline = PixArtAlphaPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    else:
        pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)

    beta_start = pipeline.scheduler.config["beta_start"]
    beta_end = pipeline.scheduler.config["beta_end"]
    beta_schedule = pipeline.scheduler.config["beta_schedule"]
    # timestep_spacing = pipeline.scheduler.config["timestep_spacing"]
    # steps_offset = pipeline.scheduler.config["step_offset"]

    if scheduler == "DDIM":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    else:
        if scheduler == "DPM-Solver++":
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        elif scheduler == "PNDM":
            pipeline.scheduler = PNDMScheduler.from_config(pipeline.scheduler.config)
        elif scheduler == "UniPC":
            pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, prediction_type="epsilon")
        elif scheduler == "DEIS":
            pipeline.scheduler = DEISMultistepScheduler.from_config(pipeline.scheduler.config, prediction_type="epsilon", algorithm_type="deis")
        elif scheduler == "STORK-4-2nd-noise":
            intra_s = int(os.getenv("INTRA_S", 5))
            scheduler_config = dict(prediction_type='epsilon', solver_order=4, s=intra_s, derivative_order=2, beta_schedule=beta_schedule, beta_start=beta_start, beta_end=beta_end, stopping_eps=stopping_eps)
            pipeline.scheduler = build_scheduler(scheduler, scheduler_config)
        elif scheduler == "STORK-4-1st-noise":
            intra_s = int(os.getenv("INTRA_S", 5))
            scheduler_config = dict(prediction_type='epsilon', solver_order=4, s=intra_s, derivative_order=1, beta_schedule=beta_schedule, beta_start=beta_start, beta_end=beta_end, stopping_eps=stopping_eps)
            pipeline.scheduler = build_scheduler(scheduler, scheduler_config)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
    pipeline.to(device)
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Enable xformers memory efficient attention
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.set_progress_bar_config(disable=True)

    with torch.inference_mode():
        total_jobs = len(jobs)
        total_batches = total_jobs // batch_size
        if total_jobs % batch_size != 0:
            total_batches += 1
        for start_idx in tqdm(range(0, total_jobs, batch_size), total=total_batches, unit="Batch", desc=f"Proc-{rank} sampling", position=rank):
            end_idx = min(start_idx + batch_size, total_jobs)
            prompts = [jobs[idx][0] for idx in range(start_idx, end_idx)]
            im_ids = [jobs[idx][1] for idx in range(start_idx, end_idx)]
            if len(prompts) == 1:
                prompts = prompts[-1]
            images = pipeline(
                prompt=prompts,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=generator,
                guidance_scale=guidance_scale
            )["images"]
            for i, image in enumerate(images):
                image.save(f"{save_dir}/{im_ids[i]}.jpg")

   
def multiprocess_wrapper(num_proc, jobs, params):
    num_available_gpus = torch.cuda.device_count()
    assert num_proc <= num_available_gpus, f"Number of processes ({num_proc}) exceeds available GPUs ({num_available_gpus})."
    job_assignments = divide_chunks(jobs, num_proc)
    mp.set_start_method("spawn", force=True)  # safe for CUDA or cross-platform
    processes = []
    for rank in range(num_proc):
        jobs = job_assignments[rank]
        device = f"cuda:{rank}"
        params["jobs"] = jobs
        params["device"] = device
        params["rank"] = rank
        p = mp.Process(target=sample, kwargs=params)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='huggingface model id', choices=MODELS, default=MODELS[-1])
    parser.add_argument('--num_samples', help='number of samples used to benchmark', type=int, default=None)
    parser.add_argument('--num_inference_steps', help='how many steps to use for inference. Not necessarily NFE', type=int, default=19)
    parser.add_argument('--scheduler', help='scheduler to use', choices=list(SCHEDULERS.keys()), default="flow_euler")
    parser.add_argument('--batch_size', help='batch size to use', type=int, default=10)
    parser.add_argument('--num_proc', help='number of processes to use. GPUs will be assigned sequentially.', type=int, default=1)
    parser.add_argument('--save_dir', help='directory to save images', default="output")
    parser.add_argument('--seed', help='seed to use for random number generation', type=int, default=0)
    parser.add_argument('--precision', help='datatype to use', default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument('--overwrite', help='whether to overwrite', action='store_true', default=False)
    parser.add_argument('--image_size', help='image size to use', default=512, type=int)
    parser.add_argument('--cfg_scale', help='classifier-free guidance scale', default=3.5, type=float)
    parser.add_argument('--dataset', help='dataset to use', default="coco-30k_512", type=str)
    parser.add_argument('--stopping_eps', help='stopping epsilon', default=1e-2, type=float)
    args = parser.parse_args()
    # Parse the arguments
    print(args)
    model_id = args.model_id
    num_samples = args.num_samples
    num_inference_steps = args.num_inference_steps
    scheduler = args.scheduler
    precision = args.precision
    batch_size = args.batch_size
    save_dir = args.save_dir
    random_seed = args.seed
    img_size = args.image_size
    cfg = args.cfg_scale
    dataset = args.dataset
    stopping_eps = args.stopping_eps
    if dataset == "hpsv2":
        prepared_jobs = load_jobs_hpsv2(
            job_path=f"/home/zhengtan/datasets/{dataset}",
        )
    else:
        prepared_jobs = load_jobs(
            job_path=f"/home/zhengtan/datasets/{dataset}.json",
            save_folder=save_dir,
            overwrite=args.overwrite
        )
    if num_samples == -1:
        print("No limit on number of samples. Generating all samples.")
        pass
    else: 
        prepared_jobs = prepared_jobs[:num_samples]

    if dataset == "hpsv2":
        total_jobs = 0
        for style in ["anime", "concept-art", "paintings", "photo"]:
            total_jobs += len(prepared_jobs[style])
        print(f"Loaded {total_jobs} jobs.")
    else:
        print(f"Loaded {len(prepared_jobs)} jobs.")

    if dataset == "hpsv2":
        for style in ["anime", "concept-art", "paintings", "photo"]:
            save_dir_style = os.path.join(save_dir, style)
            multiprocess_wrapper(
                num_proc=args.num_proc,
                jobs=prepared_jobs[style],
                params={
                    "model_id": model_id,
                    "num_inference_steps": num_inference_steps,
                    "scheduler": scheduler,
                    "precision": precision,
                    "height": img_size,
                    "width": img_size,
                    "guidance_scale": cfg,
                    "batch_size": batch_size,
                    "save_dir": save_dir_style,
                    "seed": random_seed,
                    "stopping_eps": stopping_eps
                }
            )
    else:
        multiprocess_wrapper(
            num_proc=args.num_proc,
            jobs=prepared_jobs,
            params={
                "model_id": model_id,
                "num_inference_steps": num_inference_steps,
                "scheduler": scheduler,
                "precision": precision,
                "height": img_size,
                "width": img_size,
                "guidance_scale": cfg,
                "batch_size": batch_size,
                "save_dir": save_dir,
                "seed": random_seed,
                "stopping_eps": stopping_eps
            }
        )
    