from diffusers import FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler, DiffusionPipeline, FluxPipeline, FlowMatchHeunDiscreteScheduler, \
    Lumina2Pipeline, StableDiffusionXLPipeline, UniPCMultistepScheduler
from tqdm import tqdm
import torch
import os
import argparse
import os
import json
import torch.multiprocessing as mp
from STORKScheduler import STORKScheduler


MODELS = [
    "stabilityai/stable-diffusion-3.5-large",
    "stabilityai/stable-diffusion-3.5-medium",
    'black-forest-labs/FLUX.1-dev',
    'Alpha-VLLM/Lumina-Image-2.0',
]

SCHEDULERS = {
    "flow_euler":FlowMatchEulerDiscreteScheduler,
    "flow_dpm-solver": DPMSolverMultistepScheduler,
    "flow_heun": FlowMatchHeunDiscreteScheduler,
    "flow_STORK-4-2nd": STORKScheduler,
    "flow_STORK-4-1st": STORKScheduler,
    "flow_unipc": UniPCMultistepScheduler,
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
           guidance_scale=3.5, batch_size=10, save_dir="output", seed=0):
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
    
    if model_id == 'black-forest-labs/FLUX.1-dev':
        pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    elif model_id == 'Alpha-VLLM/Lumina-Image-2.0':
        pipeline = Lumina2Pipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    elif model_id == 'stabilityai/stable-diffusion-xl-base-1.0':
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    else:
        pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    flow_shift = pipeline.scheduler.config["shift"]
    if scheduler == "flow_euler":
        # This is the default. No need to set it explicitly.
        pass   
    else:
        if scheduler == "flow_dpm-solver":
            scheduler_config = dict(
                use_flow_sigmas=True, solver_order=2, prediction_type="flow_prediction", flow_shift=flow_shift,
            )
        elif scheduler == "flow_heun":
            scheduler_config = dict(
                shift=flow_shift
            )
        elif scheduler == "flow_STORK-4-2nd":
            intra_s = int(os.getenv("INTRA_S", 5))
            scheduler_config = dict(shift=flow_shift, prediction_type='flow_prediction', solver_order=4, s=intra_s, derivative_order=2)
        elif scheduler == "flow_STORK-4-1st":
            intra_s = int(os.getenv("INTRA_S", 5))
            scheduler_config = dict(shift=flow_shift, prediction_type='flow_prediction', solver_order=4, s=intra_s, derivative_order=1)
        elif scheduler == "flow_unipc":
            scheduler_config = dict(flow_shift=flow_shift, prediction_type='flow_prediction', use_flow_sigmas=True)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
        # Special handling for FLUX.1-dev
        if model_id == 'black-forest-labs/FLUX.1-dev' and scheduler == "flow_dpm-solver":
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_flow_sigmas=True, prediction_type="flow_prediction", flow_shift=flow_shift)
        elif model_id == 'black-forest-labs/FLUX.1-dev' and scheduler == "flow_unipc":
            pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, use_flow_sigmas=True, prediction_type="flow_prediction", flow_shift=flow_shift)
        elif model_id == 'Alpha-VLLM/Lumina-Image-2.0' and scheduler == "flow_dpm-solver":
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_flow_sigmas=True, prediction_type="flow_prediction", flow_shift=flow_shift)
        elif model_id == 'Alpha-VLLM/Lumina-Image-2.0' and scheduler == "flow_unipc":
            pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, use_flow_sigmas=True, prediction_type="flow_prediction", flow_shift=flow_shift)
        else:
            pipeline.scheduler = build_scheduler(scheduler, scheduler_config)
    pipeline.to(device)
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Enable xformers memory efficient attention if available
    if not model_id == 'black-forest-labs/FLUX.1-dev' and not model_id == 'Alpha-VLLM/Lumina-Image-2.0':
        pipeline.enable_xformers_memory_efficient_attention() # This is not supported by FLUX.1-dev. Raise issue maybe.
    pipeline.set_progress_bar_config(disable=True)
    if model_id == 'black-forest-labs/FLUX.1-dev':
        batch_size = 1  # FLUX.1-dev does not support batch inference, so we set batch size to 1.
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
            if model_id == 'Alpha-VLLM/Lumina-Image-2.0':
                images = pipeline(
                    prompt=prompts,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    cfg_trunc_ratio=0.25,
                    cfg_normalization=True,
                    generator=generator
                )["images"]
            else:
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
    parser.add_argument('--num_samples', help='number of samples used to benchmark', type=int, default=-1)
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
                    "seed": random_seed
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
                "seed": random_seed
            }
        )
    