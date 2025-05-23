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