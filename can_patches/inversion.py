from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file, save_file

from .core import MODEL_ID, integer
from .generation import load_pipeline, validate_pipeline
from .masks import source_image


def paired_schedulers(config, steps: int, device):
    from diffusers import DDIMInverseScheduler, DDIMScheduler
    integer(steps, "steps")
    if not 2 <= steps <= 1000:
        raise ValueError("Inversion steps must be between 2 and 1000")
    forward = DDIMScheduler.from_config(config, timestep_spacing="leading", clip_sample=False)
    inverse = DDIMInverseScheduler.from_config(forward.config)
    forward.set_timesteps(steps, device=device)
    inverse.set_timesteps(steps, device=device)
    if int(forward.timesteps.max()) >= forward.config.num_train_timesteps or int(forward.timesteps.min()) < 0:
        raise ValueError("Timestep offset and step count produce an out-of-range noise level")
    if not torch.equal(forward.timesteps, inverse.timesteps.flip(0)):
        raise ValueError("Forward and inverse timestep grids disagree")
    return forward, inverse


def encode_image(pipe, image: Image.Image, device, dtype) -> torch.Tensor:
    array = np.array(image, dtype=np.float32) / 127.5 - 1
    pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    return pipe.vae.encode(pixels).latent_dist.mean * pipe.vae.config.scaling_factor


def decode_image(pipe, latent: torch.Tensor) -> Image.Image:
    decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def invert_trace(pipe, image_latent, source_prompt, steps=50):
    device = image_latent.device
    forward, inverse = paired_schedulers(pipe.scheduler.config, steps, device)
    cond = pipe.encode_prompt(prompt=source_prompt, device=device,
                              num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
    z = image_latent.clone()
    states = [z.detach().cpu().clone()]
    with torch.inference_mode():
        for t in inverse.timesteps:
            pred = pipe.unet(inverse.scale_model_input(z, t), t, encoder_hidden_states=cond).sample
            z = inverse.step(pred, t, z).prev_sample
            states.append(z.detach().cpu().clone())
    return torch.stack(states), forward


def invert_image(*, image, source_prompt: str, output, model=MODEL_ID, device="cuda:0",
                 steps=50, local_files_only=False, pipe=None) -> dict:
    if not isinstance(source_prompt, str) or not source_prompt.strip():
        raise ValueError("Provide a nonempty source prompt describing the original image")
    output = Path(output)
    if output.exists():
        raise ValueError("Inversion output directory already exists")
    rgb = source_image(image)
    if pipe is None:
        pipe = load_pipeline(model, device, local_files_only=local_files_only)
    validate_pipeline(pipe)
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    with torch.inference_mode():
        z0 = encode_image(pipe, rgb, device, dtype)
        trajectory, scheduler = invert_trace(pipe, z0, source_prompt, steps)
        z = trajectory[-1].to(device=device, dtype=dtype)
        cond = pipe.encode_prompt(prompt=source_prompt, device=torch.device(device),
                                  num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
        for t in scheduler.timesteps:
            pred = pipe.unet(scheduler.scale_model_input(z, t), t, encoder_hidden_states=cond).sample
            z = scheduler.step(pred, t, z, eta=0.0).prev_sample
        reconstruction = decode_image(pipe, z)
    metadata = {"schema_version": 1, "model": model, "source_prompt": source_prompt,
                "steps": steps, "scheduler_config": dict(scheduler.config),
                "forward_timesteps": scheduler.timesteps.cpu().tolist(),
                "trajectory_order": "clean_then_ascending_inverse_levels",
                "shape": list(trajectory.shape), "torch_version": str(torch.__version__),
                "device": device, "dtype": str(dtype),
                "method": "source_conditioned_ddim_inversion",
                "image_size": [768, 768]}
    output.mkdir(parents=True)
    save_file({"trajectory": trajectory.contiguous()}, str(output/"trajectory.safetensors"))
    rgb.save(output/"source.png")
    reconstruction.save(output/"reconstruction.png")
    (output/"inversion.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def load_inversion(directory):
    directory = Path(directory)
    metadata = json.loads((directory/"inversion.json").read_text())
    steps = integer(metadata.get("steps"), "inversion steps")
    if (metadata.get("schema_version") != 1 or not 2 <= steps <= 1000
            or metadata.get("trajectory_order") != "clean_then_ascending_inverse_levels"):
        raise ValueError("Unsupported inversion artifact")
    tensors = load_file(str(directory/"trajectory.safetensors"), device="cpu")
    if set(tensors) != {"trajectory"}:
        raise ValueError("Inversion archive must contain only a trajectory tensor")
    trace = tensors["trajectory"]
    if (tuple(trace.shape) != (steps+1, 1, 4, 96, 96) or not trace.is_floating_point()
            or not torch.isfinite(trace).all()):
        raise ValueError("Invalid inversion trajectory shape or values")
    forward, _ = paired_schedulers(metadata["scheduler_config"], steps, "cpu")
    if forward.timesteps.tolist() != metadata.get("forward_timesteps"):
        raise ValueError("Stored scheduler grid does not match the inversion trajectory")
    return trace, metadata, source_image(directory/"source.png")
