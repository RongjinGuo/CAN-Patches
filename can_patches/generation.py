from __future__ import annotations

from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

from .core import MODEL_ID, Placement, compose_noise, initial_noise, integer
from .database import load_database


def load_pipeline(model: str = MODEL_ID, device: str = "cuda:0", *, local_files_only: bool = False):
    from diffusers import DPMSolverSinglestepScheduler, StableDiffusionPipeline

    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Supported inference devices are CPU and CUDA")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu to load exported patches on CPU")
    dtype = torch.float16 if target.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
        local_files_only=local_files_only,
    )
    pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
    return pipe.to(target)


def validate_pipeline(pipe) -> None:
    if pipe.unet.config.in_channels != 4 or pipe.vae_scale_factor != 8:
        raise ValueError("This implementation requires a 4-channel SD latent and VAE factor 8")
    if pipe.text_encoder.config.hidden_size != 1024:
        raise ValueError("Expected the SD2.x 1024-dimensional text encoder; check the model path")


def generate_image(*, prompt: str, output: str | Path, base_seed: int = 12345,
                   placements: list[Placement] | None = None, database: str | Path | None = None,
                   model: str = MODEL_ID, device: str = "cuda:0", steps: int = 30,
                   guidance: float = 7.0,
                   negative_prompt: str = "deformed, distorted, unnatural, low quality, low resolution",
                   local_files_only: bool = False, allow_overlap: bool = False, pipe=None) -> dict:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError("guidance must be finite and nonnegative")
    integer(base_seed, "base_seed")
    if not 0 <= base_seed < 2**63:
        raise ValueError("base_seed must be between 0 and 2**63 - 1")
    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("Output must be a .png path")
    meta_path, noise_path = output.with_suffix(".json"), output.with_suffix(".latents.safetensors")
    if any(p.exists() for p in (output, meta_path, noise_path)):
        raise ValueError("Output already exists; choose a new image filename")
    placements = list(placements or [])
    patches, manifest = load_database(database) if database else ({}, None)
    if placements and database is None:
        raise ValueError("Patch placements require --database")
    compose_noise(torch.zeros((1, 4, 96, 96)), placements, patches, allow_overlap=allow_overlap)
    if pipe is None:
        pipe = load_pipeline(model, device, local_files_only=local_files_only)
    validate_pipeline(pipe)
    target = torch.device(device)
    dtype = torch.float16 if target.type == "cuda" else torch.float32
    noise = initial_noise(base_seed, device=device, dtype=dtype)
    composed = compose_noise(noise, placements, patches, allow_overlap=allow_overlap)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt, negative_prompt=negative_prompt, width=768, height=768,
            latents=composed.clone(), num_inference_steps=steps, guidance_scale=guidance,
        )
    versions = {"torch": str(torch.__version__)}
    for name in ("diffusers", "transformers", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    metadata = {
        "model": model, "prompt": prompt, "negative_prompt": negative_prompt,
        "width": 768, "height": 768, "base_seed": base_seed, "device": str(target),
        "base_noise_dtype": str(dtype), "num_inference_steps": steps, "guidance_scale": guidance,
        "scheduler": type(pipe.scheduler).__name__, "scheduler_config": dict(pipe.scheduler.config),
        "placements": [asdict(p) for p in placements], "allow_overlap": allow_overlap,
        "patch_database": manifest, "versions": versions,
        "saved_latents": noise_path.name, "latents_kind": "raw_unscaled_initial_noise",
    }
    metadata_json = json.dumps(metadata, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output)
    save_file({"latents": composed.cpu().contiguous()}, str(noise_path))
    meta_path.write_text(metadata_json)
    return metadata
