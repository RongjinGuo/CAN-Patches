from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

from PIL import Image
import torch
from safetensors.torch import save_file

from .core import Placement, compose_noise, initial_noise, integer
from .database import load_database
from .generation import load_pipeline, validate_pipeline
from .inversion import decode_image, load_inversion, paired_schedulers
from .masks import full_mask, mask_tensor, read_mask


def preserve_indices(value, steps):
    if value == "all" or value is None:
        return set(range(steps))
    if isinstance(value, str):
        indices = set()
        for token in value.split(","):
            if not token.strip():
                continue
            if ":" in token:
                start, end = map(int, token.split(":"))
                if start < 0 or end > steps or start >= end:
                    raise ValueError("Preservation range must lie within the denoising steps")
                indices.update(range(start, end))
            else:
                indices.add(int(token))
    elif isinstance(value, list):
        indices = {integer(i, "preservation index") for i in value}
    else:
        raise ValueError("preserve_steps must be 'all', a range string, or an integer list")
    if any(i < 0 or i >= steps for i in indices):
        raise ValueError("Preservation indices are zero-based denoising steps")
    return indices


def prepare_edit_noise(trace, mask, placements, patches, *, device, dtype,
                       reset_region=False, reset_seed=118, allow_overlap=False):
    if tuple(mask.shape) != (1, 1, 96, 96) or not torch.isfinite(mask).all():
        raise ValueError("Edit mask must be finite with shape 1×1×96×96")
    if not torch.all((mask == 0) | (mask == 1)) or not mask.any():
        raise ValueError("Use a nonempty binary edit mask")
    source = trace[-1].to(device=device, dtype=dtype)
    region = mask.to(device=device, dtype=dtype)
    candidate = initial_noise(reset_seed, device, dtype) if reset_region else source.clone()
    candidate = compose_noise(candidate, placements, patches, allow_overlap=allow_overlap)
    fractions = []
    for p in placements:
        h, w = patches[p.patch_id].shape[-2:]
        fraction = float(region[:, :, p.y:p.y+h, p.x:p.x+w].mean())
        if fraction == 0:
            raise ValueError(f"Patch {p.patch_id} does not intersect the white edit mask")
        fractions.append(fraction)
    return region*candidate + (1-region)*source, fractions


def denoise_edit(pipe, trace, scheduler_config, *, source_prompt, target_prompt, mask,
                 initial, guidance=7.6, method="npi", negative_prompt=None, preserve_steps="all"):
    if method not in ("npi", "ddim"):
        raise ValueError("Editing method must be npi or ddim")
    if not isinstance(target_prompt, str) or not target_prompt.strip():
        raise ValueError("Provide a target prompt")
    if not math.isfinite(guidance) or guidance < 1:
        raise ValueError("Editing guidance must be finite and at least 1")
    steps = len(trace)-1
    preserve = preserve_indices(preserve_steps, steps)
    scheduler, _ = paired_schedulers(scheduler_config, steps, initial.device)
    negative = negative_prompt if negative_prompt is not None else (source_prompt if method == "npi" else "")
    conditional, negative_embeds = pipe.encode_prompt(
        prompt=target_prompt, negative_prompt=negative, do_classifier_free_guidance=True,
        device=initial.device, num_images_per_prompt=1,
    )
    embeddings = torch.cat([negative_embeds, conditional])
    z = initial.clone()
    region = mask.to(device=z.device, dtype=z.dtype)
    with torch.inference_mode():
        for i, t in enumerate(scheduler.timesteps):
            model_input = scheduler.scale_model_input(torch.cat([z, z]), t)
            negative_noise, target_noise = pipe.unet(model_input, t, encoder_hidden_states=embeddings).sample.chunk(2)
            noise = negative_noise + guidance*(target_noise-negative_noise)
            z = scheduler.step(noise, t, z, eta=0.0).prev_sample
            if i in preserve:
                reference = trace[steps-1-i].to(device=z.device, dtype=z.dtype)
                z = region*z + (1-region)*reference
    return z, negative, sorted(preserve)


def edit_image(*, inversion, mask_path, target_prompt, output, database=None,
               placements: list[Placement] | None = None, model=None, device="cuda:0",
               guidance=7.6, method="npi", negative_prompt=None, preserve_steps="all",
               reset_region=False, reset_seed=118, pixel_preserve=True, local_files_only=False,
               allow_overlap=False, pipe=None):
    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("Edited output must be a .png path")
    raw_path = output.with_name(output.stem+".raw.png")
    latent_path = output.with_suffix(".edit-latents.safetensors")
    metadata_path = output.with_suffix(".json")
    if any(p.exists() for p in (output, raw_path, latent_path, metadata_path)):
        raise ValueError("Editing output already exists; choose a new filename")
    if not isinstance(reset_region, bool) or not isinstance(pixel_preserve, bool):
        raise ValueError("reset_region and pixel_preserve must be booleans")
    trace, metadata, original = load_inversion(inversion)
    mask_image = read_mask(mask_path)
    region = mask_tensor(mask_image)
    patches, manifest = load_database(database) if database else ({}, None)
    placements = list(placements or [])
    prepare_edit_noise(trace, region, placements, patches, device="cpu", dtype=torch.float32,
                       reset_region=False, allow_overlap=allow_overlap)
    preserve_indices(preserve_steps, metadata["steps"])
    if method not in ("npi", "ddim") or not isinstance(target_prompt, str) or not target_prompt.strip():
        raise ValueError("Provide a nonempty target prompt and method npi or ddim")
    if not math.isfinite(guidance) or guidance < 1:
        raise ValueError("Editing guidance must be finite and at least 1")
    integer(reset_seed, "reset_seed")
    if not 0 <= reset_seed < 2**63:
        raise ValueError("reset_seed is out of range")
    model = model or metadata["model"]
    if pipe is None:
        pipe = load_pipeline(model, device, local_files_only=local_files_only)
    validate_pipeline(pipe)
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    initial, fractions = prepare_edit_noise(
        trace, region, placements, patches, device=device, dtype=dtype,
        reset_region=reset_region, reset_seed=reset_seed, allow_overlap=allow_overlap,
    )
    latent, negative, indices = denoise_edit(
        pipe, trace, metadata["scheduler_config"], source_prompt=metadata["source_prompt"],
        target_prompt=target_prompt, mask=region, initial=initial, guidance=guidance,
        method=method, negative_prompt=negative_prompt, preserve_steps=preserve_steps,
    )
    with torch.inference_mode():
        raw = decode_image(pipe, latent)
    edited = Image.composite(raw, original, full_mask(mask_image)) if pixel_preserve else raw
    record = {"application": "masked_interactive_editing", "model": model, "method": method,
              "source_prompt": metadata["source_prompt"], "target_prompt": target_prompt,
              "negative_prompt": negative, "guidance": guidance, "steps": metadata["steps"],
              "scheduler_config": metadata["scheduler_config"], "preserve_steps": indices,
              "pixel_preserve": pixel_preserve,
              "reset_region": reset_region, "reset_seed": reset_seed if reset_region else None,
              "placements": [asdict(p) for p in placements], "patch_in_mask_fractions": fractions,
              "allow_overlap": allow_overlap, "patch_database": manifest,
              "torch_version": str(torch.__version__), "device": device,
              "raw_output": raw_path.name,
              "pixel_preservation": "final_binary_mask_composite" if pixel_preserve else "latent_blending_only"}
    serialized = json.dumps(record, indent=2)+"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    edited.save(output)
    raw.save(raw_path)
    save_file({"initial": initial.cpu().contiguous(), "final": latent.cpu().contiguous(),
               "mask": region.cpu().contiguous()}, str(latent_path))
    metadata_path.write_text(serialized)
    return record
