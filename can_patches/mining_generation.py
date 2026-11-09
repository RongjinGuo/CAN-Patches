from __future__ import annotations

import importlib.metadata
from importlib.resources import files
import json
import math
from pathlib import Path
import re

import torch
from safetensors.torch import load_file, save_file

from .core import LATENT_SHAPE, initial_noise, integer, validate_noise
from .generation import load_pipeline, validate_pipeline


def atomic_json(path: Path, data) -> None:
    temp = path.with_name("."+path.name+".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n")
    temp.replace(path)


def read_protocol(path=None) -> dict:
    raw = Path(path).read_text() if path else files("can_patches").joinpath("data/mining_protocol.json").read_text()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Mining generation config must use schema_version 1")
    supported = {"schema_version", "model", "image_size", "colors", "objects", "scenes", "seeds",
                 "sampling"}
    if set(config)-supported:
        raise ValueError(f"Unknown mining settings: {sorted(set(config)-supported)}")
    if config.get("image_size") != [768, 768]:
        raise ValueError("This SD2.1 mining protocol uses 768×768 images")
    if not isinstance(config.get("model"), str) or not config["model"].strip():
        raise ValueError("Specify the SD2.1 model id or a local Diffusers directory")
    return config


def prompt_records(config: dict) -> list[dict]:
    colors, objects, scenes = [config.get(k) for k in ("colors", "objects", "scenes")]
    for values, label in [(colors, "colors"), (objects, "objects")]:
        if (not isinstance(values, list) or not values
                or not all(isinstance(v, str) and v.strip() for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{label} must be a nonempty list of unique strings")
    if not all(re.fullmatch(r"[a-z]+", c) for c in colors):
        raise ValueError("Color names must be lowercase English words, e.g. red, grey, purple")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Provide scene records with name, preposition, and text")
    names = []
    for scene in scenes:
        if not isinstance(scene, dict) or set(scene) != {"name", "preposition", "text"}:
            raise ValueError("Each scene must have exactly name, preposition, and text")
        if not all(isinstance(v, str) and v.strip() for v in scene.values()):
            raise ValueError("Scene fields must be nonempty strings")
        names.append(scene["name"])
    if len(set(names)) != len(names):
        raise ValueError("Scene names must be unique")
    records = []
    for ci, color in enumerate(colors):
        for oi, obj in enumerate(objects):
            for si, scene in enumerate(scenes):
                records.append({"prompt_id": f"c{ci:02d}_o{oi:02d}_s{si:02d}",
                                "image_name": f"o{oi:02d}_s{si:02d}.png", "color": color,
                                "object": obj, "scene": scene["name"], "preposition": scene["preposition"],
                                "prompt": f"a {color} {obj} {scene['preposition']} {scene['text']}."})
    return records


def mining_plan(config, *, seed_start=None, num_seeds=None, colors=None, batch_size=1,
                model=None, device="cuda:0") -> dict:
    config = json.loads(json.dumps(config))
    if colors is not None:
        if not isinstance(colors, list) or not colors or len(set(colors)) != len(colors):
            raise ValueError("Select distinct color names")
        unknown = set(colors)-set(config["colors"])
        if unknown:
            raise ValueError(f"Colors absent from this protocol: {sorted(unknown)}")
        config["colors"] = colors
    records = prompt_records(config)
    seeds = config.get("seeds", {})
    start = seeds.get("start") if seed_start is None else seed_start
    count = seeds.get("count") if num_seeds is None else num_seeds
    integer(start, "seed_start"); integer(count, "num_seeds"); integer(batch_size, "batch_size")
    if count < 1 or not 0 <= start <= 999999 or start+count > 1000000:
        raise ValueError("Source seed interval must lie within 0..999999 and have positive length")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    sampling = config.get("sampling")
    if not isinstance(sampling, dict) or set(sampling) != {"scheduler", "steps", "guidance", "negative_prompt"}:
        raise ValueError("sampling needs scheduler, steps, guidance, and negative_prompt")
    integer(sampling["steps"], "steps")
    if sampling["scheduler"] != "ddim" or not 1 <= sampling["steps"] <= 1000:
        raise ValueError("Mining generation uses DDIM with 1..1000 steps")
    if not isinstance(sampling["guidance"], (int, float)) or not math.isfinite(sampling["guidance"]) or sampling["guidance"] < 0:
        raise ValueError("guidance must be finite and nonnegative")
    if not isinstance(sampling["negative_prompt"], str):
        raise ValueError("negative_prompt must be a string")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Mining generation supports CPU or CUDA")
    identity = {"model": model or config["model"], "image_size": [768, 768],
                "colors": config["colors"], "objects": config["objects"], "scenes": config["scenes"],
                "seed_start": start, "num_seeds": count, "batch_size": batch_size,
                "sampling": sampling, "device": str(target),
                "source_noise_dtype": "float16" if target.type == "cuda" else "float32"}
    return {"schema_version": 1, "stage": "mining_generation", "identity": identity,
            "prompts_per_seed": len(records),
            "prompts_per_seed_color": len(config["objects"])*len(config["scenes"]),
            "total_images": len(records)*count, "status": "planned",
            "prompts": records}


def runtime_record(pipe):
    versions = {"torch": str(torch.__version__)}
    for name in ("diffusers", "transformers", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return {"versions": versions, "cuda_version": torch.version.cuda,
            "scheduler_class": type(pipe.scheduler).__name__,
            "scheduler_config": {k: v for k, v in dict(pipe.scheduler.config).items() if not k.startswith("_")},
            "pipeline_class": type(pipe).__name__}


def saved_seed_noise(seed_dir, seed, identity, *, resume):
    tensor_path, info_path = seed_dir/"initial_noise.safetensors", seed_dir/"noise.json"
    if tensor_path.exists() and info_path.exists():
        info = json.loads(info_path.read_text())
        if info.get("seed") != seed or info.get("dtype") != identity["source_noise_dtype"]:
            raise ValueError(f"Saved source noise does not match seed {seed}")
        tensors = load_file(str(tensor_path), device="cpu")
        if set(tensors) != {"latents"}:
            raise ValueError("Saved source noise must contain only latents")
        noise = tensors["latents"]
        validate_noise(noise)
        if str(noise.dtype).replace("torch.", "") != info["dtype"]:
            raise ValueError("Saved source noise dtype disagrees with its metadata")
        return noise.to(identity["device"]), info
    if (tensor_path.exists() or info_path.exists()) and not resume:
        raise ValueError("Partial noise output exists; resume to regenerate the incomplete record")
    dtype = torch.float16 if identity["source_noise_dtype"] == "float16" else torch.float32
    noise = initial_noise(seed, device=identity["device"], dtype=dtype)
    seed_dir.mkdir(parents=True, exist_ok=True)
    temp = tensor_path.with_name("."+tensor_path.name+".tmp")
    save_file({"latents": noise.detach().cpu().contiguous()}, str(temp))
    temp.replace(tensor_path)
    info = {"seed": seed, "shape": list(LATENT_SHAPE), "dtype": identity["source_noise_dtype"],
            "rng_device": identity["device"], "kind": "raw_unscaled_initial_noise"}
    atomic_json(info_path, info)
    return noise, info


def run_mining_generation(*, output, config_path=None, seed_start=None, num_seeds=None,
                          colors=None, batch_size=1, model=None, device="cuda:0", resume=False,
                          dry_run=False, max_images=None, local_files_only=False, pipe=None):
    plan = mining_plan(read_protocol(config_path), seed_start=seed_start, num_seeds=num_seeds,
                       colors=colors, batch_size=batch_size, model=model, device=device)
    if max_images is not None:
        integer(max_images, "max_images")
        if max_images < 1:
            raise ValueError("max_images must be positive")
    output = Path(output)
    if output.exists():
        if not resume:
            raise ValueError("Output directory exists; use --resume for this exact run")
        if not (output/"run.json").is_file():
            raise ValueError("Existing directory has no mining run manifest")
        existing = json.loads((output/"run.json").read_text())
        if existing.get("identity") != plan["identity"]:
            raise ValueError("Resume settings differ from the recorded mining run; use a new directory")
        if existing.get("prompts") != plan["prompts"]:
            raise ValueError("Recorded mining prompts differ from the selected protocol")
        plan = existing
    else:
        output.mkdir(parents=True)
        atomic_json(output/"run.json", plan)
        with (output/"prompts.jsonl").open("w") as handle:
            for record in plan["prompts"]:
                handle.write(json.dumps(record, ensure_ascii=False)+"\n")
    if dry_run:
        return {"total_images": plan["total_images"], "prompts_per_seed": plan["prompts_per_seed"],
                "generated": 0, "skipped": 0, "status": plan["status"]}
    from diffusers import DDIMScheduler
    identity = plan["identity"]
    if pipe is None:
        pipe = load_pipeline(identity["model"], device, local_files_only=local_files_only)
    validate_pipeline(pipe)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    runtime = runtime_record(pipe)
    if plan.get("runtime") is not None and plan["runtime"] != runtime:
        raise ValueError("Software or scheduler differs from the original run; use a new directory")
    plan.update(runtime=runtime, status="running")
    atomic_json(output/"run.json", plan)
    generated, skipped, limit_reached = 0, 0, False
    sampling = identity["sampling"]
    try:
        for seed in range(identity["seed_start"], identity["seed_start"]+identity["num_seeds"]):
            seed_dir = output/f"seed_{seed:06d}"
            noise, _ = saved_seed_noise(seed_dir, seed, identity, resume=resume)
            for color in identity["colors"]:
                color_dir = seed_dir/color
                color_dir.mkdir(exist_ok=True)
                records_path = color_dir/"records.json"
                completed = json.loads(records_path.read_text()) if records_path.exists() else {}
                if not isinstance(completed, dict):
                    raise ValueError("Invalid per-color generation records")
                pending = []
                color_records = [r for r in plan["prompts"] if r["color"] == color]
                expected_ids = {r["prompt_id"] for r in color_records}
                if set(completed)-expected_ids:
                    raise ValueError("Per-color records contain unknown prompt ids")
                for record in color_records:
                    image_path = color_dir/record["image_name"]
                    previous = completed.get(record["prompt_id"])
                    if previous is not None and image_path.exists():
                        if previous.get("prompt") != record["prompt"] or previous.get("seed") != seed:
                            raise ValueError(f"Completed record does not match prompt or seed: {image_path}")
                        skipped += 1
                    else:
                        pending.append(record)
                for start in range(0, len(pending), batch_size):
                    batch = pending[start:start+batch_size]
                    if max_images is not None:
                        batch = batch[:max_images-generated]
                    if not batch:
                        limit_reached = True
                        break
                    latents = noise.repeat(len(batch), 1, 1, 1).clone()
                    with torch.inference_mode():
                        result = pipe(prompt=[r["prompt"] for r in batch], negative_prompt=sampling["negative_prompt"],
                                      width=768, height=768, latents=latents,
                                      num_inference_steps=sampling["steps"], guidance_scale=sampling["guidance"], eta=0.0)
                    if len(result.images) != len(batch):
                        raise ValueError("Pipeline returned a different number of images than prompts")
                    for record, image in zip(batch, result.images):
                        if image.size != (768, 768):
                            raise ValueError("Pipeline returned an image at the wrong resolution")
                        image_path = color_dir/record["image_name"]
                        temp = image_path.with_name("."+image_path.name+".tmp")
                        image.save(temp, format="PNG")
                        temp.replace(image_path)
                        completed[record["prompt_id"]] = {**record, "seed": seed,
                            "initial_noise": "../initial_noise.safetensors"}
                    atomic_json(records_path, completed)
                    generated += len(batch)
                    print(f"seed={seed} color={color}: {len(completed)}/{len(color_records)}; generated={generated}", flush=True)
                if limit_reached:
                    break
            if limit_reached:
                break
    except Exception as exc:
        plan.update(status="interrupted", last_error=f"{type(exc).__name__}: {exc}")
        atomic_json(output/"run.json", plan)
        raise
    plan.update(status="complete" if generated+skipped == plan["total_images"] else "partial",
                last_invocation={"generated": generated, "skipped": skipped})
    plan.pop("last_error", None)
    atomic_json(output/"run.json", plan)
    return {"total_images": plan["total_images"], "prompts_per_seed": plan["prompts_per_seed"],
            "generated": generated, "skipped": skipped, "status": plan["status"]}
