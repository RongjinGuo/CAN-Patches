from __future__ import annotations

from dataclasses import asdict
from importlib.resources import files
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .core import LATENT_SHAPE, MODEL_ID, Recipe, crop_patch, source_noise, validate_noise


def read_catalog(path: str | Path | None = None) -> list[Recipe]:
    raw = Path(path).read_text() if path else files("can_patches").joinpath("data/catalog.json").read_text()
    data = json.loads(raw)
    if data.get("schema_version") != 1 or data.get("model_id") != MODEL_ID:
        raise ValueError("Expected a schema_version=1 Stable Diffusion 2.1 catalog")
    if data.get("latent_shape") != list(LATENT_SHAPE):
        raise ValueError("Catalog must use a 1×4×96×96 latent")
    if data.get("source_rng_device") != "cuda" or data.get("source_dtype") != "float16":
        raise ValueError("Patch catalogs must declare CUDA FP16 source noise")
    recipes = [Recipe.from_dict(item) for item in data["patches"]]
    if not recipes or len({r.id for r in recipes}) != len(recipes):
        raise ValueError("Catalog must contain unique, nonempty patch ids")
    return recipes


def read_candidates(path: str | Path | None = None) -> list[Recipe]:
    raw = Path(path).read_text() if path else files("can_patches").joinpath("data/candidates.json").read_text()
    data = json.loads(raw)
    if (data.get("schema_version") != 1 or data.get("model_id") != MODEL_ID
            or data.get("latent_shape") != list(LATENT_SHAPE) or data.get("patch_size") != [24, 24]):
        raise ValueError("Expected SD2.1 candidates with a 1×4×96×96 latent and 24×24 patches")
    recipes = [Recipe(r["patch_id"], r["color"], r["seed"], r["bbox"]) for r in data["candidates"]]
    if not recipes or len({r.id for r in recipes}) != len(recipes):
        raise ValueError("Candidates must contain unique, nonempty patch ids")
    for r in recipes:
        x0, y0, x1, y1 = r.bbox
        if r.seed is None or (x1-x0, y1-y0) != (24, 24):
            raise ValueError(f"Candidate {r.id} requires a source seed and a 24×24 crop")
    return recipes


def save_database(directory: str | Path, recipes: list[Recipe], patches: dict[str, torch.Tensor],
                  source_info: dict) -> dict:
    if not recipes or {r.id for r in recipes} != set(patches) or len(recipes) != len(patches):
        raise ValueError("Recipes and tensor keys must match and be unique")
    for r in recipes:
        x0, y0, x1, y1 = r.bbox
        t = patches[r.id]
        if tuple(t.shape) != (1, 4, y1-y0, x1-x0):
            raise ValueError(f"Tensor shape disagrees with recipe {r.id}")
        if not t.is_floating_point() or not torch.isfinite(t).all():
            raise ValueError(f"Invalid patch tensor for {r.id}")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    tensor_path, manifest_path = target / "patches.safetensors", target / "manifest.json"
    if tensor_path.exists() or manifest_path.exists():
        raise ValueError("Database already exists; choose a new output directory")
    cpu_patches = {key: t.detach().to("cpu").contiguous().clone() for key, t in patches.items()}
    save_file(cpu_patches, str(tensor_path), metadata={"kind": "raw_initial_noise_patches"})
    manifest = {
        "schema_version": 1, "model_id": MODEL_ID, "latent_shape": list(LATENT_SHAPE),
        "tensor_file": "patches.safetensors",
        "source": source_info,
        "patches": [{**asdict(r), "dtype": str(cpu_patches[r.id].dtype).replace("torch.", "")}
                    for r in recipes],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_database(directory: str | Path, recipes: list[Recipe], device: str = "cuda:0") -> dict:
    if torch.device(device).type != "cuda":
        raise ValueError("Build source patches on CUDA with FP16 noise")
    patches = {}
    for seed in sorted({r.seed for r in recipes if r.seed is not None}):
        noise = source_noise(seed, device=device)
        for r in recipes:
            if r.seed == seed:
                patches[r.id] = crop_patch(noise, r.bbox).cpu()
    if any(r.seed is None for r in recipes):
        raise ValueError("Seed recipes require a source seed; use extract for an archived tensor")
    return save_database(directory, recipes, patches, {
        "kind": "regenerated_seed_recipes", "rng_device": str(torch.device(device)),
        "noise_dtype": "float16", "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(device),
    })


def load_database(directory: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("model_id") != MODEL_ID
            or manifest.get("latent_shape") != list(LATENT_SHAPE)
            or manifest.get("tensor_file") != "patches.safetensors"):
        raise ValueError("Unsupported SD2.1 patch database manifest")
    tensor_path = directory / "patches.safetensors"
    patches = load_file(str(tensor_path), device="cpu")
    records = manifest.get("patches", [])
    recipes = [Recipe.from_dict(item) for item in records]
    if not recipes or len({r.id for r in recipes}) != len(recipes) or set(patches) != {r.id for r in recipes}:
        raise ValueError("Database tensor keys and manifest disagree")
    for recipe, record in zip(recipes, records):
        t = patches[recipe.id]
        x0, y0, x1, y1 = recipe.bbox
        if (tuple(t.shape) != (1, 4, y1-y0, x1-x0) or not t.is_floating_point()
                or not torch.isfinite(t).all()
                or str(t.dtype).replace("torch.", "") != record.get("dtype")):
            raise ValueError(f"Invalid stored tensor for {recipe.id}")
    return patches, manifest


def read_archived_noise(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".safetensors":
        tensors = load_file(str(path), device="cpu")
        if set(tensors) != {"latents"}:
            raise ValueError("Archived safetensors must contain exactly one key: latents")
        tensor = tensors["latents"]
    elif path.suffix == ".pt":
        tensor = torch.load(str(path), map_location="cpu", weights_only=True)
    else:
        raise ValueError("Use a .pt tensor or .safetensors file with key 'latents'")
    validate_noise(tensor)
    return tensor
