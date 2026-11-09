from __future__ import annotations

import argparse
from importlib.resources import files
import json
from pathlib import Path

from .core import MODEL_ID, Placement, Recipe, crop_patch, source_noise
from .database import build_database, read_archived_noise, read_candidates, read_catalog, save_database


def placement(value: str) -> Placement:
    try:
        patch_id, coords = value.split("@")
        x, y = map(int, coords.split(","))
        return Placement(patch_id, x, y)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use PATCH_ID@X,Y in latent coordinates") from exc


def box(value: str) -> tuple[int, ...]:
    try:
        coords = tuple(map(int, value.split(",")))
        if len(coords) != 4:
            raise ValueError()
        return coords
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use x0,y0,x1,y1 in latent coordinates") from exc


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CAN Patches: SD2.1 initial-noise extraction and composition")
    sub = p.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("catalog", help="List patch seeds and crop coordinates")
    ls.add_argument("--catalog", type=Path)
    candidates = sub.add_parser("candidates", help="List candidate seeds")
    candidates.add_argument("--color")
    build = sub.add_parser("build-db", help="Regenerate CUDA FP16 source noise and export patches")
    build_source = build.add_mutually_exclusive_group()
    build_source.add_argument("--catalog", type=Path)
    build_source.add_argument("--candidates", action="store_true", help="Build the 24×24 patches from candidate seeds")
    build.add_argument("--id", action="append", help="Patch id to include; repeat, or omit for all")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--device", default="cuda:0")
    extract = sub.add_parser("extract", help="Extract one patch from a source seed or archived initial latent")
    src = extract.add_mutually_exclusive_group(required=True)
    src.add_argument("--seed", type=int)
    src.add_argument("--source-latent", type=Path)
    extract.add_argument("--id", required=True)
    extract.add_argument("--color", required=True)
    extract.add_argument("--bbox", type=box, required=True)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--output", type=Path, required=True)
    gen = sub.add_parser("generate", help="Generate using raw noise with optional patch placements")
    gen.add_argument("--prompt", help="Text prompt; overrides a layout's prompt")
    gen.add_argument("--layout", type=Path, help="JSON file containing prompt, base_seed, and patches")
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--model", default=MODEL_ID, help="SD2.1 model id or local Diffusers directory")
    gen.add_argument("--local-files-only", action="store_true")
    gen.add_argument("--device", default="cuda:0")
    gen.add_argument("--database", type=Path)
    gen.add_argument("--base-seed", type=int, help="Background noise seed; defaults to layout seed or 12345")
    gen.add_argument("--patch", type=placement, action="append", default=[])
    gen.add_argument("--steps", type=int, default=30)
    gen.add_argument("--guidance", type=float, default=7.0)
    gen.add_argument("--negative-prompt", default="deformed, distorted, unnatural, low quality, low resolution")
    gen.add_argument("--allow-overlap", action="store_true", help="Later placements overwrite earlier ones")
    count = sub.add_parser("count", help="Run counting cases using common fixed anchors")
    count.add_argument("--config", type=Path, required=True)
    count.add_argument("--settings", type=Path)
    count.add_argument("--database", type=Path)
    count.add_argument("--output-dir", type=Path, required=True)
    count.add_argument("--dry-run", action="store_true", help="Write validated layouts without loading models")
    count.add_argument("--model")
    count.add_argument("--device", default="cuda:0")
    count.add_argument("--local-files-only", action="store_true")
    masks = sub.add_parser("mask", help="Extract, merge, or draw white-edit/black-preserve masks")
    msub = masks.add_subparsers(dest="mask_command", required=True)
    rectangle = msub.add_parser("rectangle", help="Draw a pixel rectangle without SAM weights")
    rectangle.add_argument("--box", type=box, required=True)
    rectangle.add_argument("--output", type=Path, required=True)
    union = msub.add_parser("union", help="Merge selected object masks")
    union.add_argument("--inputs", type=Path, nargs="+", required=True)
    union.add_argument("--dilate", type=int, default=0)
    union.add_argument("--output", type=Path, required=True)
    sam = msub.add_parser("sam", help="Run original Segment Anything with a local checkpoint")
    sam.add_argument("--image", type=Path, required=True)
    sam.add_argument("--checkpoint", type=Path, required=True)
    sam.add_argument("--model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_l")
    sam.add_argument("--device", default="cuda:0")
    sam.add_argument("--box", type=box)
    sam.add_argument("--point", action="append", default=[], help="Pixel x,y,label; 1 selects, 0 excludes")
    sam.add_argument("--output-dir", type=Path, required=True)
    inv = sub.add_parser("invert", help="Save a source-conditioned DDIM inversion and reconstruction")
    inv.add_argument("--image", type=Path, required=True)
    inv.add_argument("--source-prompt", required=True)
    inv.add_argument("--output-dir", type=Path, required=True)
    inv.add_argument("--steps", type=int)
    inv.add_argument("--settings", type=Path)
    inv.add_argument("--model")
    inv.add_argument("--device", default="cuda:0")
    inv.add_argument("--local-files-only", action="store_true")
    edit = sub.add_parser("edit", help="Mask-guided NPI editing with optional patch replacement")
    edit.add_argument("--inversion", type=Path, required=True)
    edit.add_argument("--mask", type=Path, required=True)
    edit.add_argument("--target-prompt")
    edit.add_argument("--config", type=Path, help="Recolor/replace/add application settings")
    edit.add_argument("--settings", type=Path)
    edit.add_argument("--database", type=Path)
    edit.add_argument("--patch", type=placement, action="append", default=[])
    edit.add_argument("--output", type=Path, required=True)
    edit.add_argument("--model")
    edit.add_argument("--device", default="cuda:0")
    edit.add_argument("--local-files-only", action="store_true")
    edit.add_argument("--guidance", type=float)
    edit.add_argument("--method", choices=["npi", "ddim"])
    edit.add_argument("--negative-prompt")
    edit.add_argument("--preserve-steps", help="all, an empty string, or e.g. 0:9,19,24,29")
    edit.add_argument("--reset-region", action=argparse.BooleanOptionalAction, default=None)
    edit.add_argument("--reset-seed", type=int)
    edit.add_argument("--pixel-preserve", action=argparse.BooleanOptionalAction, default=None)
    edit.add_argument("--allow-overlap", action="store_true")
    mining = sub.add_parser("mine-generate", help="Generate the paper's structured color/object/scene prompt sets")
    mining.add_argument("--config", type=Path, help="Defaults to the packaged paper mining protocol")
    mining.add_argument("--output-dir", type=Path, required=True)
    mining.add_argument("--seed-start", type=int)
    mining.add_argument("--num-seeds", type=int)
    mining.add_argument("--colors", nargs="+", help="Subset of the protocol's color names")
    mining.add_argument("--batch-size", type=int, default=1)
    mining.add_argument("--max-images", type=int, help="Generate at most this many new images, then resume later")
    mining.add_argument("--model")
    mining.add_argument("--device", default="cuda:0")
    mining.add_argument("--local-files-only", action="store_true")
    mining.add_argument("--resume", action="store_true")
    mining.add_argument("--dry-run", action="store_true", help="Write the generation plan without model weights")
    return p


def main(argv=None) -> None:
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "catalog":
            for r in read_catalog(args.catalog):
                print(f"{r.id:24} {r.color:8} seed={r.seed} bbox={list(r.bbox)}")
        elif args.command == "candidates":
            data = json.loads(files("can_patches").joinpath("data/candidates.json").read_text())
            if args.color:
                data["candidates"] = [r for r in data["candidates"] if r["color"] == args.color]
                if not data["candidates"]:
                    raise ValueError(f"No candidates for color: {args.color}")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        elif args.command == "build-db":
            recipes = read_candidates() if args.candidates else read_catalog(args.catalog)
            if args.id:
                selected = set(args.id)
                unknown = selected - {r.id for r in recipes}
                if unknown:
                    raise ValueError(f"Unknown patch ids: {sorted(unknown)}")
                recipes = [r for r in recipes if r.id in selected]
            build_database(args.output, recipes, args.device)
            print(f"Saved {len(recipes)} patches to {args.output}")
        elif args.command == "extract":
            origin = "user_selected_seed_crop" if args.seed is not None else "user_selected_archived_noise_crop"
            recipe = Recipe(args.id, args.color, args.seed, args.bbox)
            if args.source_latent:
                noise = read_archived_noise(args.source_latent)
                info = {"kind": origin, "archive_name": args.source_latent.name}
            else:
                import torch
                noise = source_noise(args.seed, args.device)
                info = {"kind": origin, "rng_device": args.device, "noise_dtype": "float16",
                        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}
            save_database(args.output, [recipe], {recipe.id: crop_patch(noise, recipe.bbox)}, info)
            print(f"Saved {recipe.id} to {args.output}")
        elif args.command == "generate":
            from .generation import generate_image
            layout = json.loads(args.layout.read_text()) if args.layout else {}
            if not isinstance(layout, dict):
                raise ValueError("Layout must be a JSON object")
            records = layout.get("patches", [])
            if not isinstance(records, list):
                raise ValueError("Layout patches must be a list")
            layout_patches = []
            for r in records:
                if not isinstance(r, dict) or set(r) - {"id", "x", "y", "flip_x", "flip_y"}:
                    raise ValueError("Each layout patch needs id, x, y, and optional flip_x/flip_y")
                for key in ("flip_x", "flip_y"):
                    if key in r and not isinstance(r[key], bool):
                        raise ValueError(f"{key} must be a JSON boolean")
                layout_patches.append(Placement(r["id"], r["x"], r["y"], r.get("flip_x", False), r.get("flip_y", False)))
            prompt = args.prompt if args.prompt is not None else layout.get("prompt")
            base_seed = args.base_seed if args.base_seed is not None else layout.get("base_seed", 12345)
            generate_image(prompt=prompt, output=args.output, base_seed=base_seed,
                           placements=args.patch or layout_patches, database=args.database, model=args.model,
                           device=args.device, steps=args.steps, guidance=args.guidance,
                           negative_prompt=args.negative_prompt, local_files_only=args.local_files_only,
                           allow_overlap=args.allow_overlap)
            print(f"Saved {args.output}, metadata, and raw initial latents")
        elif args.command == "count":
            from .applications import run_counting
            result = run_counting(config_path=args.config, output=args.output_dir, database=args.database,
                                  settings_path=args.settings, model=args.model, device=args.device,
                                  local_files_only=args.local_files_only, dry_run=args.dry_run)
            print(f"{'Planned' if args.dry_run else 'Generated'} {len(result['cases'])} cases in {args.output_dir}")
        elif args.command == "mask":
            from .masks import rectangle_mask, sam_masks, union_masks
            if args.mask_command == "rectangle":
                rectangle_mask(args.box, args.output)
                print(f"Saved {args.output}")
            elif args.mask_command == "union":
                union_masks(args.inputs, args.output, args.dilate)
                print(f"Saved {args.output}")
            else:
                points = []
                for value in args.point:
                    x, y, label = value.split(",")
                    points.append([float(x), float(y), int(label)])
                result = sam_masks(image=args.image, checkpoint=args.checkpoint, output=args.output_dir,
                                   model_type=args.model_type, device=args.device, box=args.box, points=points)
                print(f"Saved {len(result['masks'])} masks and overlays to {args.output_dir}")
        elif args.command == "invert":
            from .applications import settings
            from .inversion import invert_image
            common = settings(args.settings)
            invert_image(image=args.image, source_prompt=args.source_prompt, output=args.output_dir,
                         steps=args.steps if args.steps is not None else common["editing"]["steps"],
                         model=args.model or common["model"], device=args.device,
                         local_files_only=args.local_files_only)
            print(f"Saved inversion trajectory and reconstruction to {args.output_dir}")
        elif args.command == "edit":
            from .applications import placement_records, settings
            from .editing import edit_image
            common = settings(args.settings)
            config = json.loads(args.config.read_text()) if args.config else {}
            if not isinstance(config, dict):
                raise ValueError("Editing config must be a JSON object")
            supported = {"application", "target_prompt", "patches", "method", "guidance", "negative_prompt",
                         "preserve_steps", "reset_region", "reset_seed", "pixel_preserve"}
            if set(config)-supported:
                raise ValueError(f"Unsupported editing config fields: {sorted(set(config)-supported)}")
            options = {**common["editing"], **config}
            for key in ("target_prompt", "method", "guidance", "negative_prompt", "preserve_steps",
                        "reset_region", "reset_seed", "pixel_preserve"):
                value = getattr(args, key)
                if value is not None:
                    options[key] = value
            edit_image(inversion=args.inversion, mask_path=args.mask, target_prompt=options.get("target_prompt"),
                       output=args.output, database=args.database,
                       placements=args.patch or placement_records(options.get("patches", [])),
                       model=args.model, device=args.device, local_files_only=args.local_files_only,
                       method=options["method"], guidance=options["guidance"],
                       negative_prompt=options.get("negative_prompt"), preserve_steps=options["preserve_steps"],
                       reset_region=options["reset_region"], reset_seed=options["reset_seed"],
                       pixel_preserve=options["pixel_preserve"], allow_overlap=args.allow_overlap)
            print(f"Saved edited image, raw diffusion output, and settings to {args.output}")
        elif args.command == "mine-generate":
            from .mining_generation import run_mining_generation
            result = run_mining_generation(output=args.output_dir, config_path=args.config,
                seed_start=args.seed_start, num_seeds=args.num_seeds, colors=args.colors,
                batch_size=args.batch_size, max_images=args.max_images, model=args.model,
                device=args.device, local_files_only=args.local_files_only, resume=args.resume, dry_run=args.dry_run)
            print(f"{result['status']}: {result['total_images']} planned, {result['generated']} new, {result['skipped']} skipped")
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        p.error(str(exc))
