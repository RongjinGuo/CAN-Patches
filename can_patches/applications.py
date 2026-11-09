from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .core import Placement, compose_noise, integer
from .database import load_database, read_catalog
from .generation import generate_image, load_pipeline


def settings(path=None):
    raw = Path(path).read_text() if path else files("can_patches").joinpath("data/applications.json").read_text()
    data = json.loads(raw)
    if data.get("schema_version") != 1 or data.get("image_size") != [768, 768]:
        raise ValueError("Application settings must use schema version 1 and 768×768 images")
    return data


def placement_records(records):
    if not isinstance(records, list):
        raise ValueError("patches must be a list")
    result = []
    for r in records:
        if not isinstance(r, dict) or set(r)-{"id", "x", "y", "flip_x", "flip_y"}:
            raise ValueError("Each patch needs id, x, y, and optional flip_x/flip_y")
        for key in ("flip_x", "flip_y"):
            if key in r and not isinstance(r[key], bool):
                raise ValueError(f"{key} must be a JSON boolean")
        result.append(Placement(r["id"], integer(r["x"], "x"), integer(r["y"], "y"),
                                r.get("flip_x", False), r.get("flip_y", False)))
    return result


def counting_plan(config: dict, common: dict, recipes=None) -> list[dict]:
    if not isinstance(config, dict):
        raise ValueError("Counting config must be a JSON object")
    recipes = read_catalog() if recipes is None else recipes
    by_id = {r.id: r for r in recipes}
    mapping = {**common["color_patches"], **config.get("color_patches", {})}
    anchors = config.get("anchors", common["anchors"])
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Counting config needs a nonempty cases list")
    if not isinstance(anchors, list) or not anchors:
        raise ValueError("Counting anchors must be a nonempty list")
    layouts = []
    names = set()
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError("Each counting case must be a JSON object")
        groups = case.get("groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError("Each counting case needs groups")
        placements, phrases, slot = [], [], 0
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError("Each counting group must be a JSON object")
            count = integer(group["count"], "object count")
            if count < 1 or slot+count > len(anchors):
                raise ValueError("Requested count exceeds the available fixed anchors")
            color = group["color"]
            patch_id = group.get("patch_id", mapping.get(color))
            if patch_id not in by_id or by_id[patch_id].color != color:
                raise ValueError(f"No matching crop recipe for {color}; set a color_patches mapping")
            noun = group["object"] if count == 1 else group["plural"]
            if not isinstance(noun, str) or not noun.strip():
                raise ValueError("Supply singular object and plural strings")
            phrases.append(f"{count} {color} {noun}")
            for _ in range(count):
                a = anchors[slot]
                if not isinstance(a, dict) or set(a)-{"x", "y", "flip_x", "flip_y"}:
                    raise ValueError("Anchor needs x, y, and optional flip_x/flip_y")
                placements += placement_records([{"id": patch_id, **a}])
                slot += 1
        prompt = case.get("prompt") or (" and ".join(phrases)+" "+config.get("scene", "on a wooden table")+".")
        tensors = {}
        for p in placements:
            x0, y0, x1, y1 = by_id[p.patch_id].bbox
            tensors[p.patch_id] = torch.zeros((1, 4, y1-y0, x1-x0))
        compose_noise(torch.zeros((1, 4, 96, 96)), placements, tensors)
        name = case.get("name", f"case_{i+1:02d}")
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
            raise ValueError("Counting case names must be lowercase letters, digits, '-' or '_'")
        if name in names:
            raise ValueError("Counting case names must be unique")
        names.add(name)
        seed = config.get("base_seed", common["generation"]["base_seed"])
        integer(seed, "base_seed")
        if not 0 <= seed < 2**63:
            raise ValueError("base_seed is out of range")
        layouts.append({"name": name, "prompt": prompt, "base_seed": seed,
                        "requested_groups": groups, "patches": [{"id": p.patch_id, "x": p.x, "y": p.y,
                        "flip_x": p.flip_x, "flip_y": p.flip_y} for p in placements]})
    return layouts


def run_counting(*, config_path, output, database=None, settings_path=None,
                 model=None, device="cuda:0", local_files_only=False, dry_run=False):
    common = settings(settings_path)
    config = json.loads(Path(config_path).read_text())
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new counting output directory")
    if database:
        _, manifest = load_database(database)
        from .core import Recipe
        recipes = [Recipe.from_dict(r) for r in manifest["patches"]]
    else:
        recipes = read_catalog()
    plan = counting_plan(config, common, recipes)
    model = model or common["model"]
    sampling = {**common["generation"], **config.get("generation", {})}
    if set(config.get("generation", {})) - {"steps", "guidance", "negative_prompt"}:
        raise ValueError("Override base_seed at the top level; sampling overrides are steps, guidance, negative_prompt")
    if not dry_run and database is None:
        raise ValueError("Counting generation requires an exported patch database")
    record = {"schema_version": 1, "application": "fixed_anchor_counting", "model": model,
              "sampling": sampling, "dry_run": dry_run, "cases": plan}
    output.mkdir(parents=True)
    (output/"plan.json").write_text(json.dumps(record, indent=2)+"\n")
    for layout in plan:
        draw_counting_layout(layout, output/(layout["name"]+".layout.png"), recipes)
    if dry_run:
        return record
    pipe = load_pipeline(model, device, local_files_only=local_files_only)
    for layout in plan:
        generate_image(prompt=layout["prompt"], output=output/(layout["name"]+".png"),
                       base_seed=layout["base_seed"], database=database,
                       placements=placement_records(layout["patches"]), model=model, device=device,
                       steps=sampling["steps"], guidance=sampling["guidance"],
                       negative_prompt=sampling["negative_prompt"], pipe=pipe)
    return record


def draw_counting_layout(layout, output, recipes=None):
    image = Image.new("RGB", (768, 816), "#f3f5f8")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    draw.text((12, 12), "Noise patch layout (96 x 96)", fill="#172231", font=font)
    recipes = {r.id: r for r in (read_catalog() if recipes is None else recipes)}
    colors = {"red": "#dd414b", "blue": "#3079d6", "yellow": "#edcf38", "white": "#e2e5e9",
              "orange": "#ee8a33", "green": "#39a76a", "pink": "#e880bd", "purple": "#9660c7",
              "brown": "#9c7050", "gray": "#999999", "black": "#353535"}
    for grid in range(0, 769, 64):
        draw.line((grid, 48, grid, 815), fill="#dfe4ec")
        draw.line((0, 48+grid, 767, 48+grid), fill="#dfe4ec")
    for i, p in enumerate(layout["patches"]):
        recipe = recipes[p["id"]]
        x0, y0, x1, y1 = recipe.bbox
        w, h = x1-x0, y1-y0
        color = recipe.color
        x, y = p["x"]*8, 48+p["y"]*8
        draw.rectangle((x, y, x+w*8-1, y+h*8-1), fill=colors.get(color, "#999999"), outline="#172231", width=2)
        draw.text((x+7, y+7), f"{i+1}. {p['id']}",
                  fill="white" if color in ("black", "brown", "red", "blue", "purple") else "#111111", font=font)
    image.save(output)
