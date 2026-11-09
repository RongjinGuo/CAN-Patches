from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import torch

from .core import integer


def source_image(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    if image.size != (768, 768):
        raise ValueError("Editing expects a 768×768 image; resize image and masks together explicitly")
    return image


def read_mask(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        mask = ImageOps.exif_transpose(im).convert("L")
    if mask.size not in ((768, 768), (96, 96)):
        raise ValueError("Mask must be 768×768 pixels or 96×96 latent cells")
    return Image.fromarray((np.array(mask) >= 128).astype(np.uint8) * 255)


def mask_tensor(mask: Image.Image, device="cpu", dtype=torch.float32) -> torch.Tensor:
    array = np.array(mask.convert("L")) >= 128
    if mask.size == (768, 768):
        array = array[::8, ::8]
    elif mask.size != (96, 96):
        raise ValueError("Mask must be 768×768 or 96×96")
    if not array.any():
        raise ValueError("Edit mask is empty at latent resolution; enlarge the selected region")
    return torch.from_numpy(array.copy()).to(device=device, dtype=dtype)[None, None]


def full_mask(mask: Image.Image) -> Image.Image:
    return mask.resize((768, 768), Image.Resampling.NEAREST)


def save_mask(mask: Image.Image, output: str | Path) -> None:
    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("Save binary masks as .png")
    if output.exists():
        raise ValueError(f"Mask output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    mask.convert("L").save(output)


def rectangle_mask(box, output: str | Path) -> None:
    if len(box) != 4:
        raise ValueError("Pixel rectangle must be x0,y0,x1,y1")
    x0, y0, x1, y1 = [integer(v, "pixel coordinate") for v in box]
    if not (0 <= x0 < x1 <= 768 and 0 <= y0 < y1 <= 768):
        raise ValueError("Pixel rectangle must fit the 768×768 image")
    mask = Image.new("L", (768, 768), 0)
    ImageDraw.Draw(mask).rectangle((x0, y0, x1-1, y1-1), fill=255)
    save_mask(mask, output)


def union_masks(paths, output: str | Path, dilate: int = 0) -> None:
    integer(dilate, "dilate")
    if not 0 <= dilate <= 64 or not paths:
        raise ValueError("Select at least one mask; dilation radius must be 0–64 pixels")
    masks = [read_mask(p) for p in paths]
    if len({m.size for m in masks}) != 1:
        raise ValueError("Merge masks at the same resolution")
    union = np.logical_or.reduce([np.array(m) >= 128 for m in masks])
    if not union.any():
        raise ValueError("Selected masks are empty")
    mask = Image.fromarray(union.astype(np.uint8) * 255)
    if dilate:
        mask = mask.filter(ImageFilter.MaxFilter(2*dilate+1))
    save_mask(mask, output)


def sam_masks(*, image: str | Path, checkpoint: str | Path, output: str | Path,
              model_type="vit_l", device="cuda:0", box=None, points=None) -> dict:
    if not Path(checkpoint).is_file():
        raise ValueError("Provide an existing SAM checkpoint with --checkpoint")
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new SAM output directory")
    rgb = source_image(image)
    array = np.array(rgb)
    if box is not None:
        if len(box) != 4 or not (0 <= box[0] < box[2] <= 768 and 0 <= box[1] < box[3] <= 768):
            raise ValueError("SAM box must be x0,y0,x1,y1 in 768×768 pixel coordinates")
    if points:
        for p in points:
            if len(p) != 3 or not (0 <= p[0] < 768 and 0 <= p[1] < 768 and p[2] in (0, 1)):
                raise ValueError("SAM points must be x,y,label; label 1=foreground, 0=background")
    try:
        from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError("SAM extraction needs matching torchvision and the official package: pip install git+https://github.com/facebookresearch/segment-anything.git") from exc
    if model_type not in sam_model_registry:
        raise ValueError(f"Unknown SAM model type: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device=device)
    with torch.inference_mode():
        if box is not None or points:
            predictor = SamPredictor(sam)
            predictor.set_image(array)
            coords = np.array([p[:2] for p in points], dtype=np.float32) if points else None
            labels = np.array([p[2] for p in points], dtype=np.int32) if points else None
            masks, scores, _ = predictor.predict(
                point_coords=coords, point_labels=labels,
                box=np.array(box, dtype=np.float32) if box is not None else None,
                multimask_output=True,
            )
            idx = int(np.argmax(scores))
            candidates = [{"segmentation": masks[idx], "predicted_iou": float(scores[idx])}]
        else:
            candidates = SamAutomaticMaskGenerator(sam).generate(array)
            candidates.sort(key=lambda item: float(item.get("stability_score", 0)), reverse=True)
    if not candidates:
        raise ValueError("SAM produced no masks")
    output.mkdir(parents=True)
    records = []
    for i, item in enumerate(candidates):
        mask = np.asarray(item["segmentation"], dtype=bool)
        if mask.shape != (768, 768):
            raise ValueError("SAM returned a mask that does not match the source image")
        name = f"mask_{i:03d}.png"
        save_mask(Image.fromarray(mask.astype(np.uint8)*255), output/name)
        overlay = Image.blend(rgb, Image.composite(Image.new("RGB", rgb.size, "red"), rgb,
                              Image.fromarray(mask.astype(np.uint8)*255)), 0.45)
        overlay.save(output/f"overlay_{i:03d}.png")
        records.append({"file": name, "area": int(mask.sum()),
                        "predicted_iou": float(item.get("predicted_iou", 0)),
                        "stability_score": float(item["stability_score"]) if item.get("stability_score") is not None else None})
    metadata = {"image": str(image), "model_type": model_type, "box": box, "points": points,
                "mask_semantics": "white_edit_black_preserve", "masks": records}
    (output/"manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata
