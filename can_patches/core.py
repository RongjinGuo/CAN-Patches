from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import torch

MODEL_ID = "stabilityai/stable-diffusion-2-1"
LATENT_SHAPE = (1, 4, 96, 96)


def integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def validate_noise(tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != LATENT_SHAPE:
        raise ValueError(f"Expected raw initial noise with shape {LATENT_SHAPE}")
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise ValueError("Noise must contain finite floating-point values")


def bbox_xyxy(box: Sequence[int]) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError("bbox must be [x0, y0, x1, y1] in latent coordinates")
    x0, y0, x1, y1 = [integer(v, "bbox coordinate") for v in box]
    if not (0 <= x0 < x1 <= 96 and 0 <= y0 < y1 <= 96):
        raise ValueError("bbox must have positive area and lie inside the 96×96 latent")
    return x0, y0, x1, y1


@dataclass(frozen=True)
class Recipe:
    id: str
    color: str
    seed: int | None
    bbox: tuple[int, int, int, int]

    def __post_init__(self):
        if not isinstance(self.id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.id):
            raise ValueError("Patch id must use lowercase letters, digits, '-' or '_'")
        if not isinstance(self.color, str) or not self.color.strip():
            raise ValueError("A color label is required")
        if self.seed is not None:
            integer(self.seed, "seed")
            if not 0 <= self.seed <= 999999:
                raise ValueError("Source patch seeds must be between 0 and 999999")
        object.__setattr__(self, "bbox", bbox_xyxy(self.bbox))

    @classmethod
    def from_dict(cls, data: dict) -> Recipe:
        return cls(data["id"], data["color"], data["seed"], data["bbox"])


def initial_noise(seed: int, device: str = "cuda:0", dtype=torch.float16) -> torch.Tensor:
    integer(seed, "seed")
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be between 0 and 2**63 - 1")
    if dtype not in (torch.float16, torch.float32):
        raise ValueError("Supported noise dtypes: float16, float32")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Use CPU or CUDA for explicit noise generation")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; source patch seeds require an NVIDIA GPU")
    generator = torch.Generator(device=target).manual_seed(seed)
    return torch.randn(LATENT_SHAPE, generator=generator, device=target, dtype=dtype)


def source_noise(seed: int, device: str = "cuda:0") -> torch.Tensor:
    if torch.device(device).type != "cuda":
        raise ValueError("Source patches require CUDA FP16 noise")
    return initial_noise(seed, device=device, dtype=torch.float16)


def crop_patch(noise: torch.Tensor, box: Sequence[int]) -> torch.Tensor:
    validate_noise(noise)
    x0, y0, x1, y1 = bbox_xyxy(box)
    return noise[:, :, y0:y1, x0:x1].clone().contiguous()


@dataclass(frozen=True)
class Placement:
    patch_id: str
    x: int
    y: int
    flip_x: bool = False
    flip_y: bool = False


def compose_noise(
    noise: torch.Tensor,
    placements: Sequence[Placement],
    patches: Mapping[str, torch.Tensor],
    *,
    allow_overlap: bool = False,
) -> torch.Tensor:
    validate_noise(noise)
    occupied = torch.zeros((96, 96), dtype=torch.bool)
    prepared = []
    for p in placements:
        integer(p.x, "placement x")
        integer(p.y, "placement y")
        if p.patch_id not in patches:
            raise ValueError(f"Unknown patch id: {p.patch_id}")
        patch = patches[p.patch_id]
        if patch.ndim != 4 or tuple(patch.shape[:2]) != (1, 4):
            raise ValueError(f"Invalid patch shape for {p.patch_id}: {tuple(patch.shape)}")
        if not patch.is_floating_point() or not torch.isfinite(patch).all():
            raise ValueError(f"Invalid floating-point patch: {p.patch_id}")
        h, w = patch.shape[-2:]
        if h <= 0 or w <= 0 or not (0 <= p.x <= 96-w and 0 <= p.y <= 96-h):
            raise ValueError(f"Patch {p.patch_id} does not fit at x={p.x}, y={p.y}")
        if not allow_overlap and occupied[p.y:p.y+h, p.x:p.x+w].any():
            raise ValueError("Patches overlap; move them or explicitly set allow_overlap")
        occupied[p.y:p.y+h, p.x:p.x+w] = True
        dims = ([-1] if p.flip_x else []) + ([-2] if p.flip_y else [])
        prepared.append((p, patch.flip(dims) if dims else patch))
    result = noise.clone()
    for p, patch in prepared:
        h, w = patch.shape[-2:]
        result[:, :, p.y:p.y+h, p.x:p.x+w] = patch.to(device=result.device, dtype=result.dtype)
    return result
