from __future__ import annotations

from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def read_id_list(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def find_file(directory: str | Path, stem: str, extensions: Iterable[str]) -> Path:
    directory = Path(directory)
    direct = directory / stem
    if direct.exists():
        return direct

    for ext in extensions:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    matches = [p for p in directory.glob(f"{stem}.*") if p.is_file()]
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(f"Could not uniquely find '{stem}' in {directory}")


def _find_optional_file(
    directory: str | Path,
    stem: str,
    extensions: Iterable[str],
) -> Path | None:
    directory = Path(directory)

    for ext in extensions:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    matches = [p for p in directory.glob(f"{stem}.*") if p.is_file()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files match '{stem}' in {directory}: "
            f"{[p.name for p in matches]}"
        )
    return None


def _read_single_channel_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")

    if mask.ndim == 3:
        if (
            np.array_equal(mask[..., 0], mask[..., 1])
            and np.array_equal(mask[..., 1], mask[..., 2])
        ):
            mask = mask[..., 0]
        else:
            raise ValueError(
                f"Mask must be single-channel or gray RGB, but got a color mask: {path}"
            )

    return mask


def load_semantic_mask(
    masks_dir: str | Path,
    sample_id: str,
    extensions: Iterable[str],
    num_classes: int = 3,
    ignore_index: int = 255,
) -> tuple[np.ndarray, dict]:
    """
    Load either:

    1) one indexed semantic mask named:
       <sample_id>.png

    or the project's three-mask bundle:
       <sample_id>_background.png
       <sample_id>_movers.png
       <sample_id>_wires.png

    Output class IDs:
       0 = background
       1 = wire
       2 = mover

    Wire pixels overwrite mover pixels when both masks are active, matching
    the wire-priority convention used by the project.
    """
    masks_dir = Path(masks_dir)
    extensions = list(extensions)

    indexed_path = _find_optional_file(masks_dir, sample_id, extensions)
    if indexed_path is not None:
        mask = _read_single_channel_mask(indexed_path).astype(np.int64)
        valid_values = set(range(num_classes)) | {ignore_index}
        invalid = sorted(set(np.unique(mask).tolist()) - valid_values)
        if invalid:
            raise ValueError(
                f"Invalid class IDs {invalid} in indexed mask {indexed_path}"
            )
        return mask, {
            "format": "indexed",
            "indexed_path": str(indexed_path),
            "wire_mover_overlap_pixels": 0,
            "background_conflict_pixels": 0,
            "uncovered_pixels": 0,
        }

    suffixes = {
        "background": "background",
        "mover": "movers",
        "wire": "wires",
    }
    paths: dict[str, Path] = {}
    for key, suffix in suffixes.items():
        path = _find_optional_file(
            masks_dir,
            f"{sample_id}_{suffix}",
            extensions,
        )
        if path is None:
            raise FileNotFoundError(
                f"Missing '{key}' mask for '{sample_id}'. Expected a file like "
                f"'{sample_id}_{suffix}.png' in {masks_dir}"
            )
        paths[key] = path

    background_raw = _read_single_channel_mask(paths["background"])
    mover_raw = _read_single_channel_mask(paths["mover"])
    wire_raw = _read_single_channel_mask(paths["wire"])

    shapes = {
        "background": background_raw.shape,
        "mover": mover_raw.shape,
        "wire": wire_raw.shape,
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(
            f"Mask shape mismatch for {sample_id}: {shapes}"
        )

    background = background_raw > 0
    mover = mover_raw > 0
    wire = wire_raw > 0

    class_map = np.zeros(background.shape, dtype=np.int64)
    class_map[mover] = 2
    class_map[wire] = 1  # Wire priority over mover.

    foreground = np.logical_or(mover, wire)
    coverage = np.logical_or(background, foreground)

    info = {
        "format": "three_binary_masks",
        "background_path": str(paths["background"]),
        "mover_path": str(paths["mover"]),
        "wire_path": str(paths["wire"]),
        "wire_mover_overlap_pixels": int(np.logical_and(wire, mover).sum()),
        "background_conflict_pixels": int(
            np.logical_and(background, foreground).sum()
        ),
        "uncovered_pixels": int((~coverage).sum()),
    }
    return class_map, info


def build_train_transform(
    height: int,
    width: int,
    seed: int | None = None,
) -> A.Compose:
    return A.Compose([
        A.PadIfNeeded(
            min_height=height,
            min_width=width,
            border_mode=cv2.BORDER_REFLECT_101,
        ),
        A.RandomCrop(height=height, width=width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=(-10, 10),
            shear=(-3, 3),
            p=0.5,
        ),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ], seed=seed)

def build_eval_transform() -> A.Compose:
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_split_directories(dataset_config: dict, split: str) -> tuple[Path, Path]:
    if split in {"train", "val"}:
        return (
            Path(dataset_config["development_images_dir"]),
            Path(dataset_config["development_masks_dir"]),
        )
    if split == "test":
        return (
            Path(dataset_config["test_images_dir"]),
            Path(dataset_config["test_masks_dir"]),
        )
    raise ValueError(f"Unsupported split: {split}")


class SemanticSegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        ids: list[str],
        image_extensions: list[str],
        transform: A.Compose,
        num_classes: int,
        ignore_index: int = 255,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.ids = list(ids)
        self.extensions = list(image_extensions)
        self.transform = transform
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        sample_id = self.ids[index]
        image_path = find_file(self.images_dir, sample_id, self.extensions)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask, _ = load_semantic_mask(
            self.masks_dir,
            sample_id,
            self.extensions,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch for {sample_id}: "
                f"{image.shape[:2]} vs {mask.shape[:2]}"
            )

        transformed = self.transform(image=image, mask=mask)
        return (
            transformed["image"].float(),
            transformed["mask"].long(),
            sample_id,
        )
