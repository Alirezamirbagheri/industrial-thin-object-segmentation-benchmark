from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from benchmark.common import load_yaml
from benchmark.dataset import (
    SemanticSegmentationDataset,
    build_eval_transform,
    build_train_transform,
    get_split_directories,
    read_id_list,
)
from benchmark.losses import CombinedLoss
from benchmark.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a segmentation benchmark model.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--model", required=True, help="Model key, e.g. unet_resnet34")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg.get("experiment", {}).get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    loss_cfg = cfg.get("loss", {})
    num_classes = int(data_cfg.get("num_classes", 3))
    image_extensions = data_cfg.get("image_extensions", [".png", ".jpg", ".jpeg", ".bmp"])

    train_images, train_masks = get_split_directories(data_cfg, "train")
    val_images, val_masks = get_split_directories(data_cfg, "val")
    train_ids = read_id_list(data_cfg["train_ids"])
    val_ids = read_id_list(data_cfg["val_ids"])

    crop_h = int(train_cfg.get("crop_height", 1024))
    crop_w = int(train_cfg.get("crop_width", 1024))
    train_dataset = SemanticSegmentationDataset(
        train_images, train_masks, train_ids, image_extensions,
        build_train_transform(crop_h, crop_w, seed=seed), num_classes=num_classes,
    )
    val_dataset = SemanticSegmentationDataset(
        val_images, val_masks, val_ids, image_extensions,
        build_eval_transform(), num_classes=num_classes,
    )

    batch_size = int(train_cfg.get("batch_size", 2))
    num_workers = int(train_cfg.get("num_workers", 4))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg.get("model", {})
    model = build_model(
        args.model,
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", True)),
        model_options=model_cfg.get("options", {}),
    ).to(device)

    class_weights = loss_cfg.get("class_weights", [0.1, 10.0, 1.0])
    if len(class_weights) != num_classes:
        raise ValueError("loss.class_weights must contain one value per class")
    criterion = CombinedLoss(
        num_classes=num_classes,
        class_weights=class_weights,
        wire_class_id=int(loss_cfg.get("wire_class_id", 1)),
        profile=loss_cfg.get("profile", {"ce_weight": 1.0, "dice_weight": 1.0}),
        ignore_index=int(loss_cfg.get("ignore_index", 255)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 100))
    output_dir = Path(cfg.get("experiment", {}).get("output_dir", "outputs/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks, _ in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for images, masks, _ in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                val_loss += float(criterion(model(images), masks))

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": args.model,
                "num_classes": num_classes,
                "state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            }, output_dir / "best.pt")


if __name__ == "__main__":
    main()
