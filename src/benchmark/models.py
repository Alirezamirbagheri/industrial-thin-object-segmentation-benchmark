from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class SegFormerWrapper(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        checkpoint = "nvidia/mit-b0"
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig(num_labels=num_classes)
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=x).logits
        return F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class HRNetSegmentationWrapper(nn.Module):
    """Lightweight semantic-segmentation head on a timm HRNet backbone."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        backbone_name: str = "hrnet_w18_small_v2.ms_in1k",
        projection_channels: int = 48,
        fusion_channels: int = 128,
    ):
        super().__init__()
        import timm

        candidates = [
            backbone_name,
            "hrnet_w18_small_v2",
            "hrnet_w18_small.ms_in1k",
            "hrnet_w18_small",
        ]
        last_error = None
        self.backbone = None
        for candidate in candidates:
            try:
                self.backbone = timm.create_model(
                    candidate,
                    pretrained=pretrained,
                    features_only=True,
                )
                self.backbone_name = candidate
                break
            except Exception as exc:
                last_error = exc
        if self.backbone is None:
            raise RuntimeError(
                f"Could not create an HRNet features-only backbone. Last error: {last_error}"
            )

        channels = self.backbone.feature_info.channels()
        self.projections = nn.ModuleList([
            ConvBNReLU(ch, projection_channels, kernel_size=1)
            for ch in channels
        ])
        self.fusion = nn.Sequential(
            ConvBNReLU(projection_channels * len(channels), fusion_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(fusion_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.backbone(x)
        target_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            value = projection(feature)
            if value.shape[-2:] != target_size:
                value = F.interpolate(
                    value,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(value)
        logits = self.fusion(torch.cat(projected, dim=1))
        return F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )


class PIDNetWrapper(nn.Module):
    def __init__(
        self,
        num_classes: int,
        repository_path: str | Path,
        pretrained_path: str | Path | None = None,
    ):
        super().__init__()
        repo = Path(repository_path).resolve()
        if not (repo / "models" / "pidnet.py").exists():
            raise FileNotFoundError(
                "PIDNet official repository was not found. Run:\n"
                "  python setup_pidnet.py\n"
                f"Expected location: {repo}"
            )

        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        module = importlib.import_module("models.pidnet")
        self.model = module.get_pred_model("pidnet_s", num_classes=num_classes)

        if pretrained_path:
            self._load_pretrained(Path(pretrained_path))
        else:
            warnings.warn(
                "PIDNet-S is being initialized without ImageNet-pretrained weights. "
                "For a paper-level fair comparison, provide the official pretrained checkpoint.",
                RuntimeWarning,
            )

    def _load_pretrained(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"PIDNet pretrained checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        current = self.model.state_dict()
        cleaned = {}
        for key, value in state.items():
            candidates = [
                key,
                key.removeprefix("module."),
                key.removeprefix("model."),
                key[6:] if len(key) > 6 else key,
            ]
            for candidate in candidates:
                if candidate in current and current[candidate].shape == value.shape:
                    cleaned[candidate] = value
                    break
        current.update(cleaned)
        self.model.load_state_dict(current, strict=False)
        if not cleaned:
            warnings.warn(
                f"No matching PIDNet pretrained tensors were loaded from {path}.",
                RuntimeWarning,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        logits = self.model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[1] if len(logits) > 1 else logits[0]
        return F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    model_options: dict[str, Any] | None = None,
) -> nn.Module:
    name = name.lower()
    options = dict(model_options or {})

    if name == "unet_resnet34":
        import segmentation_models_pytorch as smp
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )

    if name == "unetplusplus_resnet34":
        import segmentation_models_pytorch as smp
        return smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )

    if name == "deeplabv3plus_resnet34":
        import segmentation_models_pytorch as smp
        return smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
            encoder_output_stride=16,
        )

    if name == "hrnet_w18_small_v2":
        return HRNetSegmentationWrapper(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone_name=options.get(
                "backbone_name",
                "hrnet_w18_small_v2.ms_in1k",
            ),
        )

    if name == "segformer_b0":
        return SegFormerWrapper(num_classes=num_classes, pretrained=pretrained)

    if name == "pidnet_s":
        repository_path = options.get(
            "repository_path",
            Path(__file__).resolve().parents[1] / "third_party" / "PIDNet",
        )
        return PIDNetWrapper(
            num_classes=num_classes,
            repository_path=repository_path,
            pretrained_path=options.get("pretrained_path") or None,
        )

    if name == "yolo26s_sem":
        from .yolo26_semantic_model import YOLO26SemanticBenchmarkModel
        return YOLO26SemanticBenchmarkModel(
            num_classes=num_classes,
            pretrained=pretrained,
            pretrained_path=options.get("pretrained_path", "yolo26s-sem.pt"),
            model_yaml=options.get("model_yaml", "yolo26s-sem.yaml"),
        )

    raise ValueError(f"Unknown model: {name}")


MODEL_METADATA = {
    "yolo26s_sem": {"display_name": "YOLO26s-Sem", "family": "Real-time semantic segmentation"},
    "unet_resnet34": {"display_name": "U-Net", "family": "Encoder-decoder CNN"},
    "unetplusplus_resnet34": {"display_name": "U-Net++", "family": "Nested encoder-decoder CNN"},
    "deeplabv3plus_resnet34": {"display_name": "DeepLabV3+", "family": "Multi-scale context CNN"},
    "hrnet_w18_small_v2": {"display_name": "HRNet-W18-Small-v2", "family": "High-resolution CNN"},
    "segformer_b0": {"display_name": "SegFormer-B0", "family": "Transformer segmentation"},
    "pidnet_s": {"display_name": "PIDNet-S", "family": "Real-time semantic segmentation"},
}
