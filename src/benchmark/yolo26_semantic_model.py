from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.tasks import SemanticSegmentationModel


class YOLO26SemanticBenchmarkModel(nn.Module):
    """
    YOLO26 semantic segmentation adapted to the common benchmark interface.

    The Ultralytics model returns:
      eval:  main_logits
      train: (main_logits, auxiliary_logits)

    The common benchmark uses only the main prediction and applies the same
    external CombinedLoss used for all other architectures.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        pretrained_path: str = "yolo26s-sem.pt",
        model_yaml: str = "yolo26s-sem.yaml",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.network = SemanticSegmentationModel(
            cfg=model_yaml,
            ch=3,
            nc=self.num_classes,
            verbose=False,
        )
        if pretrained:
            pretrained_model = YOLO(pretrained_path).model
            self.network.load(pretrained_model, verbose=True)
            del pretrained_model

    @staticmethod
    def _main_logits(output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            if not output:
                raise RuntimeError("YOLO26 returned an empty output.")
            if not torch.is_tensor(output[0]):
                raise TypeError(
                    "YOLO26 main output is not a tensor: "
                    f"{type(output[0]).__name__}"
                )
            return output[0]
        raise TypeError(
            "Unsupported YOLO26 output type: "
            f"{type(output).__name__}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x)
        logits = self._main_logits(output)
        if logits.ndim != 4:
            raise RuntimeError(
                "Expected YOLO26 logits in BCHW format, "
                f"received shape={tuple(logits.shape)}"
            )
        if logits.shape[1] != self.num_classes:
            raise RuntimeError(
                f"Expected {self.num_classes} output classes, "
                f"received {logits.shape[1]}"
            )
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits
