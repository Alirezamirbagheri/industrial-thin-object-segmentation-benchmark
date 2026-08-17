from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    start = 1 if ignore_background else 0
    probs = probs[:, start:]
    one_hot = one_hot[:, start:]
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    denominator = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] == 1:
        return -F.max_pool2d(-image, kernel_size=3, stride=1, padding=1)
    p1 = -F.max_pool2d(-image, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-image, (1, 3), (1, 1), (0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, kernel_size=3, stride=1, padding=1)


def soft_open(image: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(image))


def soft_skeletonize(image: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    image = image.clamp(0, 1)
    opened = soft_open(image)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = soft_open(image)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    wire_class_id: int,
    iterations: int = 20,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, wire_class_id:wire_class_id + 1]
    gt = (target == wire_class_id).float().unsqueeze(1)
    skel_pred = soft_skeletonize(probs, iterations)
    skel_gt = soft_skeletonize(gt, iterations)
    topology_precision = ((skel_pred * gt).sum(dim=(1, 2, 3)) + eps) / (skel_pred.sum(dim=(1, 2, 3)) + eps)
    topology_sensitivity = ((skel_gt * probs).sum(dim=(1, 2, 3)) + eps) / (skel_gt.sum(dim=(1, 2, 3)) + eps)
    cldice = 2.0 * topology_precision * topology_sensitivity / (topology_precision + topology_sensitivity + eps)
    return 1.0 - cldice.mean()


def soft_boundary_map(value: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(value, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-value, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0, 1)


def soft_boundary_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    wire_class_id: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, wire_class_id:wire_class_id + 1]
    gt = (target == wire_class_id).float().unsqueeze(1)
    pred_boundary = soft_boundary_map(probs)
    gt_boundary = soft_boundary_map(gt)
    intersection = (pred_boundary * gt_boundary).sum(dim=(1, 2, 3))
    denominator = pred_boundary.sum(dim=(1, 2, 3)) + gt_boundary.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int, class_weights: list[float], wire_class_id: int, profile: dict, ignore_index: int = 255):
        super().__init__()
        self.num_classes = int(num_classes)
        self.wire_class_id = int(wire_class_id)
        self.ignore_index = int(ignore_index)
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.ce_weight = float(profile.get("ce_weight", 1.0))
        self.dice_weight = float(profile.get("dice_weight", 0.0))
        self.cldice_weight = float(profile.get("cldice_weight", 0.0))
        self.boundary_weight = float(profile.get("boundary_weight", 0.0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = logits.new_tensor(0.0)
        if self.ce_weight:
            total = total + self.ce_weight * F.cross_entropy(logits, target, weight=self.class_weights, ignore_index=self.ignore_index)
        if self.dice_weight:
            total = total + self.dice_weight * multiclass_dice_loss(logits, target, self.num_classes, ignore_background=True)
        if self.cldice_weight:
            total = total + self.cldice_weight * soft_cldice_loss(logits, target, self.wire_class_id)
        if self.boundary_weight:
            total = total + self.boundary_weight * soft_boundary_dice_loss(logits, target, self.wire_class_id)
        return total
