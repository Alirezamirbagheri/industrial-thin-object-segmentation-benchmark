from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize


def safe_div(a: float, b: float, empty_value: float = float("nan")) -> float:
    return float(a / b) if b > 0 else float(empty_value)


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return tp, fp, fn, tn


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp, fp, fn, _ = confusion_counts(pred, gt)
    return {
        "precision": safe_div(tp, tp + fp, 1.0 if not gt.any() else 0.0),
        "recall": safe_div(tp, tp + fn, 1.0 if not gt.any() else 0.0),
        "dice": safe_div(2 * tp, 2 * tp + fp + fn, 1.0),
        "iou": safe_div(tp, tp + fp + fn, 1.0),
    }


def boundary_map(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    return np.logical_xor(mask, binary_erosion(mask))


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance_px: int = 3) -> float:
    pb = boundary_map(pred)
    gb = boundary_map(gt)
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0
    d_to_gt = distance_transform_edt(~gb)
    d_to_pred = distance_transform_edt(~pb)
    precision = float((d_to_gt[pb] <= tolerance_px).mean())
    recall = float((d_to_pred[gb] <= tolerance_px).mean())
    return safe_div(2 * precision * recall, precision + recall, 0.0)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pb = boundary_map(pred)
    gb = boundary_map(gt)
    if not pb.any() and not gb.any():
        return 0.0
    if not pb.any() or not gb.any():
        return float("nan")
    d_to_gt = distance_transform_edt(~gb)[pb]
    d_to_pred = distance_transform_edt(~pb)[gb]
    return float(np.percentile(np.concatenate([d_to_gt, d_to_pred]), 95))


def skeleton_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return {
            "cldice": 1.0,
            "skeleton_precision": 1.0,
            "skeleton_recall": 1.0,
        }
    if not pred.any() or not gt.any():
        return {
            "cldice": 0.0,
            "skeleton_precision": 0.0,
            "skeleton_recall": 0.0,
        }

    skel_pred = skeletonize(pred)
    skel_gt = skeletonize(gt)
    topology_precision = safe_div(
        np.logical_and(skel_pred, gt).sum(),
        skel_pred.sum(),
        0.0,
    )
    topology_sensitivity = safe_div(
        np.logical_and(skel_gt, pred).sum(),
        skel_gt.sum(),
        0.0,
    )
    score = safe_div(
        2 * topology_precision * topology_sensitivity,
        topology_precision + topology_sensitivity,
        0.0,
    )
    return {
        "cldice": score,
        "skeleton_precision": topology_precision,
        "skeleton_recall": topology_sensitivity,
    }


def connected_component_stats(mask: np.ndarray, min_area: int) -> dict[str, float]:
    mask_u8 = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    areas = [
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_area
    ]
    total = int(mask_u8.sum())
    largest_fraction = (
        float(max(areas) / total)
        if total > 0 and areas
        else (1.0 if total == 0 else 0.0)
    )
    return {
        "component_count": len(areas),
        "largest_component_fraction": largest_fraction,
    }


def endpoint_count(mask: np.ndarray) -> int:
    skeleton = skeletonize(mask.astype(bool)).astype(np.uint8)
    if not skeleton.any():
        return 0
    neighbours = cv2.filter2D(
        skeleton,
        ddepth=cv2.CV_16S,
        kernel=np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    ) - skeleton
    return int(np.logical_and(skeleton == 1, neighbours == 1).sum())


def evaluate_class(
    pred_classmap: np.ndarray,
    gt_classmap: np.ndarray,
    class_id: int,
    boundary_tolerance_px: int,
    min_component_area_px: int,
) -> dict[str, float]:
    pred = pred_classmap == class_id
    gt = gt_classmap == class_id
    out = binary_metrics(pred, gt)
    out["boundary_f1"] = boundary_f1(pred, gt, boundary_tolerance_px)
    out["hd95_px"] = hd95(pred, gt)
    out.update(skeleton_metrics(pred, gt))

    pred_components = connected_component_stats(pred, min_component_area_px)
    gt_components = connected_component_stats(gt, min_component_area_px)
    pred_endpoints = endpoint_count(pred)
    gt_endpoints = endpoint_count(gt)

    out.update({
        "pred_component_count": pred_components["component_count"],
        "gt_component_count": gt_components["component_count"],
        "component_count_error": (
            pred_components["component_count"] - gt_components["component_count"]
        ),
        "fragmentation_excess": max(
            0,
            pred_components["component_count"] - gt_components["component_count"],
        ),
        "pred_largest_component_fraction": pred_components[
            "largest_component_fraction"
        ],
        "gt_largest_component_fraction": gt_components[
            "largest_component_fraction"
        ],
        "pred_endpoint_count": pred_endpoints,
        "gt_endpoint_count": gt_endpoints,
        "endpoint_count_error": pred_endpoints - gt_endpoints,
        "absolute_endpoint_count_error": abs(pred_endpoints - gt_endpoints),
        "gt_pixels": int(gt.sum()),
        "pred_pixels": int(pred.sum()),
    })
    return out
