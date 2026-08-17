from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def _orientation_deg(points_xy: np.ndarray) -> float:
    if len(points_xy) < 2:
        return float("nan")
    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    angle = np.degrees(np.arctan2(major[1], major[0])) % 180.0
    return float(angle)


def _angle_error_deg(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    difference = abs(a - b) % 180.0
    return float(min(difference, 180.0 - difference))


def _extract_components(mask: np.ndarray, min_area: int) -> list[dict]:
    mask_u8 = mask.astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        ys, xs = np.where(labels == label)
        points = np.column_stack([xs, ys]).astype(np.float64)
        components.append({
            "label": label,
            "area": area,
            "centroid": centroids[label].astype(np.float64),
            "orientation_deg": _orientation_deg(points),
            "width": int(stats[label, cv2.CC_STAT_WIDTH]),
            "height": int(stats[label, cv2.CC_STAT_HEIGHT]),
        })
    return components


def transform_points_homography(
    points_xy: np.ndarray,
    homography: np.ndarray | None,
) -> np.ndarray | None:
    if homography is None or len(points_xy) == 0:
        return None
    points = np.asarray(points_xy, dtype=np.float64)
    ones = np.ones((len(points), 1), dtype=np.float64)
    homogeneous = np.concatenate([points, ones], axis=1)
    mapped = (np.asarray(homography, dtype=np.float64) @ homogeneous.T).T
    valid = np.abs(mapped[:, 2]) > 1e-12
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    result[valid] = mapped[valid, :2] / mapped[valid, 2:3]
    return result


def evaluate_mover_geometry(
    pred_classmap: np.ndarray,
    gt_classmap: np.ndarray,
    mover_class_id: int,
    min_area_px: int,
    max_match_distance_px: float,
    homography_px_to_mm: np.ndarray | None = None,
) -> dict[str, float]:
    pred_components = _extract_components(
        pred_classmap == mover_class_id,
        min_area_px,
    )
    gt_components = _extract_components(
        gt_classmap == mover_class_id,
        min_area_px,
    )

    result = {
        "gt_mover_count": len(gt_components),
        "pred_mover_count": len(pred_components),
        "matched_mover_count": 0,
        "mover_detection_precision": 0.0 if pred_components else (
            1.0 if not gt_components else 0.0
        ),
        "mover_detection_recall": 0.0 if gt_components else 1.0,
        "center_error_px_mean": float("nan"),
        "center_error_px_p95": float("nan"),
        "center_error_px_max": float("nan"),
        "center_error_mm_mean": float("nan"),
        "orientation_error_deg_mean": float("nan"),
        "orientation_error_deg_p95": float("nan"),
        "area_relative_error_mean": float("nan"),
    }

    if not pred_components or not gt_components:
        return result

    pred_centers = np.stack([x["centroid"] for x in pred_components])
    gt_centers = np.stack([x["centroid"] for x in gt_components])
    distances = np.linalg.norm(
        gt_centers[:, None, :] - pred_centers[None, :, :],
        axis=2,
    )
    gt_indices, pred_indices = linear_sum_assignment(distances)

    matches = [
        (g, p)
        for g, p in zip(gt_indices, pred_indices)
        if distances[g, p] <= max_match_distance_px
    ]
    if not matches:
        return result

    center_errors = np.array(
        [distances[g, p] for g, p in matches],
        dtype=np.float64,
    )
    orientation_errors = np.array([
        _angle_error_deg(
            gt_components[g]["orientation_deg"],
            pred_components[p]["orientation_deg"],
        )
        for g, p in matches
    ], dtype=np.float64)
    area_errors = np.array([
        abs(pred_components[p]["area"] - gt_components[g]["area"])
        / max(gt_components[g]["area"], 1)
        for g, p in matches
    ], dtype=np.float64)

    result.update({
        "matched_mover_count": len(matches),
        "mover_detection_precision": len(matches) / len(pred_components),
        "mover_detection_recall": len(matches) / len(gt_components),
        "center_error_px_mean": float(center_errors.mean()),
        "center_error_px_p95": float(np.percentile(center_errors, 95)),
        "center_error_px_max": float(center_errors.max()),
        "orientation_error_deg_mean": float(np.nanmean(orientation_errors)),
        "orientation_error_deg_p95": float(
            np.nanpercentile(orientation_errors, 95)
        ),
        "area_relative_error_mean": float(area_errors.mean()),
    })

    if homography_px_to_mm is not None:
        gt_points = np.stack([gt_components[g]["centroid"] for g, _ in matches])
        pred_points = np.stack([pred_components[p]["centroid"] for _, p in matches])
        gt_mm = transform_points_homography(gt_points, homography_px_to_mm)
        pred_mm = transform_points_homography(pred_points, homography_px_to_mm)
        if gt_mm is not None and pred_mm is not None:
            mm_errors = np.linalg.norm(gt_mm - pred_mm, axis=1)
            result["center_error_mm_mean"] = float(np.nanmean(mm_errors))

    return result
