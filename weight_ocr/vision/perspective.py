import cv2
import numpy as np


def order_quad_points(points):
    quad = np.asarray(points, dtype=np.float32)
    if quad.shape != (4, 2):
        raise ValueError(f"Expected four corner points, received {quad.shape}")
    if not np.isfinite(quad).all():
        raise ValueError("Corner points must be finite")
    if np.unique(quad, axis=0).shape[0] != 4:
        raise ValueError("Corner points must be unique")

    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    ordered = quad[np.argsort(angles)]
    top_left_index = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -top_left_index, axis=0)
    if cv2.contourArea(ordered) <= 1.0:
        raise ValueError("Corner points do not form a valid quadrilateral")
    return ordered.astype(np.float32)


def expand_quad_points(points, horizontal=0.0, vertical=0.0):
    if horizontal < 0.0 or vertical < 0.0:
        raise ValueError("Quad expansion margins cannot be negative")
    top_left, top_right, bottom_right, bottom_left = order_quad_points(points)
    return np.asarray(
        (
            top_left
            - horizontal * (top_right - top_left)
            - vertical * (bottom_left - top_left),
            top_right
            + horizontal * (top_right - top_left)
            - vertical * (bottom_right - top_right),
            bottom_right
            + horizontal * (bottom_right - bottom_left)
            + vertical * (bottom_right - top_right),
            bottom_left
            - horizontal * (bottom_right - bottom_left)
            + vertical * (bottom_left - top_left),
        ),
        dtype=np.float32,
    )


def rectify_lcd(
    clean_roi,
    corners,
    inner_margin=0.05,
    expand_x=0.0,
    expand_y=0.0,
):
    if clean_roi is None or clean_roi.size == 0:
        raise ValueError("Cannot rectify an empty ROI image")
    if not 0.0 <= inner_margin < 0.5:
        raise ValueError("Inner margin must be between 0.0 and 0.5")

    top_left, top_right, bottom_right, bottom_left = expand_quad_points(
        corners,
        expand_x,
        expand_y,
    )
    width = max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
    )
    height = max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right),
    )
    target_width = max(2, int(round(float(width))))
    target_height = max(2, int(round(float(height))))
    destination = np.array(
        (
            (0, 0),
            (target_width - 1, 0),
            (target_width - 1, target_height - 1),
            (0, target_height - 1),
        ),
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(
        np.array((top_left, top_right, bottom_right, bottom_left)),
        destination,
    )
    raw = cv2.warpPerspective(
        clean_roi,
        transform,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
    )
    if raw.shape[0] > raw.shape[1]:
        raw = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)

    margin_x = int(round(raw.shape[1] * inner_margin))
    margin_y = int(round(raw.shape[0] * inner_margin))
    rectified = raw[
        margin_y : raw.shape[0] - margin_y,
        margin_x : raw.shape[1] - margin_x,
    ].copy()
    if rectified.size == 0:
        raise ValueError("Inner margin removed the entire LCD image")
    return raw, rectified
