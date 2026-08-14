import cv2
import numpy as np


def model_corners_to_roi(model_corners, transform):
    corners = np.asarray(model_corners, dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError(f"Expected four model corners, received {corners.shape}")

    roi_corners = corners.copy()
    roi_corners[:, 0] = (roi_corners[:, 0] - transform.pad_left) / transform.scale
    roi_corners[:, 1] = (roi_corners[:, 1] - transform.pad_top) / transform.scale
    roi_corners[:, 0] = np.clip(roi_corners[:, 0], 0, transform.original_width - 1)
    roi_corners[:, 1] = np.clip(roi_corners[:, 1], 0, transform.original_height - 1)
    return roi_corners


def roi_corners_to_full(roi_corners, roi_bounds, full_image_shape):
    corners = np.asarray(roi_corners, dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError(f"Expected four ROI corners, received {corners.shape}")
    if len(roi_bounds) != 4:
        raise ValueError("ROI bounds must be (x1, y1, x2, y2)")

    x1, y1, x2, y2 = (int(value) for value in roi_bounds)
    full_height, full_width = full_image_shape[:2]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI bounds: {roi_bounds}")

    full_corners = corners + np.array((x1, y1), dtype=np.float32)
    full_corners[:, 0] = np.clip(full_corners[:, 0], 0, full_width - 1)
    full_corners[:, 1] = np.clip(full_corners[:, 1], 0, full_height - 1)
    return full_corners


def map_model_corners(model_corners, transform, roi_bounds, full_image_shape):
    x1, y1, x2, y2 = (int(value) for value in roi_bounds)
    if (
        x2 - x1 != transform.original_width
        or y2 - y1 != transform.original_height
    ):
        raise ValueError("ROI bounds do not match the letterboxed source image")
    roi_corners = model_corners_to_roi(model_corners, transform)
    full_corners = roi_corners_to_full(
        roi_corners,
        roi_bounds,
        full_image_shape,
    )
    return roi_corners, full_corners


def draw_polygon_copy(image, corners, label):
    if image is None or image.size == 0:
        raise ValueError("Cannot draw on an empty image")
    polygon = np.asarray(corners, dtype=np.float32)
    if polygon.shape != (4, 2):
        raise ValueError(f"Expected four polygon corners, received {polygon.shape}")

    output = image.copy()
    integer_polygon = np.round(polygon).astype(np.int32)
    cv2.polylines(output, [integer_polygon], True, (0, 255, 0), 3, cv2.LINE_AA)
    label_origin = (
        max(5, int(integer_polygon[:, 0].min())),
        max(24, int(integer_polygon[:, 1].min()) - 8),
    )
    cv2.putText(
        output,
        label,
        label_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        label,
        label_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return output
