#!/usr/bin/env python3
"""Create camera_matrix/dist_coeffs JSON from ChArUco calibration images."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np


def _dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("Install opencv-contrib-python for ChArUco support.")
    dictionary_id = getattr(cv2.aruco, name, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown OpenCV ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _run_camera_calibration(
    board,
    all_corners: list[np.ndarray],
    all_ids: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], str]:
    """Calibrate with either the legacy ArUco wrapper or the current core API."""
    calibrate_charuco = getattr(cv2.aruco, "calibrateCameraCharuco", None)
    if callable(calibrate_charuco):
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_charuco(
            charucoCorners=all_corners,
            charucoIds=all_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )
        return (
            float(rms),
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            "cv2.aruco.calibrateCameraCharuco",
        )

    # Some OpenCV builds provide ChArUco detection but omit the deprecated
    # calibrateCameraCharuco convenience wrapper. ChArUco IDs directly index
    # CharucoBoard.getChessboardCorners(), so build the equivalent 3D-to-2D
    # correspondences and use the standard camera calibration API.
    board_corners = np.asarray(
        board.getChessboardCorners(),
        dtype=np.float32,
    ).reshape(-1, 3)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for corners, ids in zip(all_corners, all_ids):
        corner_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        detected = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if len(corner_ids) != len(detected):
            raise RuntimeError(
                "Detected ChArUco corner and ID counts do not match."
            )
        if np.any(corner_ids < 0) or np.any(corner_ids >= len(board_corners)):
            raise RuntimeError("Detected ChArUco corner ID is outside the board.")
        object_points.append(board_corners[corner_ids].copy())
        image_points.append(detected.copy())

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    return (
        float(rms),
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
        "cv2.calibrateCamera",
    )


def calibrate(
    image_patterns: list[str],
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dictionary_name: str,
    min_views: int = 8,
    legacy_pattern: bool = False,
) -> dict:
    paths: list[Path] = []
    for pattern in image_patterns:
        paths.extend(Path(value) for value in glob.glob(pattern))
    paths = sorted({path.resolve() for path in paths if path.is_file()})
    if not paths:
        raise FileNotFoundError("No ChArUco calibration images matched.")

    dictionary = _dictionary(dictionary_name)
    board = cv2.aruco.CharucoBoard(
        (int(squares_x), int(squares_y)),
        float(square_length_mm),
        float(marker_length_mm),
        dictionary,
    )
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(bool(legacy_pattern))

    detector_parameters = cv2.aruco.DetectorParameters()
    if hasattr(detector_parameters, "cornerRefinementMethod"):
        detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(detector_parameters, "cornerRefinementWinSize"):
        detector_parameters.cornerRefinementWinSize = 5
    if hasattr(detector_parameters, "cornerRefinementMaxIterations"):
        detector_parameters.cornerRefinementMaxIterations = 50
    if hasattr(detector_parameters, "cornerRefinementMinAccuracy"):
        detector_parameters.cornerRefinementMinAccuracy = 0.01

    detector = cv2.aruco.CharucoDetector(
        board,
        cv2.aruco.CharucoParameters(),
        detector_parameters,
    )
    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    used_images: list[str] = []
    detection_summary: list[dict] = []

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            detection_summary.append(
                {"image": str(path), "markers": 0, "charuco_corners": 0, "status": "unreadable"}
            )
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        elif image_size != (gray.shape[1], gray.shape[0]):
            continue
        charuco_corners, charuco_ids, _marker_corners, _marker_ids = (
            detector.detectBoard(gray)
        )
        marker_count = 0 if _marker_ids is None else len(_marker_ids)
        corner_count = 0 if charuco_ids is None else len(charuco_ids)
        detection_summary.append(
            {
                "image": str(path),
                "markers": marker_count,
                "charuco_corners": corner_count,
                "status": "usable" if corner_count >= 6 else "rejected",
            }
        )
        print(
            f"{path.name}: markers={marker_count}, "
            f"ChArUco corners={corner_count}"
            + (" [usable]" if corner_count >= 6 else " [rejected]")
        )
        if charuco_ids is None or charuco_corners is None or corner_count < 6:
            continue
        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)
        used_images.append(str(path))

    if image_size is None or len(all_corners) < max(3, int(min_views)):
        best = sorted(
            detection_summary,
            key=lambda item: int(item["charuco_corners"]),
            reverse=True,
        )[:3]
        best_text = ", ".join(
            f"{Path(item['image']).name}={item['charuco_corners']} corners"
            for item in best
        )
        raise RuntimeError(
            f"Need at least {max(3, int(min_views))} usable ChArUco views with "
            "6 or more detected corners. "
            f"Usable: {len(all_corners)}/{len(paths)}. Best images: "
            f"{best_text or 'none'}. Check board rows/columns, dictionary, focus, "
            "glare, and whether enough of the board is visible."
        )

    (
        rms,
        camera_matrix,
        dist_coeffs,
        _rvecs,
        _tvecs,
        calibration_method,
    ) = _run_camera_calibration(
        board,
        all_corners,
        all_ids,
        image_size,
    )
    return {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "image_size": list(image_size),
        "rms_reprojection_error": float(rms),
        "calibration_method": calibration_method,
        "board": {
            "squares_x": int(squares_x),
            "squares_y": int(squares_y),
            "square_length_mm": float(square_length_mm),
            "marker_length_mm": float(marker_length_mm),
            "dictionary": dictionary_name,
            "legacy_pattern": bool(legacy_pattern),
        },
        "usable_image_count": len(used_images),
        "usable_images": used_images,
        "detection_summary": detection_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Image glob(s), e.g. captures/charuco/*.jpg")
    parser.add_argument("--output", default="camera_calibration.json")
    parser.add_argument("--squares-x", type=int, required=True)
    parser.add_argument("--squares-y", type=int, required=True)
    parser.add_argument("--square-mm", type=float, required=True)
    parser.add_argument("--marker-mm", type=float, required=True)
    parser.add_argument("--dictionary", default="DICT_5X5_50")
    parser.add_argument("--min-views", type=int, default=8)
    parser.add_argument(
        "--legacy-pattern",
        action="store_true",
        help="Use the pre-OpenCV-4.6 ChArUco layout used by older printed boards.",
    )
    args = parser.parse_args()

    result = calibrate(
        args.images,
        args.squares_x,
        args.squares_y,
        args.square_mm,
        args.marker_mm,
        args.dictionary,
        args.min_views,
        args.legacy_pattern,
    )
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Saved {output} using {result['usable_image_count']} images; "
        f"RMS={result['rms_reprojection_error']:.4f}"
    )


if __name__ == "__main__":
    main()
