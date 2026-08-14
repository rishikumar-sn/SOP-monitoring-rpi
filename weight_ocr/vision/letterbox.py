from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    original_width: int
    original_height: int
    target_width: int
    target_height: int
    scale: float
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    def roi_to_model_point(self, x: float, y: float):
        return (
            float(x) * self.scale + self.pad_left,
            float(y) * self.scale + self.pad_top,
        )

    def model_to_roi_point(self, x: float, y: float):
        return (
            (float(x) - self.pad_left) / self.scale,
            (float(y) - self.pad_top) / self.scale,
        )

    def as_dict(self):
        return asdict(self)


def letterbox(image, target_width: int = 512, target_height: int = 512, pad_value=114):
    if image is None or image.size == 0:
        raise ValueError("Letterbox input image is empty")
    if image.ndim not in (2, 3):
        raise ValueError(f"Letterbox expects a 2D or 3D image, received {image.shape}")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Letterbox target dimensions must be positive")

    original_height, original_width = image.shape[:2]
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    horizontal_padding = target_width - resized_width
    vertical_padding = target_height - resized_height
    pad_left = horizontal_padding // 2
    pad_top = vertical_padding // 2
    pad_right = horizontal_padding - pad_left
    pad_bottom = vertical_padding - pad_top

    output_shape = (target_height, target_width) + image.shape[2:]
    model_image = np.empty(output_shape, dtype=image.dtype)
    model_image[...] = pad_value
    model_image[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized

    transform = LetterboxTransform(
        original_width=original_width,
        original_height=original_height,
        target_width=target_width,
        target_height=target_height,
        scale=scale,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )
    return model_image, transform
