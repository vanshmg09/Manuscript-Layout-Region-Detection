"""
I/O and geometric utility functions for Manuscript Layout Region Detection.
Provides Unicode/Windows-safe image reading/writing, robust path resolution,
bounding box clipping, and JSON serialization.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder converting NumPy scalars and arrays to native Python types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def read_image(path: Union[str, Path]) -> np.ndarray:
    """
    Safely read an image from disk supporting Unicode and Windows paths.

    Standard cv2.imread fails on Windows when paths contain non-ASCII characters.
    This function reads raw bytes via standard Python I/O and decodes via cv2.imdecode.

    Args:
        path: Path to the target image file (str or Path).

    Returns:
        np.ndarray: BGR image array.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file cannot be decoded as an image or is empty.
    """
    str_path = str(os.path.abspath(path))
    if not os.path.isfile(str_path):
        raise FileNotFoundError(f"Input image not found: {str_path}")

    try:
        with open(str_path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        if file_bytes.size == 0:
            raise ValueError(f"Image file is zero bytes / corrupted: {str_path}")

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError(f"Failed to decode image data (unsupported or corrupted format): {str_path}")
        return img
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Error reading image {str_path}: {e}") from e


def write_image(path: Union[str, Path], img: np.ndarray) -> str:
    """
    Safely write an image to disk supporting Unicode and Windows paths.

    Args:
        path: Destination file path.
        img: Image numpy array (BGR).

    Returns:
        str: Absolute path of the saved file.

    Raises:
        ValueError: If encoding fails or input image is invalid.
    """
    str_path = str(os.path.abspath(path))
    os.makedirs(os.path.dirname(str_path), exist_ok=True)

    ext = os.path.splitext(str_path)[1]
    if not ext:
        ext = ".png"
        str_path += ext

    success, encoded = cv2.imencode(ext, img)
    if not success:
        raise ValueError(f"Failed to encode image to {ext} format: {str_path}")

    with open(str_path, "wb") as f:
        f.write(encoded.tobytes())

    return str_path


def find_images(
    input_path: Union[str, Path],
    supported_extensions: Sequence[str] = (
        ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"
    ),
) -> List[str]:
    """
    Find image file(s) from a single path or directory.
    Resolves relative paths to absolute paths.

    Args:
        input_path: Single image file or directory containing images.
        supported_extensions: Sequence of valid image extensions (case-insensitive).

    Returns:
        List[str]: Sorted list of absolute file paths.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If no valid image files are found.
    """
    abs_path = os.path.abspath(input_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path} (resolved to {abs_path})")

    valid_exts = tuple(ext.lower() for ext in supported_extensions)

    if os.path.isfile(abs_path):
        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in valid_exts:
            raise ValueError(
                f"File '{abs_path}' has unsupported extension '{ext}'. "
                f"Supported: {valid_exts}"
            )
        return [abs_path]

    if os.path.isdir(abs_path):
        image_files: List[str] = []
        for root, _, files in os.walk(abs_path):
            for file in sorted(files):
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    image_files.append(os.path.abspath(os.path.join(root, file)))

        if not image_files:
            raise ValueError(
                f"No supported images found in directory: {abs_path} "
                f"with extensions {valid_exts}"
            )
        return sorted(image_files)

    raise ValueError(f"Path is neither a regular file nor a directory: {abs_path}")


def clip_bbox(
    bbox: Sequence[Union[int, float]],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """
    Strictly clips and validates a bounding box inside image boundaries:
    [0 <= x_min < x_max <= W, 0 <= y_min < y_max <= H].

    Args:
        bbox: Sequence of [x_min, y_min, x_max, y_max].
        width: Image width (W).
        height: Image height (H).

    Returns:
        Tuple[int, int, int, int]: Strictly clipped (x_min, y_min, x_max, y_max).

    Raises:
        ValueError: If width or height is non-positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: W={width}, H={height}")

    x_min, y_min, x_max, y_max = bbox

    # Convert to integers
    x1 = int(round(x_min))
    y1 = int(round(y_min))
    x2 = int(round(x_max))
    y2 = int(round(y_max))

    # Order properly if inverted
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # Clamp boundaries
    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))

    return int(x1), int(y1), int(x2), int(y2)


def compute_iou(
    box_a: Sequence[Union[int, float]],
    box_b: Sequence[Union[int, float]],
) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2].

    Args:
        box_a: First bounding box.
        box_b: Second bounding box.

    Returns:
        float: IoU value in range [0.0, 1.0].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def save_json(data: Dict[str, Any], output_path: Union[str, Path]) -> str:
    """
    Save dictionary data to a nicely formatted JSON file.

    Args:
        data: Python dictionary to serialize.
        output_path: Path where JSON will be saved.

    Returns:
        str: Absolute path of written JSON file.
    """
    str_path = str(os.path.abspath(output_path))
    os.makedirs(os.path.dirname(str_path), exist_ok=True)

    with open(str_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)

    return str_path

