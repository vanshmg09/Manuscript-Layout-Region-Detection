"""
Manuscript Layout Region Detection Package.
Provides end-to-end modules for historical manuscript layout analysis:
- Preprocessing (illumination flattening, hysteresis binarization, substrate masking, deskewing)
- Region Detection (adaptive character units, line grouping, block clustering, multi-column splitting)
- Region Classification (relative spatial anchoring across 5 target layout classes)
- Visualization (color-coded bounding boxes and badges)
- Safe I/O and utility functions
"""

from config import (
    CLASS_FILLER,
    CLASS_FOOTER,
    CLASS_HEADER,
    CLASS_MAIN_TEXT,
    CLASS_SIDE_TEXT,
    DEFAULT_CONFIG,
    PipelineConfig,
    TARGET_CLASSES,
)
from src.preprocessing import PreprocessingResult, Preprocessor
from src.region_classifier import ClassifiedRegion, RegionClassifier
from src.region_detector import DetectedRegion, RegionDetector
from src.utils import (
    clip_bbox,
    compute_iou,
    find_images,
    read_image,
    save_json,
    write_image,
)
from src.visualization import Visualizer

__all__ = [
    "CLASS_HEADER",
    "CLASS_FOOTER",
    "CLASS_MAIN_TEXT",
    "CLASS_SIDE_TEXT",
    "CLASS_FILLER",
    "TARGET_CLASSES",
    "DEFAULT_CONFIG",
    "PipelineConfig",
    "Preprocessor",
    "PreprocessingResult",
    "RegionDetector",
    "DetectedRegion",
    "RegionClassifier",
    "ClassifiedRegion",
    "Visualizer",
    "read_image",
    "write_image",
    "find_images",
    "save_json",
    "clip_bbox",
    "compute_iou",
]

