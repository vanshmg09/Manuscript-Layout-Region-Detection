"""
Centralized Configuration for Manuscript Specific Layout Region Detection.
Defines parameters for preprocessing, region detection, classification,
visualization, and supported classes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ==============================================================================
# Target Layout Classes
# ==============================================================================
CLASS_HEADER = "header"
CLASS_FOOTER = "footer"
CLASS_MAIN_TEXT = "main_text"
CLASS_SIDE_TEXT = "side_text"
CLASS_FILLER = "filler"

TARGET_CLASSES: List[str] = [
    CLASS_HEADER,
    CLASS_FOOTER,
    CLASS_MAIN_TEXT,
    CLASS_SIDE_TEXT,
    CLASS_FILLER,
]


# ==============================================================================
# Visualization Palette (BGR format for OpenCV, RGB for Pillow/Docs)
# ==============================================================================
# High-contrast, publication-grade color scheme
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    CLASS_HEADER: (71, 160, 67),     # Emerald Green
    CLASS_FOOTER: (170, 36, 142),    # Royal Violet
    CLASS_MAIN_TEXT: (235, 145, 30), # Vibrant Ocean Blue
    CLASS_SIDE_TEXT: (0, 140, 255),  # Amber Orange
    CLASS_FILLER: (53, 57, 229),     # Crimson Red
}

CLASS_COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    cls_name: (bgr[2], bgr[1], bgr[0])
    for cls_name, bgr in CLASS_COLORS_BGR.items()
}


# ==============================================================================
# Preprocessing Configuration
# ==============================================================================
@dataclass
class PreprocessingConfig:
    # Background estimation for illumination flattening
    bg_blur_ratio: float = 0.08          # Fraction of image height for Gaussian blur
    min_bg_blur_ksize: int = 31          # Minimum blur kernel size (must be odd)
    max_bg_blur_ksize: int = 151         # Maximum blur kernel size (must be odd)

    # Contrast enhancement & Denoising
    clahe_clip_limit: float = 2.0        # Contrast Limited Adaptive Histogram Equalization
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    bilateral_d: int = 5                 # Neighborhood diameter for edge-preserving filter
    bilateral_sigma_color: float = 35.0  # Filter sigma in color space
    bilateral_sigma_space: float = 35.0  # Filter sigma in coordinate space

    # Dual-scale hysteresis binarization
    hysteresis_high_ratio: float = 0.95  # Confident ink seed threshold multiplier (of Otsu)
    hysteresis_low_ratio: float = 0.60   # Faint stroke continuation multiplier (of Otsu)

    # Page substrate isolation
    substrate_min_area_ratio: float = 0.15  # Minimum fraction of page to be considered substrate
    substrate_morph_ksize: int = 15         # Kernel size to smooth substrate boundary

    # Ruling / border line suppression
    ruling_min_v_ratio: float = 0.12        # Minimum vertical line length as fraction of height
    ruling_min_h_ratio: float = 0.15        # Minimum horizontal line length as fraction of width
    ruling_max_thickness_px: int = 4        # Maximum thickness of ruling borders
    ruling_dilation_px: int = 3             # Dilation of suppressed line mask

    # Deskewing projection profile search space
    deskew_max_angle_deg: float = 12.0      # Angular search range [-deskew_max_angle, +deskew_max_angle]
    deskew_coarse_step_deg: float = 0.5     # Coarse search angular step
    deskew_fine_step_deg: float = 0.1       # Fine search angular step around coarse peak


# ==============================================================================
# Region Detector Configuration
# ==============================================================================
@dataclass
class RegionDetectorConfig:
    # Connected component speck filtering
    min_speck_area_px: int = 8              # Minimum component area to filter dust/specks
    min_speck_dimension_px: int = 3         # Minimum component width & height
    max_component_area_ratio: float = 0.85  # Reject giant non-text artifacts

    # Character unit estimation
    min_char_height_px: int = 7             # Minimum valid character stroke height
    max_char_height_ratio: float = 0.25     # Maximum character height relative to image
    min_char_width_px: int = 5              # Minimum valid character stroke width
    max_char_width_ratio: float = 0.25      # Maximum character width relative to image

    # Line grouping morphology
    line_kernel_w_multiplier: float = 2.4   # Horizontal kernel width = W_char * multiplier
    line_kernel_h_px: int = 2               # Kernel height for horizontal grouping

    # Block clustering
    block_kernel_w_multiplier: float = 1.2  # Horizontal bridge for block clustering
    block_kernel_h_multiplier: float = 1.4  # Vertical bridge kernel = H_char * multiplier
    max_line_gap_multiplier: float = 1.4    # Maximum vertical gap between stacked lines

    # Multi-column layout splitting
    enable_multi_column_splitting: bool = True
    col_valley_width_multiplier: float = 1.5   # Valley width threshold = W_char * multiplier
    col_valley_max_density: float = 0.04      # Maximum stroke density in gutter to declare a split
    min_column_width_multiplier: float = 3.5  # Minimum valid column width = W_char * multiplier


# ==============================================================================
# Region Classifier Configuration
# ==============================================================================
@dataclass
class RegionClassifierConfig:
    # Modern digitization banner / stamp heuristics (filler)
    banner_margin_ratio: float = 0.085        # Top/bottom margin zone for digitization overlays
    banner_min_aspect_ratio: float = 8.0     # Metadata banners have high aspect ratio (w/h)
    banner_max_stroke_height_ratio: float = 0.85 # Modern font height is smaller than manuscript

    # Relative spatial margin thresholds (relative to page substrate & main text body)
    header_max_y_ratio: float = 0.28         # Header must be located in top portion
    footer_min_y_ratio: float = 0.72         # Footer must be located in bottom portion
    side_text_max_width_ratio: float = 0.45  # Marginalia column is narrower than main body
    side_text_min_v_overlap: float = 0.30    # Marginalia shares vertical range with main body

    # Filler heuristics
    filler_min_faintness_stroke_density: float = 0.015  # Extremely sparse markings (pencil/noise)
    filler_off_substrate_margin_px: int = 10            # Pixels outside substrate boundary

    # Confidence calibration weights
    conf_base: float = 0.80
    conf_geometric_weight: float = 0.10
    conf_density_weight: float = 0.06
    conf_area_weight: float = 0.04


# ==============================================================================
# Visualization Configuration
# ==============================================================================
@dataclass
class VisualizationConfig:
    bbox_thickness: int = 2
    font_scale: float = 0.52
    font_thickness: int = 1
    badge_padding_x: int = 6
    badge_padding_y: int = 4
    alpha_overlay: float = 0.18              # Semi-transparent colored tint inside bounding box


# ==============================================================================
# Master Pipeline Configuration
# ==============================================================================
@dataclass
class PipelineConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    detector: RegionDetectorConfig = field(default_factory=RegionDetectorConfig)
    classifier: RegionClassifierConfig = field(default_factory=RegionClassifierConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Supported image extensions for directory traversal
    supported_extensions: Tuple[str, ...] = (
        ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"
    )


# Default global instance
DEFAULT_CONFIG = PipelineConfig()
