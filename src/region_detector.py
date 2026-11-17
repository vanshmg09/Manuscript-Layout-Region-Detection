"""
Region Detector for Historical Manuscript Layout Analysis.
Extracts candidate text regions, lines, and blocks via:
1. Connected Component Analysis (CCA) with speck/non-text filtering.
2. Adaptive Character Unit Estimation (median stroke height, width, thickness).
3. Digitization Banner & Marginal Noise Isolation (isolating modern metadata overlays).
4. Substrate-guided Content Region Extraction (lines and blocks inside manuscript page).
5. Multi-column Splitting (vertical projection profile valley analysis).
6. Inverse Affine Coordinate Remapping to original unrotated image coordinates.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import RegionDetectorConfig
from src.preprocessing import PreprocessingResult
from src.utils import clip_bbox


@dataclass
class DetectedRegion:
    """Represents an extracted region candidate prior to semantic classification."""
    bbox: Tuple[int, int, int, int]           # [x_min, y_min, x_max, y_max] in ORIGINAL unrotated coordinates
    deskewed_bbox: Tuple[int, int, int, int]  # [x_min, y_min, x_max, y_max] in deskewed coordinates
    stroke_pixel_count: int                   # Number of foreground ink pixels inside region
    stroke_density: float                     # Ratio of ink pixels to bounding box area
    line_count: int                           # Estimated number of distinct text lines in this region
    char_height: float                        # Estimated median character height
    char_width: float                         # Estimated median character width
    is_on_substrate: bool                     # Whether the region lies on the page substrate
    is_banner: bool = False                   # True if identified as digitization banner candidate


class RegionDetector:
    """
    Adaptive region detector that leverages manuscript character morphology
    to group lines, cluster blocks, split multi-column layouts, and remap
    coordinates back to unrotated original image space.
    """

    def __init__(self, config: Optional[RegionDetectorConfig] = None) -> None:
        self.config = config or RegionDetectorConfig()

    def estimate_character_unit(
        self,
        binary_ink: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Tuple[float, float, float]:
        """
        Estimate character unit: (median_height, median_width, median_stroke_thickness).
        Robust against noise and large non-text artifacts.

        Args:
            binary_ink: Binary ink mask (255=ink).
            image_shape: (height, width).

        Returns:
            Tuple[float, float, float]: (char_height, char_width, stroke_thickness).
        """
        h, w = image_shape
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_ink, connectivity=8)

        heights: List[int] = []
        widths: List[int] = []

        min_h = self.config.min_char_height_px
        max_h = int(h * self.config.max_char_height_ratio)
        min_w = self.config.min_char_width_px
        max_w = int(w * self.config.max_char_width_ratio)
        min_area = self.config.min_speck_area_px

        for i in range(1, num_labels):
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            ca = stats[i, cv2.CC_STAT_AREA]

            if min_h <= ch <= max_h and min_w <= cw <= max_w and ca >= min_area:
                heights.append(ch)
                widths.append(cw)

        med_h = float(np.median(heights)) if heights else 18.0
        med_w = float(np.median(widths)) if widths else 16.0

        # Estimate stroke thickness via distance transform on ink mask
        dist = cv2.distanceTransform(binary_ink, cv2.DIST_L2, 3)
        nonzero_dist = dist[dist > 0]
        stroke_thickness = float(np.median(nonzero_dist) * 2.0) if nonzero_dist.size > 0 else 2.5

        return med_h, med_w, stroke_thickness

    def split_multi_columns(
        self,
        binary_block: np.ndarray,
        x_offset: int,
        y_offset: int,
        w_char: float,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Split a wide text block into multiple parallel columns using
        vertical projection profile valley/gutter analysis.

        Args:
            binary_block: Binary crop of the text block.
            x_offset: Block left coordinate in deskewed space.
            y_offset: Block top coordinate in deskewed space.
            w_char: Median character width.

        Returns:
            List[Tuple[int, int, int, int]]: List of [x1, y1, x2, y2] sub-blocks.
        """
        bh, bw = binary_block.shape
        # Do not split if block is not wide enough to contain multi-columns
        if bw < int(self.config.min_column_width_multiplier * 2.0 * w_char):
            return [(x_offset, y_offset, x_offset + bw, y_offset + bh)]

        col_proj = np.sum(binary_block > 0, axis=0).astype(np.float32)

        # Smooth vertical projection to avoid spurious word gaps
        smooth_k = max(3, int(w_char * 0.8)) | 1
        smoothed = cv2.blur(col_proj, (smooth_k, 1)).flatten()

        max_val = float(np.max(smoothed)) if smoothed.size > 0 else 0.0
        if max_val <= 0:
            return [(x_offset, y_offset, x_offset + bw, y_offset + bh)]

        threshold = max(1.0, self.config.col_valley_max_density * max_val)
        is_valley = smoothed <= threshold

        # Identify contiguous valley intervals (gutters)
        min_valley_w = max(4, int(w_char * self.config.col_valley_width_multiplier))
        valleys: List[Tuple[int, int]] = []
        in_valley = False
        v_start = 0

        for col_idx in range(bw):
            if is_valley[col_idx]:
                if not in_valley:
                    in_valley = True
                    v_start = col_idx
            else:
                if in_valley:
                    in_valley = False
                    if (col_idx - v_start) >= min_valley_w:
                        valleys.append((v_start, col_idx))

        if in_valley and (bw - v_start) >= min_valley_w:
            valleys.append((v_start, bw))

        # Filter out valleys too close to the outer margins
        min_margin = int(w_char * self.config.min_column_width_multiplier)
        valid_valleys = [v for v in valleys if v[0] >= min_margin and v[1] <= (bw - min_margin)]

        if not valid_valleys:
            return [(x_offset, y_offset, x_offset + bw, y_offset + bh)]

        # Determine split boundaries at the midpoints of valid valleys
        split_x = [0] + [int((v[0] + v[1]) / 2) for v in valid_valleys] + [bw]
        split_boxes: List[Tuple[int, int, int, int]] = []

        for i in range(len(split_x) - 1):
            sub_x1 = split_x[i]
            sub_x2 = split_x[i + 1]
            sub_crop = binary_block[:, sub_x1:sub_x2]
            if np.sum(sub_crop > 0) >= self.config.min_speck_area_px * 2:
                ink_ys, ink_xs = np.where(sub_crop > 0)
                tight_x1 = sub_x1 + int(np.min(ink_xs))
                tight_x2 = sub_x1 + int(np.max(ink_xs)) + 1
                tight_y1 = int(np.min(ink_ys))
                tight_y2 = int(np.max(ink_ys)) + 1
                split_boxes.append((
                    x_offset + tight_x1,
                    y_offset + tight_y1,
                    x_offset + tight_x2,
                    y_offset + tight_y2,
                ))

        return split_boxes if split_boxes else [(x_offset, y_offset, x_offset + bw, y_offset + bh)]

    def remap_to_original_coords(
        self,
        bbox: Tuple[int, int, int, int],
        inv_rotation_matrix: np.ndarray,
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        """
        Remap a bounding box from deskewed space back to original unrotated image coordinates.

        Args:
            bbox: [x1, y1, x2, y2] in deskewed coordinates.
            inv_rotation_matrix: 2x3 inverse affine rotation matrix.
            width: Original image width.
            height: Original image height.

        Returns:
            Tuple[int, int, int, int]: Clipped unrotated [x_min, y_min, x_max, y_max].
        """
        x1, y1, x2, y2 = bbox
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ], dtype=np.float32).reshape(-1, 1, 2)

        transformed = cv2.transform(corners, inv_rotation_matrix).reshape(-1, 2)
        min_x = float(np.min(transformed[:, 0]))
        min_y = float(np.min(transformed[:, 1]))
        max_x = float(np.max(transformed[:, 0]))
        max_y = float(np.max(transformed[:, 1]))

        return clip_bbox((min_x, min_y, max_x, max_y), width, height)

    def detect(self, prep_result: PreprocessingResult) -> List[DetectedRegion]:
        """
        Extract candidate manuscript layout regions from preprocessed data.
        Isolates banners, substrate contents, and off-substrate markings.

        Args:
            prep_result: PreprocessingResult from Preprocessor.process().

        Returns:
            List[DetectedRegion]: Detected region candidates.
        """
        h = prep_result.image_height
        w = prep_result.image_width
        binary_ink = prep_result.binary_ink_mask
        substrate_mask = prep_result.substrate_mask
        inv_rot = prep_result.inv_rotation_matrix

        # 1. Estimate adaptive character unit
        char_h, char_w, stroke_thick = self.estimate_character_unit(binary_ink, (h, w))

        # Working copy of binary ink mask for segmentation
        content_ink = binary_ink.copy()
        detected_candidates: List[Tuple[Tuple[int, int, int, int], bool, bool]] = []
        # Tuples of (deskewed_bbox, is_on_substrate, is_banner)

        # ----------------------------------------------------------------------
        # Step 2: Detect Modern Digitization Banners (filler) in Extreme Margins
        # ----------------------------------------------------------------------
        kw_line = max(5, int(char_w * 2.2))
        line_mask = cv2.morphologyEx(
            binary_ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw_line, 2))
        )
        cnts_line, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        banner_boxes: List[Tuple[int, int, int, int]] = []
        for c in cnts_line:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 20 or bh < 6:
                continue
            mid_y = by + bh / 2.0
            aspect_ratio = float(bw / max(1, bh))

            # Banner heuristic: extreme margins + wide aspect ratio or substantial width
            in_extreme_margin = (mid_y <= h * 0.09) or (mid_y >= h * 0.91)
            is_banner_shape = (bw >= w * 0.20) or (aspect_ratio >= 6.0 and bw >= 80)

            if in_extreme_margin and is_banner_shape:
                banner_boxes.append((bx, by, bx + bw, by + bh))
                # Clear banner pixels so they don't interfere with manuscript body
                content_ink[by:by + bh, bx:bx + bw] = 0

        # Cluster banner fragments in the top margin and bottom margin
        top_banners = [b for b in banner_boxes if (b[1] + b[3]) / 2.0 <= h * 0.5]
        bot_banners = [b for b in banner_boxes if (b[1] + b[3]) / 2.0 > h * 0.5]

        for b_group in [top_banners, bot_banners]:
            if b_group:
                bx1 = min(b[0] for b in b_group)
                by1 = min(b[1] for b in b_group)
                bx2 = max(b[2] for b in b_group)
                by2 = max(b[3] for b in b_group)
                detected_candidates.append(((bx1, by1, bx2, by2), False, True))

        # ----------------------------------------------------------------------
        # Step 3: Off-Substrate Flecks (e.g. background cloth flecks outside leaf)
        # ----------------------------------------------------------------------
        outside_ink = cv2.bitwise_and(content_ink, cv2.bitwise_not(substrate_mask))
        cnts_out, _ = cv2.findContours(outside_ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_out:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Only consider significant background markings (area >= 150 px)
            if bw >= 30 and bh >= 10 and cv2.contourArea(c) >= 150:
                detected_candidates.append(((bx, by, bx + bw, by + bh), False, False))
                content_ink[by:by + bh, bx:bx + bw] = 0

        # ----------------------------------------------------------------------
        # Step 4: Inside-Substrate Manuscript Content (Main Text, Headers, Marginalia)
        # ----------------------------------------------------------------------
        inside_ink = cv2.bitwise_and(content_ink, substrate_mask)

        # Block clustering inside substrate
        kw_block = max(3, int(char_w * self.config.block_kernel_w_multiplier))
        kh_block = max(5, int(char_h * self.config.block_kernel_h_multiplier))
        block_mask = cv2.morphologyEx(
            inside_ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw_block, kh_block))
        )

        cnts_in, _ = cv2.findContours(block_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        in_blocks: List[Tuple[int, int, int, int]] = []

        for c in cnts_in:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < int(char_w * 0.7) or bh < int(char_h * 0.5):
                continue

            crop = inside_ink[by:by + bh, bx:bx + bw]
            if np.sum(crop > 0) < self.config.min_speck_area_px:
                continue

            # Tighten bbox to exact ink bounds
            ys, xs = np.where(crop > 0)
            tx1 = bx + int(np.min(xs))
            tx2 = bx + int(np.max(xs)) + 1
            ty1 = by + int(np.min(ys))
            ty2 = by + int(np.max(ys)) + 1
            in_blocks.append((tx1, ty1, tx2, ty2))

        # Check for multi-column layouts on wide inside blocks
        for blk in in_blocks:
            bx1, by1, bx2, by2 = blk
            bw = bx2 - bx1
            bh = by2 - by1

            if self.config.enable_multi_column_splitting and bw >= int(self.config.min_column_width_multiplier * 2.0 * char_w):
                sub_crop = inside_ink[by1:by2, bx1:bx2]
                split_cols = self.split_multi_columns(sub_crop, bx1, by1, char_w)
                for sc in split_cols:
                    detected_candidates.append((sc, True, False))
            else:
                detected_candidates.append((blk, True, False))

        # ----------------------------------------------------------------------
        # Step 5: Construct Final DetectedRegion Objects
        # ----------------------------------------------------------------------
        detected_regions: List[DetectedRegion] = []
        for (dx1, dy1, dx2, dy2), is_on_sub, is_banner in detected_candidates:
            bw = dx2 - dx1
            bh = dy2 - dy1
            if bw <= 3 or bh <= 3:
                continue

            box_ink = binary_ink[dy1:dy2, dx1:dx2]
            ink_count = int(np.sum(box_ink > 0))
            if ink_count < self.config.min_speck_area_px and not is_banner:
                continue

            density = float(ink_count / max(1, bw * bh))

            # Estimate number of lines via horizontal projection
            row_proj = np.sum(box_ink > 0, axis=1)
            mean_proj = np.mean(row_proj) if row_proj.size > 0 else 0.0
            line_peaks = int(np.sum((row_proj > mean_proj) & (np.roll(row_proj, 1) <= mean_proj)))
            line_count = max(1, line_peaks)

            # Inverse affine coordinate remapping
            orig_bbox = self.remap_to_original_coords((dx1, dy1, dx2, dy2), inv_rot, w, h)

            detected_regions.append(
                DetectedRegion(
                    bbox=orig_bbox,
                    deskewed_bbox=(dx1, dy1, dx2, dy2),
                    stroke_pixel_count=ink_count,
                    stroke_density=density,
                    line_count=line_count,
                    char_height=char_h,
                    char_width=char_w,
                    is_on_substrate=is_on_sub,
                    is_banner=is_banner,
                )
            )

        # Sort top-to-bottom, left-to-right
        detected_regions = sorted(detected_regions, key=lambda r: (r.bbox[1], r.bbox[0]))
        return detected_regions
