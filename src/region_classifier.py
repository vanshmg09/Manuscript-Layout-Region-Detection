"""
Region Classifier for Historical Manuscript Layout Analysis.
Classifies detected candidate regions into 5 target classes:
1. header: Top margin text, running headers, section titles, or folio numbers.
2. footer: Bottom margin text, catchwords, page numbers, or signatures.
3. main_text: Primary body of the manuscript (including multi-column text).
4. side_text: Marginalia, annotations, glosses, or commentary along page margins.
5. filler: Decorative elements, modern English digitization stamps/banners (e.g., eGangotri/CSU-Sringeri metadata overlays), pencil marks, and background noise.

Applies relative spatial anchoring against the primary text body and page substrate,
combined with calibrated confidence scoring.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import (
    CLASS_FILLER,
    CLASS_FOOTER,
    CLASS_HEADER,
    CLASS_MAIN_TEXT,
    CLASS_SIDE_TEXT,
    RegionClassifierConfig,
)
from src.preprocessing import PreprocessingResult
from src.region_detector import DetectedRegion


@dataclass
class ClassifiedRegion:
    """Represents a classified manuscript layout region ready for output serialization."""
    bbox: Tuple[int, int, int, int]  # [x_min, y_min, x_max, y_max] in ORIGINAL unrotated coordinates
    label: str                       # One of: header, footer, main_text, side_text, filler
    confidence: float                # Calibrated confidence score in [0.0, 1.0]


class RegionClassifier:
    """
    Classifies layout regions using relative spatial anchoring,
    manuscript domain heuristics, and calibrated confidence scoring.
    """

    def __init__(self, config: Optional[RegionClassifierConfig] = None) -> None:
        self.config = config or RegionClassifierConfig()

    def _compute_confidence(
        self,
        label: str,
        geom_alignment: float,
        stroke_density: float,
        area_ratio: float,
    ) -> float:
        """
        Compute a calibrated confidence score between 0.75 and 0.98 based on
        geometric fit, stroke density plausibility, and relative area.

        Args:
            label: Assigned target class.
            geom_alignment: Geometric canonicality score [0.0, 1.0].
            stroke_density: Ratio of ink pixels to region area.
            area_ratio: Ratio of region area to total page area.

        Returns:
            float: Calibrated confidence score rounded to 2 decimal places.
        """
        base = self.config.conf_base

        if label == CLASS_FILLER:
            dens_score = 0.90
        else:
            dens_score = float(1.0 - min(1.0, abs(stroke_density - 0.20) / 0.20))

        score = (
            base
            + self.config.conf_geometric_weight * geom_alignment
            + self.config.conf_density_weight * max(0.0, dens_score)
            + self.config.conf_area_weight * min(1.0, area_ratio * 4.0)
        )

        return round(float(np.clip(score, 0.75, 0.98)), 2)

    def classify(
        self,
        regions: List[DetectedRegion],
        prep_result: PreprocessingResult,
    ) -> List[ClassifiedRegion]:
        """
        Classify all detected candidate regions relative to the manuscript page substrate
        and primary main text body.

        Args:
            regions: List of DetectedRegion candidates.
            prep_result: PreprocessingResult containing deskewed masks and substrate bounds.

        Returns:
            List[ClassifiedRegion]: Final classified layout regions with calibrated confidence.
        """
        if not regions:
            return []

        h = prep_result.image_height
        w = prep_result.image_width
        total_page_area = float(w * h)

        sub_x, sub_y, sub_w, sub_h = prep_result.substrate_bbox
        sub_top = sub_y
        sub_bot = sub_y + sub_h
        sub_left = sub_x
        sub_right = sub_x + sub_w

        classified: List[ClassifiedRegion] = []

        for r in regions:
            x1, y1, x2, y2 = r.bbox
            dx1, dy1, dx2, dy2 = r.deskewed_bbox
            bw = dx2 - dx1
            bh = dy2 - dy1
            mid_x = (dx1 + dx2) / 2.0
            mid_y = (dy1 + dy2) / 2.0
            reg_area = float((x2 - x1) * (y2 - y1))
            area_ratio = reg_area / total_page_area

            # ------------------------------------------------------------------
            # Rule 1: Modern Digitization Banners or Off-Substrate Regions -> FILLER
            # ------------------------------------------------------------------
            if r.is_banner or not r.is_on_substrate:
                conf = self._compute_confidence(
                    CLASS_FILLER,
                    geom_alignment=0.96 if r.is_banner else 0.92,
                    stroke_density=r.stroke_density,
                    area_ratio=area_ratio,
                )
                classified.append(ClassifiedRegion(bbox=r.bbox, label=CLASS_FILLER, confidence=conf))
                continue

            # ------------------------------------------------------------------
            # Rule 2: Substantial Manuscript Text Body -> MAIN_TEXT
            # ------------------------------------------------------------------
            is_substantial_body = (bw >= sub_w * 0.35 and bh >= sub_h * 0.35) or (
                r.line_count >= 3 and bw >= sub_w * 0.25
            )
            if is_substantial_body:
                conf = self._compute_confidence(
                    CLASS_MAIN_TEXT,
                    geom_alignment=0.96,
                    stroke_density=r.stroke_density,
                    area_ratio=area_ratio,
                )
                classified.append(ClassifiedRegion(bbox=r.bbox, label=CLASS_MAIN_TEXT, confidence=conf))
                continue

            # ------------------------------------------------------------------
            # Rule 3: Side Marginalia along Left/Right Margins -> SIDE_TEXT
            # ------------------------------------------------------------------
            is_in_side_margin = (mid_x <= sub_left + sub_w * 0.25) or (mid_x >= sub_right - sub_w * 0.25)
            if is_in_side_margin and bw <= sub_w * 0.40:
                conf = self._compute_confidence(
                    CLASS_SIDE_TEXT,
                    geom_alignment=0.93,
                    stroke_density=r.stroke_density,
                    area_ratio=area_ratio,
                )
                classified.append(ClassifiedRegion(bbox=r.bbox, label=CLASS_SIDE_TEXT, confidence=conf))
                continue

            # ------------------------------------------------------------------
            # Rule 4: Top Margin Running Headers / Titles / Folios -> HEADER
            # ------------------------------------------------------------------
            is_in_header_margin = (mid_y <= sub_top + sub_h * self.config.header_max_y_ratio) or (mid_y <= h * 0.25)
            if is_in_header_margin and bh <= sub_h * 0.30:
                conf = self._compute_confidence(
                    CLASS_HEADER,
                    geom_alignment=0.92,
                    stroke_density=r.stroke_density,
                    area_ratio=area_ratio,
                )
                classified.append(ClassifiedRegion(bbox=r.bbox, label=CLASS_HEADER, confidence=conf))
                continue

            # ------------------------------------------------------------------
            # Rule 5: Bottom Margin Catchwords / Numbers / Signatures -> FOOTER
            # ------------------------------------------------------------------
            is_in_footer_margin = (mid_y >= sub_top + sub_h * self.config.footer_min_y_ratio) or (mid_y >= h * 0.75)
            if is_in_footer_margin and bh <= sub_h * 0.30:
                conf = self._compute_confidence(
                    CLASS_FOOTER,
                    geom_alignment=0.92,
                    stroke_density=r.stroke_density,
                    area_ratio=area_ratio,
                )
                classified.append(ClassifiedRegion(bbox=r.bbox, label=CLASS_FOOTER, confidence=conf))
                continue

            # ------------------------------------------------------------------
            # Rule 6: Faint Marks, Specks or Small Fragments -> FILLER
            # ------------------------------------------------------------------
            if r.stroke_pixel_count < 80 or r.stroke_density < self.config.filler_min_faintness_stroke_density:
                conf = self._compute_confidence(
                    CLASS_FILLER,
                    geom_alignment=0.86,
                    stroke_density=r.stroke_density,
                    area_ratio=area_ratio,
                )
                classified.append(ClassifiedRegion(bbox=r.bbox, label=CLASS_FILLER, confidence=conf))
                continue

            # Default fallback for inner text block: MAIN_TEXT
            conf = self._compute_confidence(
                CLASS_MAIN_TEXT,
                geom_alignment=0.88,
                stroke_density=r.stroke_density,
                area_ratio=area_ratio,
            )
            classified.append(ClassifiedRegion(bbox=r.bbox, label=CLASS_MAIN_TEXT, confidence=conf))

        # Sort classified regions top-to-bottom, left-to-right
        classified = sorted(classified, key=lambda r: (r.bbox[1], r.bbox[0]))
        return classified
