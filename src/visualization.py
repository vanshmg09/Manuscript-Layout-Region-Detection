"""
Visualization module for Manuscript Layout Region Detection.
Draws color-coded bounding boxes, semi-transparent region overlays,
and high-contrast text badges on a non-destructive copy of the input image.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import CLASS_COLORS_BGR, VisualizationConfig
from src.region_classifier import ClassifiedRegion
from src.utils import clip_bbox


class Visualizer:
    """
    Renders high-visibility, publication-grade annotations for detected layout regions.
    Guarantees non-destructive processing by operating on an isolated copy of the image.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None) -> None:
        self.config = config or VisualizationConfig()

    def annotate(
        self,
        image: np.ndarray,
        regions: List[ClassifiedRegion],
    ) -> np.ndarray:
        """
        Draw color-coded bounding boxes, translucent fills, and label badges
        on an image copy.

        Args:
            image: Original BGR image numpy array.
            regions: List of ClassifiedRegion objects.

        Returns:
            np.ndarray: Annotated BGR image.
        """
        # Non-destructive copy
        annotated = image.copy()
        overlay = annotated.copy()
        h, w = annotated.shape[:2]

        # First pass: semi-transparent tinted fill overlay
        alpha = self.config.alpha_overlay
        if alpha > 0.0:
            for reg in regions:
                x1, y1, x2, y2 = clip_bbox(reg.bbox, w, h)
                color = CLASS_COLORS_BGR.get(reg.label, (180, 180, 180))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

        # Second pass: crisp borders and text badges
        for reg in regions:
            x1, y1, x2, y2 = clip_bbox(reg.bbox, w, h)
            color = CLASS_COLORS_BGR.get(reg.label, (180, 180, 180))

            # Draw outer rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.config.bbox_thickness)

            # Badge text formatting
            badge_text = f"{reg.label} {reg.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = self.config.font_scale
            font_thickness = self.config.font_thickness

            (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thickness)
            pad_x = self.config.badge_padding_x
            pad_y = self.config.badge_padding_y

            # Position badge above bbox if room permits, otherwise inside
            badge_h = text_h + 2 * pad_y
            badge_w = text_w + 2 * pad_x

            if y1 - badge_h >= 0:
                badge_y1 = y1 - badge_h
                badge_y2 = y1
            else:
                badge_y1 = y1
                badge_y2 = min(h, y1 + badge_h)

            badge_x1 = max(0, x1)
            badge_x2 = min(w, badge_x1 + badge_w)

            # Draw filled badge background
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)

            # Text color (white)
            text_x = badge_x1 + pad_x
            text_y = badge_y2 - pad_y
            cv2.putText(
                annotated,
                badge_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        return annotated

