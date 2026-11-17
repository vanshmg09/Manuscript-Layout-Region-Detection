"""
Preprocessing Pipeline for Historical Manuscript Layout Analysis.
Implements:
1. Illumination flattening (background division with heavy Gaussian blur)
2. Contrast enhancement (CLAHE + edge-preserving bilateral denoising)
3. Dual-scale hysteresis binarization (preserving faint strokes, suppressing grain)
4. Page substrate mask (isolating palm-leaf / paper from background cloth/table)
5. Ruling / border line suppression (morphological elimination of scribe borders)
6. Deskewing (projection profile variance maximization)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from config import PreprocessingConfig


@dataclass
class PreprocessingResult:
    """Holds the outputs of the preprocessing stage."""
    original_image: np.ndarray          # Original unrotated BGR image
    deskewed_image: np.ndarray          # Deskewed BGR image
    binary_ink_mask: np.ndarray         # Cleaned binary ink mask (255=ink, 0=bg)
    substrate_mask: np.ndarray          # Binary mask of the manuscript page (255=page)
    substrate_bbox: Tuple[int, int, int, int] # (x, y, w, h) of page substrate
    skew_angle_deg: float               # Detected rotation angle in degrees
    rotation_matrix: np.ndarray         # 2x3 affine forward rotation matrix
    inv_rotation_matrix: np.ndarray     # 2x3 affine inverse rotation matrix
    image_width: int                    # Original width
    image_height: int                   # Original height


class Preprocessor:
    """
    Robust Preprocessor tailored for historical manuscripts (palm-leaf and paper).
    Handles uneven lighting, faded ink, grain, skew, and scribe ruling lines.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()

    def flatten_illumination(self, gray: np.ndarray) -> np.ndarray:
        """
        Normalize non-uniform background lighting and vignetting by dividing
        the grayscale image by a heavy Gaussian-blurred background estimation.

        Args:
            gray: Single-channel uint8 grayscale image.

        Returns:
            np.ndarray: Illumination-flattened uint8 image.
        """
        h, w = gray.shape
        # Dynamic kernel size proportional to image height, guaranteed odd
        ksize = int(h * self.config.bg_blur_ratio) | 1
        ksize = max(self.config.min_bg_blur_ksize, min(self.config.max_bg_blur_ksize, ksize))

        # Heavy Gaussian blur captures low-frequency background luminance
        bg = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Division normalization: flattened = (gray / bg) * 255
        flattened = cv2.divide(gray, bg, scale=255)
        return flattened

    def enhance_contrast_and_denoise(self, gray: np.ndarray) -> np.ndarray:
        """
        Enhance faint ink strokes using CLAHE while smoothing grain via bilateral filter.

        Args:
            gray: Illumination-flattened grayscale image.

        Returns:
            np.ndarray: Enhanced and denoised grayscale image.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )
        enhanced = clahe.apply(gray)

        # Bilateral filter eliminates paper/leaf grain while keeping crisp edges
        denoised = cv2.bilateralFilter(
            enhanced,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space,
        )
        return denoised

    def hysteresis_binarize(self, enhanced_gray: np.ndarray) -> np.ndarray:
        """
        Dual-scale hysteresis binarization.
        Inverts image so dark ink is bright foreground.
        High threshold identifies confident ink stroke seeds.
        Low threshold identifies faint stroke continuations.
        Reconstruction retains only low-threshold components connected to high-threshold seeds.

        Args:
            enhanced_gray: Denoised grayscale image.

        Returns:
            np.ndarray: Binary mask where 255 represents ink, 0 represents background.
        """
        # Invert so ink strokes are bright foreground
        inv = cv2.bitwise_not(enhanced_gray)

        # Base Otsu threshold calculation
        otsu_val, _ = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_thresh = otsu_val * self.config.hysteresis_high_ratio
        low_thresh = otsu_val * self.config.hysteresis_low_ratio

        seeds = (inv >= high_thresh).astype(np.uint8) * 255
        potential = (inv >= low_thresh).astype(np.uint8) * 255

        # Connected component reconstruction
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(potential, connectivity=8)
        if num_labels <= 1:
            return seeds

        # Find labels touched by seed pixels
        seed_labels = set(np.unique(labels[seeds > 0]))
        seed_labels.discard(0)  # Exclude background

        # Filter out isolated tiny grain specks even if seeded (min 4 pixels)
        hysteresis_mask = np.zeros_like(potential)
        for lbl in seed_labels:
            if stats[lbl, cv2.CC_STAT_AREA] >= 4:
                hysteresis_mask[labels == lbl] = 255

        return hysteresis_mask

    def detect_page_substrate(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Automatically segment the manuscript page substrate (palm-leaf or paper)
        from dark background cloth, scanner table, or surrounding imaging borders.

        Args:
            bgr_image: Original BGR image.

        Returns:
            Tuple[np.ndarray, Tuple[int, int, int, int]]:
                - substrate_mask: Binary mask (255 on substrate, 0 off)
                - substrate_bbox: (x, y, w, h) bounding box of substrate
        """
        h, w, _ = bgr_image.shape
        total_area = float(w * h)

        # Convert to CIELAB: L-channel cleanly distinguishes bright substrate from dark cloth
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]

        # Otsu threshold on Lightness
        otsu_val, _ = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_bright = (L > otsu_val).astype(np.uint8) * 255

        # Morphological closing to seal natural holes, leaf cracks, and inner text
        k_size = self.config.substrate_morph_ksize
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        closed = cv2.morphologyEx(binary_bright, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area >= self.config.substrate_min_area_ratio * total_area:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [largest], -1, 255, -1)
                bx, by, bw, bh = cv2.boundingRect(largest)
                return mask, (int(bx), int(by), int(bw), int(bh))

        # Default fallback: full image frame is substrate (e.g. cropped paper manuscripts)
        full_mask = np.full((h, w), 255, dtype=np.uint8)
        return full_mask, (0, 0, int(w), int(h))

    def suppress_ruling_borders(
        self,
        binary_ink: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Eliminate scribe borders and ruling lines using directional morphological filtering.
        Prevents vertical/horizontal borders from falsely bridging text lines into giant blocks.

        Args:
            binary_ink: Binarized ink mask (255=ink).
            image_shape: (height, width).

        Returns:
            np.ndarray: Cleaned binary ink mask with borders suppressed.
        """
        h, w = image_shape
        min_v_len = max(20, int(h * self.config.ruling_min_v_ratio))
        min_h_len = max(20, int(w * self.config.ruling_min_h_ratio))

        # Horizontal ruling line detector (long 1D kernel)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h_len, 1))
        h_lines = cv2.morphologyEx(binary_ink, cv2.MORPH_OPEN, h_kernel)

        # Vertical ruling line detector (tall 1D kernel)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_len))
        v_lines = cv2.morphologyEx(binary_ink, cv2.MORPH_OPEN, v_kernel)

        border_lines = cv2.bitwise_or(h_lines, v_lines)

        # Dilate slightly to ensure full border removal including anti-aliased edges
        d_ksize = self.config.ruling_dilation_px
        border_dilated = cv2.dilate(
            border_lines,
            cv2.getStructuringElement(cv2.MORPH_RECT, (d_ksize, d_ksize)),
        )

        # Subtract border lines from ink mask
        clean_ink = cv2.bitwise_and(binary_ink, cv2.bitwise_not(border_dilated))
        return clean_ink

    def estimate_skew_angle(self, binary_ink: np.ndarray) -> float:
        """
        Estimate page skew angle using horizontal projection profile variance maximization.
        Text lines produce distinct peaks and troughs when aligned horizontally, maximizing variance.

        Args:
            binary_ink: Binarized ink mask.

        Returns:
            float: Detected skew angle in degrees (-angle_max to +angle_max).
        """
        h, w = binary_ink.shape
        center = (w / 2.0, h / 2.0)
        max_ang = self.config.deskew_max_angle_deg
        coarse_step = self.config.deskew_coarse_step_deg
        fine_step = self.config.deskew_fine_step_deg

        # Downsample for fast search if image is very large
        scale = 1.0
        if max(h, w) > 1200:
            scale = 1200.0 / max(h, w)
            small_mask = cv2.resize(binary_ink, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
            sh, sw = small_mask.shape
            scenter = (sw / 2.0, sh / 2.0)
        else:
            small_mask = binary_ink
            sh, sw = h, w
            scenter = center

        best_angle = 0.0
        best_score = -1.0

        # Phase 1: Coarse search
        coarse_angles = np.arange(-max_ang, max_ang + coarse_step, coarse_step)
        for ang in coarse_angles:
            M = cv2.getRotationMatrix2D(scenter, ang, 1.0)
            rotated = cv2.warpAffine(small_mask, M, (sw, sh), flags=cv2.INTER_NEAREST)
            proj = np.sum(rotated, axis=1, dtype=np.float64)
            # Second order difference variance measures sharpness of line transitions
            score = float(np.var(proj))
            if score > best_score:
                best_score = score
                best_angle = float(ang)

        # Phase 2: Fine search around coarse peak
        fine_angles = np.arange(best_angle - coarse_step, best_angle + coarse_step + fine_step, fine_step)
        for ang in fine_angles:
            M = cv2.getRotationMatrix2D(scenter, ang, 1.0)
            rotated = cv2.warpAffine(small_mask, M, (sw, sh), flags=cv2.INTER_NEAREST)
            proj = np.sum(rotated, axis=1, dtype=np.float64)
            score = float(np.var(proj))
            if score > best_score:
                best_score = score
                best_angle = float(ang)

        return round(best_angle, 2)

    def process(self, image: np.ndarray) -> PreprocessingResult:
        """
        Execute the full preprocessing pipeline on an input BGR image.

        Args:
            image: Original BGR image numpy array.

        Returns:
            PreprocessingResult: Preprocessed and deskewed data package.
        """
        h, w = image.shape[:2]

        # 1. Page substrate detection (identifies palm leaf vs cloth / paper boundary)
        substrate_mask, substrate_bbox = self.detect_page_substrate(image)

        # 2. Illumination flattening
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flattened = self.flatten_illumination(gray)

        # 3. Contrast enhancement & Denoising
        enhanced = self.enhance_contrast_and_denoise(flattened)

        # 4. Dual-scale hysteresis binarization
        binary_ink = self.hysteresis_binarize(enhanced)

        # 5. Ruling / border line suppression
        clean_ink = self.suppress_ruling_borders(binary_ink, (h, w))

        # 6. Deskewing
        skew_angle = self.estimate_skew_angle(clean_ink)
        center = (w / 2.0, h / 2.0)
        rot_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        inv_rot_matrix = cv2.invertAffineTransform(rot_matrix)

        if abs(skew_angle) > 0.05:
            deskewed_image = cv2.warpAffine(
                image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            deskewed_ink = cv2.warpAffine(
                clean_ink, rot_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            deskewed_substrate = cv2.warpAffine(
                substrate_mask, rot_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            # Recompute substrate bbox in deskewed space
            cnts, _ = cv2.findContours(deskewed_substrate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                largest = max(cnts, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(largest)
                deskewed_sub_bbox = (int(bx), int(by), int(bw), int(bh))
            else:
                deskewed_sub_bbox = substrate_bbox
        else:
            deskewed_image = image.copy()
            deskewed_ink = clean_ink.copy()
            deskewed_substrate = substrate_mask.copy()
            deskewed_sub_bbox = substrate_bbox

        return PreprocessingResult(
            original_image=image,
            deskewed_image=deskewed_image,
            binary_ink_mask=deskewed_ink,
            substrate_mask=deskewed_substrate,
            substrate_bbox=deskewed_sub_bbox,
            skew_angle_deg=skew_angle,
            rotation_matrix=rot_matrix,
            inv_rotation_matrix=inv_rot_matrix,
            image_width=w,
            image_height=h,
        )

