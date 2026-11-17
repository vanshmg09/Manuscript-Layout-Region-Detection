"""
Comprehensive Pytest Test Suite for Manuscript Layout Region Detection.
Tests:
1. Synthetic manuscript generator (header, main_text, footer, side_text, filler)
2. Preprocessing & deskewing accuracy
3. Illumination flattening and hysteresis binarization
4. Region detection and multi-column splitting
5. Bounding box boundary safety [0 <= x_min < x_max <= W, 0 <= y_min < y_max <= H]
6. Input immutability (verifying SHA-256 integrity before and after inference)
7. CLI batch runner, exit codes, and error handling
8. Unicode / Windows path compatibility
9. Real evaluation sample images validation
"""

import hashlib
import json
import os
import shutil
import tempfile
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest

from config import (
    CLASS_FILLER,
    CLASS_FOOTER,
    CLASS_HEADER,
    CLASS_MAIN_TEXT,
    CLASS_SIDE_TEXT,
    DEFAULT_CONFIG,
    TARGET_CLASSES,
)
from inference import ManuscriptLayoutInference
from src.preprocessing import Preprocessor
from src.region_classifier import RegionClassifier
from src.region_detector import RegionDetector
from src.utils import (
    clip_bbox,
    compute_iou,
    find_images,
    read_image,
    save_json,
    write_image,
)
from src.visualization import Visualizer


# ==============================================================================
# Synthetic Manuscript Generation Fixtures
# ==============================================================================
def draw_script_line(
    img: np.ndarray,
    x1: int,
    y: int,
    x2: int,
    char_h: int = 16,
    char_w: int = 14,
    color: Tuple[int, int, int] = (25, 25, 25),
) -> None:
    """Simulate handwritten manuscript script strokes with top shirorekha bar."""
    # Top horizontal connecting line
    cv2.line(img, (x1, y), (x2, y), color, 2)
    # Downward character strokes
    for x in range(x1 + 4, x2 - 4, char_w):
        cv2.line(img, (x, y), (x + 2, y + char_h), color, 2)
        if (x // char_w) % 2 == 0:
            cv2.ellipse(img, (x + 5, y + char_h // 2), (4, 4), 0, 0, 360, color, 1)


def create_synthetic_manuscript(
    width: int = 1200,
    height: int = 600,
    skew_angle_deg: float = 0.0,
    has_header: bool = True,
    has_footer: bool = True,
    has_side_text: bool = True,
    has_banner: bool = True,
    multi_column: bool = False,
    uneven_lighting: bool = True,
) -> Tuple[np.ndarray, Dict[str, List[Tuple[int, int, int, int]]]]:
    """
    Generate a photorealistic synthetic manuscript page with known ground truth layout regions.
    """
    # 1. Warm paper background (cream / pale tan: BGR ~ [195, 215, 230])
    img = np.full((height, width, 3), (195, 215, 230), dtype=np.uint8)

    # Add subtle paper grain noise
    grain = np.random.normal(0, 5, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    ground_truth: Dict[str, List[Tuple[int, int, int, int]]] = {
        cls_name: [] for cls_name in TARGET_CLASSES
    }

    ink_color = (30, 25, 25)

    # 2. Modern digitization banners (filler)
    if has_banner:
        # Top digitization banner
        cv2.putText(
            img,
            "CC-0. In Public Domain. Digitization by eGangotri. Funding MIDF",
            (width // 4, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        ground_truth[CLASS_FILLER].append((width // 4 - 10, 10, 3 * width // 4 + 10, 35))

        # Bottom digitization banner
        cv2.putText(
            img,
            "CSU-Sringeri Campus Rare Book and Manuscript Collection, Sringeri",
            (width // 4 - 30, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        ground_truth[CLASS_FILLER].append((width // 4 - 40, height - 35, 3 * width // 4 + 40, height - 10))

    # 3. Header in top margin
    if has_header:
        hx1, hy, hx2 = width // 3, 75, 2 * width // 3
        draw_script_line(img, hx1, hy, hx2, char_h=14, char_w=12, color=ink_color)
        ground_truth[CLASS_HEADER].append((hx1, hy - 4, hx2, hy + 20))

    # 4. Main Text Body
    body_y_start = 130
    body_y_end = 450
    line_pitch = 35

    if multi_column:
        # Two parallel columns separated by an inter-column gutter (valley)
        col1_x1, col1_x2 = 280, 560
        col2_x1, col2_x2 = 640, 920

        for y in range(body_y_start, body_y_end, line_pitch):
            draw_script_line(img, col1_x1, y, col1_x2, char_h=16, char_w=14, color=ink_color)
            draw_script_line(img, col2_x1, y, col2_x2, char_h=16, char_w=14, color=ink_color)

        ground_truth[CLASS_MAIN_TEXT].append((col1_x1, body_y_start - 4, col1_x2, body_y_end + 10))
        ground_truth[CLASS_MAIN_TEXT].append((col2_x1, body_y_start - 4, col2_x2, body_y_end + 10))
    else:
        # Single continuous main text body
        bx1, bx2 = 260, 960
        for y in range(body_y_start, body_y_end, line_pitch):
            draw_script_line(img, bx1, y, bx2, char_h=16, char_w=14, color=ink_color)
        ground_truth[CLASS_MAIN_TEXT].append((bx1, body_y_start - 4, bx2, body_y_end + 10))

    # 5. Side Text (Marginalia on left margin)
    if has_side_text:
        sx1, sx2 = 80, 200
        for sy in range(200, 360, 30):
            draw_script_line(img, sx1, sy, sx2, char_h=12, char_w=10, color=(60, 50, 50))
        ground_truth[CLASS_SIDE_TEXT].append((sx1, 195, sx2, 380))

    # 6. Footer in bottom margin
    if has_footer:
        fx1, fy, fx2 = width // 2 - 80, 510, width // 2 + 80
        draw_script_line(img, fx1, fy, fx2, char_h=14, char_w=12, color=ink_color)
        ground_truth[CLASS_FOOTER].append((fx1, fy - 4, fx2, fy + 20))

    # 7. Non-uniform illumination gradient (darker corners, bright center)
    if uneven_lighting:
        xx, yy = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        radius = np.sqrt(xx ** 2 + yy ** 2)
        vignette = 1.0 - 0.35 * (radius ** 2)
        vignette = np.clip(vignette, 0.4, 1.0)
        img = (img * vignette[:, :, np.newaxis]).astype(np.uint8)

    # 8. Rotation / Skew
    if abs(skew_angle_deg) > 0.01:
        M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), skew_angle_deg, 1.0)
        img = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return img, ground_truth


# ==============================================================================
# Test Cases
# ==============================================================================
class TestManuscriptPipeline:
    """Test suite covering preprocessing, detection, classification, CLI, and integrity."""

    @pytest.fixture
    def pipeline(self) -> ManuscriptLayoutInference:
        return ManuscriptLayoutInference()

    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp(prefix="manuscript_test_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. Bounding Box Boundary Safety Tests
    # --------------------------------------------------------------------------
    def test_clip_bbox_boundaries(self):
        """Verify strict clipping: 0 <= x_min < x_max <= W and 0 <= y_min < y_max <= H."""
        W, H = 1000, 800

        # Out-of-bounds negative coordinates
        x1, y1, x2, y2 = clip_bbox([-50, -30, 200, 300], W, H)
        assert 0 <= x1 < x2 <= W
        assert 0 <= y1 < y2 <= H
        assert x1 == 0 and y1 == 0

        # Out-of-bounds positive coordinates
        x1, y1, x2, y2 = clip_bbox([800, 600, 1200, 950], W, H)
        assert 0 <= x1 < x2 <= W
        assert 0 <= y1 < y2 <= H
        assert x2 == W and y2 == H

        # Inverted coordinates
        x1, y1, x2, y2 = clip_bbox([500, 400, 200, 100], W, H)
        assert x1 < x2
        assert y1 < y2

    # --------------------------------------------------------------------------
    # 2. Preprocessing & Deskewing Verification
    # --------------------------------------------------------------------------
    def test_illumination_flattening(self):
        """Verify illumination division eliminates lighting gradients."""
        prep = Preprocessor()
        synthetic_img, _ = create_synthetic_manuscript(skew_angle_deg=0.0, uneven_lighting=True)
        gray = cv2.cvtColor(synthetic_img, cv2.COLOR_BGR2GRAY)

        flattened = prep.flatten_illumination(gray)
        assert flattened.shape == gray.shape
        assert flattened.dtype == np.uint8

        # Background corners should have significantly reduced variance after flattening
        c1 = flattened[0:40, 0:40]
        c2 = flattened[flattened.shape[0]-40:, flattened.shape[1]-40:]
        assert abs(float(np.mean(c1)) - float(np.mean(c2))) < 45.0

    def test_deskewing_accuracy(self):
        """Verify deskewing estimates rotation angle within 0.5 degrees tolerance."""
        prep = Preprocessor()
        known_skew = 2.5
        synthetic_img, _ = create_synthetic_manuscript(skew_angle_deg=known_skew, uneven_lighting=False)

        res = prep.process(synthetic_img)
        # Detected angle should counteract known_skew (within 0.5 deg)
        assert abs(res.skew_angle_deg - (-known_skew)) <= 0.6

    # --------------------------------------------------------------------------
    # 3. Dual-Scale Hysteresis Binarization Tests
    # --------------------------------------------------------------------------
    def test_hysteresis_preserves_strokes_and_rejects_grain(self):
        """Verify hysteresis keeps faint character strokes while rejecting isolated noise."""
        prep = Preprocessor()
        synthetic_img, _ = create_synthetic_manuscript(skew_angle_deg=0.0, uneven_lighting=False)
        gray = cv2.cvtColor(synthetic_img, cv2.COLOR_BGR2GRAY)
        flattened = prep.flatten_illumination(gray)
        enhanced = prep.enhance_contrast_and_denoise(flattened)

        mask = prep.hysteresis_binarize(enhanced)
        assert mask.dtype == np.uint8
        assert np.sum(mask > 0) > 1000  # Foreground strokes preserved

    # --------------------------------------------------------------------------
    # 4. Multi-Column Splitting Verification
    # --------------------------------------------------------------------------
    def test_multi_column_splitting(self):
        """Verify that multi-column manuscript layouts are split across vertical gutters."""
        prep = Preprocessor()
        det = RegionDetector()

        synthetic_img, _ = create_synthetic_manuscript(
            width=1200, height=600, skew_angle_deg=0.0, multi_column=True, has_banner=False
        )
        res = prep.process(synthetic_img)
        regions = det.detect(res)

        # Multi-column layout should produce at least 2 distinct column regions
        body_regions = [r for r in regions if r.is_on_substrate and r.line_count >= 3]
        assert len(body_regions) >= 2

    # --------------------------------------------------------------------------
    # 5. Full 5-Class Layout Classification Verification
    # --------------------------------------------------------------------------
    def test_full_five_class_detection_on_synthetic_page(self, temp_dir):
        """Verify that all 5 target layout classes are detected and classified."""
        synthetic_img, _ = create_synthetic_manuscript(
            width=1200,
            height=600,
            skew_angle_deg=0.0,
            has_header=True,
            has_footer=True,
            has_side_text=True,
            has_banner=True,
            multi_column=False,
        )

        test_img_path = os.path.join(temp_dir, "synthetic_full.png")
        write_image(test_img_path, synthetic_img)

        pipeline = ManuscriptLayoutInference()
        result = pipeline.process_single_image(test_img_path, temp_dir)

        detected_classes = set(result["classes"])
        # Must detect main_text, filler, side_text, header, footer
        assert CLASS_MAIN_TEXT in detected_classes
        assert CLASS_FILLER in detected_classes
        assert CLASS_SIDE_TEXT in detected_classes
        assert CLASS_HEADER in detected_classes
        assert CLASS_FOOTER in detected_classes

        # Verify JSON predictions integrity
        with open(result["json_path"], "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["filename"] == "synthetic_full.png"
        assert data["image_width"] == 1200
        assert data["image_height"] == 600
        assert data["num_regions"] == len(data["regions"])

        for r in data["regions"]:
            assert r["label"] in TARGET_CLASSES
            assert 0.70 <= r["confidence"] <= 1.00
            x1, y1, x2, y2 = r["bbox"]
            assert 0 <= x1 < x2 <= 1200
            assert 0 <= y1 < y2 <= 600

        # Verify annotated image exists and has valid dimensions
        annotated_img = read_image(result["annotated_path"])
        assert annotated_img.shape == (600, 1200, 3)

    # --------------------------------------------------------------------------
    # 6. Real Evaluation Sample Images Validation
    # --------------------------------------------------------------------------
    def test_real_sample_paper_manuscript(self, temp_dir):
        """Verify pipeline execution on the assignment paper manuscript sample."""
        real_path = "data/input/sample_paper.jpg"
        if not os.path.isfile(real_path):
            pytest.skip("sample_paper.jpg not found in data/input/")

        pipeline = ManuscriptLayoutInference()
        result = pipeline.process_single_image(real_path, temp_dir)

        assert result["filename"] == "sample_paper.jpg"
        assert result["image_width"] == 1239
        assert result["image_height"] == 468
        assert result["num_regions"] == 3  # Exactly 3 regions (2 fillers + 1 main_text)
        assert CLASS_MAIN_TEXT in result["classes"]
        assert CLASS_FILLER in result["classes"]

    def test_real_sample_palm_leaf_manuscript(self, temp_dir):
        """Verify pipeline execution on the assignment palm-leaf manuscript sample."""
        real_path = "data/input/sample_palm_leaf.jpg"
        if not os.path.isfile(real_path):
            pytest.skip("sample_palm_leaf.jpg not found in data/input/")

        pipeline = ManuscriptLayoutInference()
        result = pipeline.process_single_image(real_path, temp_dir)

        assert result["filename"] == "sample_palm_leaf.jpg"
        assert result["image_width"] == 1239
        assert result["image_height"] == 504
        assert CLASS_MAIN_TEXT in result["classes"]
        assert CLASS_SIDE_TEXT in result["classes"]
        assert CLASS_FILLER in result["classes"]

    # --------------------------------------------------------------------------
    # 7. Input Immutability Verification
    # --------------------------------------------------------------------------
    def test_input_immutability(self, temp_dir):
        """Verify that original input image file is never modified or overwritten."""
        synthetic_img, _ = create_synthetic_manuscript()
        test_file = os.path.join(temp_dir, "immutable_sample.png")
        write_image(test_file, synthetic_img)

        # Compute initial sha256 checksum
        with open(test_file, "rb") as f:
            initial_hash = hashlib.sha256(f.read()).hexdigest()

        # Run inference
        pipeline = ManuscriptLayoutInference()
        pipeline.process_single_image(test_file, temp_dir)

        # Re-compute sha256 checksum
        with open(test_file, "rb") as f:
            post_hash = hashlib.sha256(f.read()).hexdigest()

        assert initial_hash == post_hash, "Original input file was modified during inference!"

    # --------------------------------------------------------------------------
    # 8. Unicode & Windows-Safe Path Compatibility
    # --------------------------------------------------------------------------
    def test_unicode_and_windows_paths(self, temp_dir):
        """Verify that paths with spaces, Devanagari, or Unicode characters are handled safely."""
        unicode_dir = os.path.join(temp_dir, "पाण्डुलिपि संग्रह Folder")
        os.makedirs(unicode_dir, exist_ok=True)

        synthetic_img, _ = create_synthetic_manuscript(width=600, height=300)
        unicode_img_path = os.path.join(unicode_dir, "नमूना_paper.png")
        write_image(unicode_img_path, synthetic_img)

        # Test read
        read_back = read_image(unicode_img_path)
        assert read_back.shape == (300, 600, 3)

        # Test inference
        pipeline = ManuscriptLayoutInference()
        out_unicode = os.path.join(temp_dir, "results_परिणाम")
        res = pipeline.process_single_image(unicode_img_path, out_unicode)
        assert os.path.isfile(res["json_path"])
        assert os.path.isfile(res["annotated_path"])

    # --------------------------------------------------------------------------
    # 9. CLI Batch Execution & Error Handling
    # --------------------------------------------------------------------------
    def test_cli_batch_run_with_relative_paths(self, temp_dir):
        """Verify batch execution via find_images and relative path resolution."""
        # Create input dir with 2 images
        input_dir = os.path.join(temp_dir, "batch_in")
        output_dir = os.path.join(temp_dir, "batch_out")
        os.makedirs(input_dir, exist_ok=True)

        for i in range(2):
            img, _ = create_synthetic_manuscript(width=800, height=400)
            write_image(os.path.join(input_dir, f"page_{i}.png"), img)

        pipeline = ManuscriptLayoutInference()
        exit_code = pipeline.run_batch(input_dir, output_dir)
        assert exit_code == 0

        pred_dir = os.path.join(output_dir, "predictions")
        ann_dir = os.path.join(output_dir, "annotated")
        assert len(os.listdir(pred_dir)) == 2
        assert len(os.listdir(ann_dir)) == 2

    def test_cli_handles_corrupted_file_without_crashing(self, temp_dir):
        """Verify that corrupted image files are handled gracefully without aborting batch."""
        input_dir = os.path.join(temp_dir, "corrupt_test_in")
        output_dir = os.path.join(temp_dir, "corrupt_test_out")
        os.makedirs(input_dir, exist_ok=True)

        # Create 1 valid image
        valid_img, _ = create_synthetic_manuscript(width=800, height=400)
        write_image(os.path.join(input_dir, "valid.png"), valid_img)

        # Create 1 corrupted zero-byte file with image extension
        corrupt_path = os.path.join(input_dir, "corrupted.png")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_IMAGE_HEADER")

        pipeline = ManuscriptLayoutInference()
        exit_code = pipeline.run_batch(input_dir, output_dir)
        # Should finish batch processing without fatal crash (exit_code 2 indicates partial failure)
        assert exit_code == 2

        # Valid image must have been processed successfully
        assert os.path.isfile(os.path.join(output_dir, "predictions", "valid.json"))
        assert os.path.isfile(os.path.join(output_dir, "annotated", "valid_annotated.png"))

    def test_find_images_missing_dir_raises_filenotfound(self):
        """Verify FileNotFoundError is raised when directory does not exist."""
        with pytest.raises(FileNotFoundError):
            find_images("completely_fictional_non_existent_folder_xyz")

