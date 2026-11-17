"""
Command-Line Interface (CLI) for Manuscript Specific Layout Region Detection.
Usage:
    python inference.py --input <path_to_image_or_folder> --output <results_dir>

Generates:
    <output_dir>/annotated/<filename>_annotated.png
    <output_dir>/predictions/<filename>.json
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from config import DEFAULT_CONFIG, TARGET_CLASSES, PipelineConfig
from src.preprocessing import Preprocessor
from src.region_classifier import RegionClassifier
from src.region_detector import RegionDetector
from src.utils import (
    clip_bbox,
    find_images,
    read_image,
    save_json,
    write_image,
)
from src.visualization import Visualizer


class ManuscriptLayoutInference:
    """
    Production-grade inference pipeline runner for manuscript layout region detection.
    Encapsulates preprocessing, detection, classification, annotation, and serialization.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.preprocessor = Preprocessor(self.config.preprocessing)
        self.detector = RegionDetector(self.config.detector)
        self.classifier = RegionClassifier(self.config.classifier)
        self.visualizer = Visualizer(self.config.visualization)

    def process_single_image(
        self,
        image_path: str,
        output_dir: str,
    ) -> Dict[str, any]:
        """
        Process a single manuscript image, generating prediction JSON and annotated image.

        Args:
            image_path: Absolute or relative path to manuscript image.
            output_dir: Destination results directory.

        Returns:
            Dict containing processing status, predictions, and runtime statistics.
        """
        filename = os.path.basename(image_path)
        stem = os.path.splitext(filename)[0]

        # Read image safely with Unicode / Windows path support
        img = read_image(image_path)
        h, w = img.shape[:2]

        # Step 1: Preprocessing & Deskewing
        prep_result = self.preprocessor.process(img)

        # Step 2: Region Detection
        detected_regions = self.detector.detect(prep_result)

        # Step 3: Semantic Region Classification
        classified_regions = self.classifier.classify(detected_regions, prep_result)

        # Step 4: Strict Bounding Box Validation and JSON Payload Construction
        regions_payload: List[Dict[str, any]] = []
        for reg in classified_regions:
            x1, y1, x2, y2 = clip_bbox(reg.bbox, w, h)
            regions_payload.append({
                "bbox": [x1, y1, x2, y2],
                "label": reg.label,
                "confidence": reg.confidence,
            })

        prediction_data = {
            "filename": filename,
            "image_width": w,
            "image_height": h,
            "skew_angle_deg": prep_result.skew_angle_deg,
            "num_regions": len(regions_payload),
            "regions": regions_payload,
        }

        # Step 5: Save JSON predictions
        predictions_dir = os.path.join(output_dir, "predictions")
        json_path = os.path.join(predictions_dir, f"{stem}.json")
        save_json(prediction_data, json_path)

        # Step 6: Generate and Save Visualized Annotations (Non-destructive)
        annotated_image = self.visualizer.annotate(img, classified_regions)
        annotated_dir = os.path.join(output_dir, "annotated")
        annotated_path = os.path.join(annotated_dir, f"{stem}_annotated.png")
        write_image(annotated_path, annotated_image)

        return {
            "filename": filename,
            "image_width": w,
            "image_height": h,
            "skew_angle_deg": prep_result.skew_angle_deg,
            "num_regions": len(regions_payload),
            "json_path": json_path,
            "annotated_path": annotated_path,
            "classes": [r["label"] for r in regions_payload],
        }

    def run_batch(
        self,
        input_path: str,
        output_dir: str,
    ) -> int:
        """
        Execute layout detection over a single image or an entire folder of images.

        Args:
            input_path: Path to image file or directory.
            output_dir: Target output root directory.

        Returns:
            int: Exit status code (0 for success).
        """
        resolved_input = os.path.abspath(input_path)
        resolved_output = os.path.abspath(output_dir)

        print("=" * 76)
        print("  Manuscript Specific Layout Region Detection Pipeline")
        print("=" * 76)
        print(f"  Input  : {resolved_input}")
        print(f"  Output : {resolved_output}")
        print("-" * 76)

        try:
            image_paths = find_images(resolved_input, self.config.supported_extensions)
        except Exception as e:
            print(f"[-] Error finding images: {e}", file=sys.stderr)
            return 1

        total_images = len(image_paths)
        print(f"[*] Found {total_images} image(s) to process.")
        print("-" * 76)

        os.makedirs(os.path.join(resolved_output, "annotated"), exist_ok=True)
        os.makedirs(os.path.join(resolved_output, "predictions"), exist_ok=True)

        class_counts = {cls_name: 0 for cls_name in TARGET_CLASSES}
        success_count = 0
        failure_count = 0
        start_time = time.time()

        for idx, img_path in enumerate(image_paths, 1):
            base_name = os.path.basename(img_path)
            prefix = f"[{idx}/{total_images}]"
            try:
                t0 = time.time()
                result = self.process_single_image(img_path, resolved_output)
                elapsed = time.time() - t0

                success_count += 1
                for c in result["classes"]:
                    class_counts[c] = class_counts.get(c, 0) + 1

                print(
                    f"{prefix} SUCCESS: {base_name} "
                    f"({result['image_width']}x{result['image_height']}, "
                    f"skew: {result['skew_angle_deg']:+.2f} deg, "
                    f"{result['num_regions']} regions, {elapsed:.2f}s)"
                )
            except Exception as e:
                failure_count += 1
                print(
                    f"{prefix} WARNING: Failed to process '{base_name}': {e}",
                    file=sys.stderr,
                )

        total_elapsed = time.time() - start_time
        print("=" * 76)
        print("  Execution Summary")
        print("=" * 76)
        print(f"  Total Images Processed : {total_images}")
        print(f"  Successfully Finished  : {success_count}")
        print(f"  Failed / Corrupted     : {failure_count}")
        print(f"  Total Elapsed Time     : {total_elapsed:.2f}s (Avg: {total_elapsed/max(1, total_images):.2f}s/img)")
        print("-" * 76)
        print("  Detected Regions Breakdown by Class:")
        for cls_name in TARGET_CLASSES:
            count = class_counts.get(cls_name, 0)
            print(f"    - {cls_name:<12}: {count:>4} region(s)")
        print("=" * 76)
        print(f"  Outputs saved to:\n    - Predictions : {os.path.join(resolved_output, 'predictions')}\n    - Annotated   : {os.path.join(resolved_output, 'annotated')}")
        print("=" * 76)

        return 0 if failure_count == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Manuscript Specific Layout Region Detection CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to an input image file or a directory containing manuscript images.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=str,
        help="Path to the destination results directory.",
    )
    return parser


def main() -> None:
    """Entry point for the CLI script."""
    parser = build_parser()
    args = parser.parse_args()

    pipeline = ManuscriptLayoutInference()
    exit_code = pipeline.run_batch(args.input, args.output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

