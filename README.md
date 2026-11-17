# Manuscript Specific Layout Region Detection

A clean, production-grade Python tool to automatically detect, segment, and classify text regions in historical manuscripts (both palm-leaf and paper formats).

---

## 1. Problem Statement

Historical manuscripts are difficult for standard software to analyze because:
- **Different Materials**: Palm leaves are long and narrow with dark background cloth and string binding holes; paper manuscripts often have double-line border frames.
- **Aging & Damage**: Inks fade over time, ink bleeds through from the backside, pages have dark water stains, uneven lighting, and tilted (skewed) scans.
- **Mixed Layouts**: Pages contain body text, margin notes (*tika*), top titles, bottom catchwords, and modern library scan stamps (e.g., eGangotri/CSU-Sringeri metadata).
- **Standard OCR Fails**: Normal OCR expects clean printed books. On ancient manuscripts, border lines fuse text together and background cloth gets mistaken for text.

---

## 2. Project Objective

The goal of this project is to build an automated, reliable Python pipeline that:
1. **Finds and labels 5 layout regions**:
   - `header`: Top titles, running headers, or folio numbers.
   - `footer`: Bottom catchwords, page numbers, or signatures.
   - `main_text`: Primary body of text (including multi-column text).
   - `side_text`: Marginal notes, commentary, or glosses along page edges.
   - `filler`: Modern English scan stamps/banners, pencil marks, and background noise.
2. **Works across both palm-leaf and paper** formats automatically without manual tuning.
3. **Cleans degradation**: Fixes uneven lighting, enhances faint ink, removes border ruling lines, and straightens tilted pages.
4. **Protects data**: Keeps original images 100% unchanged and ensures bounding boxes stay strictly inside page borders.

---

## Target Layout Classes

| Class | Box Color | Description | Example |
|---|---|---|---|
| `main_text` | **Ocean Blue** | Primary body text of the manuscript | Central paragraph lines or columns |
| `header` | **Emerald Green** | Text in the top margin | Chapter titles, top folio numbers |
| `footer` | **Royal Violet** | Text in the bottom margin | Bottom catchwords (*pratika*), page numbers |
| `side_text` | **Amber Orange** | Margin notes along left/right edges | Annotations, commentary, glosses (*tika*) |
| `filler` | **Crimson Red** | Non-manuscript or decorative elements | Modern English digitization stamps, noise |

---

## 3. Submission Deliverables

1. **Complete modular Python source code** in a private GitHub repository.
2. **CLI inference script** for batch processing.
3. **Annotated output images and JSON metadata**.
4. **Comprehensive README.md** with setup, model rationale, and execution instructions.

### CLI Example
```bash
python inference.py --input ./data/test_images --output ./results
```

---

## 4. Submission Instructions

1. Push the implementation to a private GitHub repository and grant access to the hiring team.
2. Ensure the CLI works using relative paths (e.g., `./data/test_images`, `./results`).
3. Include all dependencies in `requirements.txt` or `environment.yml`.

---

## 5. Quick Setup & How to Run

### Installation
```bash
# 1. Clone repository
git clone https://github.com/<your-username>/Manuscript-Layout-Detection.git
cd Manuscript-Layout-Detection

# 2. Install dependencies
pip install -r requirements.txt
```

### Running Inference

**Batch mode (process a whole folder):**
```bash
python inference.py --input ./data/test_images --output ./results
```

**Single image mode:**
```bash
python inference.py --input ./data/test_images/sample_paper.jpg --output ./results
```

### Running the Test Suite
```bash
python run_tests.py
```
*(Runs 13 automated tests verifying boundary safety, deskewing, and accuracy).*

---

## 6. How the Pipeline Works (Model Rationale)

The system processes each manuscript in 3 simple, automated steps:

```
[Input Image] ──> [1. Preprocessing] ──> [2. Detection] ──> [3. Classification] ──> [Results]
```

1. **Preprocessing (`src/preprocessing.py`)**:
   - **Flattens Lighting**: Divides the image by a background blur to remove uneven shadows and stains.
   - **Boosts Faint Ink**: Uses CLAHE and edge filtering to make faded strokes clear without adding grain noise.
   - **Removes Border Lines**: Detects and subtracts scribe frame lines so text lines do not merge.
   - **Isolates Page**: Separates the manuscript page from the dark scanner cloth.
   - **Straightens Skew**: Automatically detects tilt and rotates the page so text lines are horizontal.

2. **Region Detection (`src/region_detector.py`)**:
   - **Measures Character Size**: Estimates average character height and width dynamically.
   - **Groups Lines & Paragraphs**: Connects neighboring letters into lines, and stacks lines into blocks.
   - **Splits Multi-Columns**: Checks for empty vertical valleys (like the palm-leaf string hole) and splits text into parallel columns.
   - **Remaps Coordinates**: Converts bounding boxes back to the original unrotated image coordinates.

3. **Classification (`src/region_classifier.py`)**:
   - **Identifies Main Body**: Finds the largest central block of text lines.
   - **Classifies by Position**:
     - Modern scan stamps in top/bottom margins $\to$ `filler`
     - Text above the main body $\to$ `header`
     - Text below the main body $\to$ `footer`
     - Text on left or right margins $\to$ `side_text`
     - Main text lines $\to$ `main_text`
   - **Confidence Score**: Assigns a calibrated score ($0.75 - 0.98$) based on position and stroke density.

---

## 7. Output Format & JSON Example

For every processed image, two files are created in `--output`:
1. **Annotated Image** (`results/annotated/<filename>_annotated.png`): Visual image with color-coded bounding boxes and label badges.
2. **Prediction JSON** (`results/predictions/<filename>.json`): Structured data with coordinates and labels.

### Sample Output (`sample_paper.json`):
```json
{
  "filename": "sample_paper.jpg",
  "image_width": 1239,
  "image_height": 468,
  "skew_angle_deg": -0.4,
  "num_regions": 3,
  "regions": [
    {
      "bbox": [357, 13, 884, 37],
      "label": "filler",
      "confidence": 0.95
    },
    {
      "bbox": [142, 53, 1239, 447],
      "label": "main_text",
      "confidence": 0.98
    },
    {
      "bbox": [340, 435, 920, 460],
      "label": "filler",
      "confidence": 0.95
    }
  ]
}
```

---

## 8. Project Structure

```
Manuscript-Layout-Detection/
├── config.py                 # Central configuration (colors, thresholds)
├── requirements.txt          # Minimal dependencies (numpy, opencv, pillow, pytest)
├── inference.py              # Main CLI inference script
├── run_tests.py              # Automated test runner
├── pytest.ini                # Pytest configuration
├── .gitignore                # Git ignore rules
├── src/
│   ├── utils.py              # Safe image reading/writing, JSON saving, coordinate clipping
│   ├── preprocessing.py      # Illumination flattening, binarization, border removal, deskewing
│   ├── region_detector.py    # Character units, line grouping, multi-column splitting
│   ├── region_classifier.py  # 5-class layout classification and confidence scoring
│   └── visualization.py      # Color-coded box and badge drawing
├── data/
│   ├── test_images/          # Sample manuscript images for testing
│   └── input/                # Evaluation input images
├── results/                  # Generated outputs (annotated images and JSON predictions)
└── tests/
    └── test_pipeline.py      # 13 automated tests (100% passing)
```
