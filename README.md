# Smart Parking AI

**ISTE-500/501 Senior Development Project**
**Team:** Dilshod, Walid, Sayed, Meiir

A computer-vision-based smart parking detection system that uses YOLOv8 to detect vehicles in parking lot images and determine slot occupancy. The project includes four different detection approaches, each with its own trade-offs.

---

## Prerequisites

- Python 3.12+
- A virtual environment is included (`venv/`)

### Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` (cv2) | Image processing, drawing, morphological operations |
| `ultralytics` (YOLO) | YOLOv8 object detection model |
| `numpy` | Numerical operations, array handling |
| `scikit-learn` | DBSCAN clustering (used by `smart_parking_lines.py`) |

### Setup

```bash
# Activate the virtual environment
source venv/bin/activate

# Or install dependencies manually
pip install opencv-python ultralytics numpy scikit-learn
```

The YOLOv8 nano model (`yolov8n.pt`) is included in the repository root. All scripts default to using it.

---

## Python Files

### 1. `smart_parking_simple.py` (Recommended for demos)

**Purpose:** The simplest and most reliable version. Optimized for live presentations and demos where consistent, predictable results matter most.

**How it works:**
1. Detects all vehicles in the image using YOLOv8 (confidence threshold: 0.25)
2. Groups detected vehicles into horizontal rows by clustering their Y-coordinates
3. Calculates the median vehicle size and uses it to define a uniform slot grid (1.25x the median vehicle dimensions)
4. Extends the grid 2 slots beyond the outermost vehicles on each side to show potential empty spaces
5. Checks each slot for occupancy using center-point containment and bounding-box overlap (20% threshold)

**Usage:**
```bash
python smart_parking_simple.py <image_path>
```

**Example:**
```bash
python smart_parking_simple.py parking_lot.jpg
```

**Output:** Saves the annotated result to `result.jpg` and displays it in a window. Press any key to close.

**When to use:** Best choice for presentations and demos. Produces clean, uniform grids that are easy to understand visually.

---

### 2. `smart_parking_detection.py`

**Purpose:** A multi-mode detection script that supports vehicle detection, manual slot definition, and detection with pre-defined slots.

**How it works — three modes:**

**Mode 1: Vehicle Detection (default)**
- Runs YOLOv8 on the input image and draws bounding boxes around all detected vehicles (cars, motorcycles, buses, trucks)
- Displays a stats panel with vehicle count and breakdown by type

**Mode 2: Interactive Slot Definition (`--define`)**
- Opens the image in a GUI window where you click to define parking slot polygons
- Click 4 corners per slot, then press `n` to confirm
- Keyboard controls:
  - `n` — finish current slot, start next
  - `s` — save all slots to a JSON file
  - `u` — undo last point
  - `r` — reset current slot
  - `q` — quit

**Mode 3: Slot-Based Detection (`--config`)**
- Loads a previously saved slot configuration (JSON file with polygon coordinates)
- Runs YOLO detection and checks each predefined slot for vehicle overlap (30% threshold)
- Colors slots green (available) or red (occupied) with a semi-transparent overlay

**Usage:**
```bash
# Detect vehicles only
python smart_parking_detection.py <image_path>

# Interactively define parking slots
python smart_parking_detection.py <image_path> --define

# Detect using saved slot configuration
python smart_parking_detection.py <image_path> --config slots.json
```

**Examples:**
```bash
python smart_parking_detection.py parking_lot.jpg
python smart_parking_detection.py parking_lot.jpg --define
python smart_parking_detection.py parking_lot.jpg --config slots.json
```

**Output:** Saves annotated result to `detection_result.jpg`.

**When to use:** Use this when you need manual control over slot placement, or when working with a parking lot where automatic slot detection doesn't produce good results. The `--define` mode lets you create a `slots.json` once, then reuse it for repeated analysis.

---

### 3. `smart_parking_auto.py`

**Purpose:** Automatically detects parking slots without any manual input by analyzing vehicle positions and parking lot line markings.

**How it works:**
1. Detects vehicles using YOLOv8 (confidence threshold: 0.25)
2. Estimates slot dimensions from average detected vehicle sizes (1.25x multiplier)
3. Attempts automatic slot detection using one of three methods:
   - **Lines method:** Uses Canny edge detection + Hough Line Transform to find slot boundary lines, then clusters vertical/horizontal lines into slot regions
   - **Vehicles method:** Groups vehicles into rows by Y-coordinate, estimates slot spacing from vehicle center distances, and generates a slot grid
   - **Hybrid method (default):** Combines both approaches — detects vehicles first to estimate dimensions, groups them into rows, calculates spacing from inter-vehicle distances, then generates slots and matches vehicles to them
4. Matches vehicles to slots using bounding-box overlap (20% threshold) and center-point containment
5. Falls back to vehicle-only mode if no slots can be detected

**Usage:**
```bash
python smart_parking_auto.py <image_path>
```

**Example:**
```bash
python smart_parking_auto.py parking_lot.jpg
```

**Output:** Saves annotated result to `detection_result.jpg` with a stats panel showing total slots, occupied/available count, and an occupancy bar.

**When to use:** Good for quick automated analysis where you don't want to manually define slots. Works best with top-down or near-top-down parking lot images where vehicles are arranged in clear rows.

---

### 4. `smart_parking_lines.py`

**Purpose:** The most advanced detection approach. Detects parking slots by finding white painted line markings on the pavement, then checks for vehicle occupancy.

**How it works:**
1. **White line detection** — uses three combined methods:
   - HSV color filtering (low saturation, high value)
   - Grayscale thresholding (brightness > 200)
   - Adaptive Gaussian thresholding
2. **Line segment detection** — applies Canny edge detection followed by Probabilistic Hough Transform to find line segments
3. **Line classification** — separates lines into vertical (slot dividers, 60-120 degrees) and horizontal (row boundaries, 0-30 degrees)
4. **Line clustering** — uses DBSCAN to merge nearby duplicate line detections into single dividers
5. **Slot detection** — tries four methods and picks the best result:
   - Connected components on inverted line mask
   - Contour analysis of line regions
   - Vertical line-based grid generation
   - Horizontal line-based band detection
6. **Vehicle matching** — detects vehicles with YOLO (confidence threshold: 0.15 for better recall) and matches them to slots using overlap ratio (15% threshold) and center containment

**Usage:**
```bash
python smart_parking_lines.py <image_path>
python smart_parking_lines.py <image_path> --debug
```

**Example:**
```bash
python smart_parking_lines.py parking_lot.jpg
python smart_parking_lines.py parking_lot.jpg --debug
```

The `--debug` flag saves intermediate processing images (`debug_dilated.jpg`, `debug_closed.jpg`, `debug_inverted.jpg`, `debug_cleaned.jpg`) for troubleshooting the line detection pipeline.

**Output:** Saves annotated result to `detection_result.jpg`.

**When to use:** Best for parking lots with clearly visible white line markings. The `--debug` flag is helpful for tuning detection on new images. Requires `scikit-learn` for DBSCAN clustering.

---

## Other Files

| File | Description |
|---|---|
| `slots.json` | Example slot configuration with 16 polygon-defined parking spaces (created via the `--define` mode in `smart_parking_detection.py`) |
| `yolov8n.pt` | Pre-trained YOLOv8 nano model weights (COCO dataset) |
| `parking_lot.jpg`, `parking_lot_2.jpg`, `sample_parking.jpg`, `sample_parkingv2.jpg` | Sample parking lot images for testing |
| `result.jpg`, `detection_result.jpg` | Output images from previous runs |
| `debug_*.jpg` | Debug output images from `smart_parking_lines.py --debug` |
| `venv/` | Python virtual environment |

---

## Quick Comparison

| Script | Slot Detection | Manual Setup | Best For |
|---|---|---|---|
| `smart_parking_simple.py` | Vehicle-based grid | None | Demos and presentations |
| `smart_parking_detection.py` | Manual polygon definition | Click-to-define GUI | Precise slot boundaries |
| `smart_parking_auto.py` | Hybrid (lines + vehicles) | None | Quick automated analysis |
| `smart_parking_lines.py` | Line marking detection | None (optional `--debug`) | Lots with visible painted lines |

---

## Typical Workflow

```bash
# 1. Quick demo — just run the simple version
python smart_parking_simple.py parking_lot.jpg

# 2. For precise results — define slots once, reuse forever
python smart_parking_detection.py parking_lot.jpg --define
# (click corners, press 'n' after each slot, press 's' to save)
python smart_parking_detection.py parking_lot.jpg --config slots.json

# 3. Fully automatic — let the AI figure out slots
python smart_parking_auto.py parking_lot.jpg

# 4. Line-based — for lots with clear white markings
python smart_parking_lines.py parking_lot.jpg --debug
```
