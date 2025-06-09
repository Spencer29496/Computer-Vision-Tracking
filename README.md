# Computer Vision Tracking

A comprehensive implementation of object tracking algorithms using Lucas-Kanade optical flow methods. This project implements multiple variants of the Lucas-Kanade algorithm for tracking objects in video sequences.

## Overview

This project contains implementations of:

1. **Lucas-Kanade Tracker** - Basic translation-only tracking using optical flow
2. **Lucas-Kanade Affine Tracker** - Enhanced tracking with affine transformations (scaling, rotation, translation)
3. **Inverse Compositional Affine Tracker** - More efficient affine tracking using inverse compositional formulation

The algorithms are tested on three different datasets: `car1`, `car2`, and `landing`, each containing video sequences for object tracking evaluation.

## Project Structure

```
├── python/
│   ├── LucasKanade.py              # Basic Lucas-Kanade implementation
│   ├── LucasKanadeAffine.py        # Affine Lucas-Kanade implementation
│   ├── InverseCompositionAffine.py # Inverse compositional affine tracker
│   ├── test_lk.py                  # Test basic Lucas-Kanade tracker
│   ├── test_lk_affine.py           # Test affine Lucas-Kanade tracker
│   ├── test_ic_affine.py           # Test inverse compositional tracker
│   └── file_utils.py               # Utility functions for file operations
├── data/
│   ├── car1.npy                    # Car tracking sequence 1
│   ├── car2.npy                    # Car tracking sequence 2
│   └── landing.npy                 # Landing sequence
└── results/                        # Output directory for tracking results
```

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Computer-Vision-Tracking
```

2. Activate the virtual environment (if available):
```bash
source venv/bin/activate
```

3. If no virtual environment exists, create one and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib
```

**Note:** This project includes a pre-configured virtual environment in the `venv/` directory with all required dependencies installed.

## Usage

### Basic Lucas-Kanade Tracker

Track objects using translation-only motion model:

```bash
cd python
python3 test_lk.py [dataset] [display_flag]
```

**Parameters:**
- `dataset`: Choose from `car1`, `car2`, or `landing` (default: `car1`)
- `display_flag`: Set to `0` to disable live display, `1` to enable (default: `1`)

**Examples:**
```bash
# Track car1 with live display
python3 test_lk.py car1 1

# Track landing sequence without display
python3 test_lk.py landing 0

# Use default parameters (car1 with display)
python3 test_lk.py
```

### Affine Lucas-Kanade Tracker

Track objects with affine transformations (handles scaling, rotation, translation):

```bash
cd python
python3 test_lk_affine.py [dataset] [display_flag]
```

**Parameters:**
- `dataset`: Choose from `car1`, `car2`, or `landing` (default: `car2`)
- `display_flag`: Set to `0` to disable live display, `1` to enable (default: `1`)

### Inverse Compositional Affine Tracker

More efficient affine tracking implementation:

```bash
cd python
python3 test_ic_affine.py [dataset] [display_flag]
```

**Parameters:**
- `dataset`: Choose from `car1`, `car2`, or `landing` (default: `car2`)
- `display_flag`: Set to `0` to disable live display, `1` to enable (default: `1`)

## Algorithm Details

### Lucas-Kanade Tracker
- Implements the classic Lucas-Kanade optical flow algorithm
- Tracks objects using translation-only motion model
- Iteratively minimizes the sum of squared differences between template and warped image

### Affine Lucas-Kanade Tracker
- Extends basic Lucas-Kanade to handle affine transformations
- Can track objects undergoing scaling, rotation, shearing, and translation
- Uses 6-parameter affine warp model

### Inverse Compositional Affine Tracker
- More computationally efficient variant of affine tracking
- Pre-computes gradient and Hessian on the template image
- Reduces computational cost per iteration

## Output

- Tracking results are saved as image sequences in the `results/` directory:
  - `results/lk/[dataset]/` - Basic Lucas-Kanade tracker results
  - `results/lk_affine/[dataset]/` - Affine Lucas-Kanade tracker results  
  - `results/ic_affine/[dataset]/` - Inverse compositional tracker results
- Each frame shows the tracked bounding box overlaid on the current image
- Results can be viewed as individual frames or compiled into videos

## Dataset Information

- **car1**: Vehicle tracking in urban environment (86MB, high resolution)
- **car2**: Vehicle tracking sequence 2 (30MB, medium resolution)  
- **landing**: Aircraft landing sequence (52MB, aerial view)

Each dataset is stored as a NumPy array with shape `(height, width, num_frames)`.

## Performance Notes

- Basic Lucas-Kanade is fastest but limited to translation
- Affine trackers handle more complex motion but are computationally intensive
- Inverse compositional method provides good balance of accuracy and speed
- Processing time varies significantly based on sequence resolution and length