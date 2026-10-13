# Camera-Based Conveyor Inspection and Tracking

A Python project that uses classical computer vision to detect, measure, track, and recognize boxes moving on a conveyor-like surface.

## Requirements

- Python 3.10 or newer
- A real video clip showing boxes moving across a table or conveyor
- Optional checkerboard images for camera calibration
- OpenCV, NumPy, Matplotlib, scikit-image, and scikit-learn

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Put your video in `data/conveyor.mp4`. Put checkerboard photos in `data/calibration/` if you want to run calibration. You can use `--video` to provide a different video path.

## Run the modules

```bash
python modules/segmentation/compare_segmentation.py --video data/conveyor.mp4
python modules/calibration/calibrate.py --images data/calibration --output data/calibration.npz
python modules/calibration/measure_box.py --calibration data/calibration.npz
python modules/motion/optical_flow.py --video data/conveyor.mp4
python modules/tracking/kalman_tracker.py --video data/conveyor.mp4
python modules/recognition/recognize.py --templates data/templates --video data/conveyor.mp4
python pipeline.py --video data/conveyor.mp4
```

The calibration and recognition commands need their respective input data. For calibration, use a printed checkerboard and enter the number of inner corners and square size when prompted or through command-line arguments. For recognition, place labeled crop images in folders such as `data/templates/type_a/`, `data/templates/type_b/`, and `data/templates/type_c/`.

## Modules

- **Segmentation:** compares watershed, graph-based segmentation, mean shift, and region splitting on five frames. The output is a comparison image.
- **Calibration:** estimates camera intrinsics and lens distortion from checkerboard images. The measurement script compares approximate orthographic, weak-perspective, affine, and perspective estimates using a known reference box.
- **Motion:** implements a local-window Lucas–Kanade optical-flow estimate and compares it with OpenCV sparse feature tracking. It also estimates an affine flow model with RANSAC.
- **Tracking:** uses a Kalman filter with state `[x, y, vx, vy]` to smooth the centroid path and demonstrates a short simulated measurement dropout.
- **Recognition:** uses template alignment, PCA eigenspace nearest-neighbor classification, and Hu moments. It reports a simple held-out accuracy estimate when enough images are available.
- **Pipeline:** combines segmentation, centroid tracking, approximate pixel-to-centimeter conversion when calibration scale is provided, motion estimation, and optional box-type recognition.

## Calibration notes

A checkerboard calibration gives camera intrinsics and distortion coefficients. Converting image measurements into real-world centimeters also needs a known plane/scale or a suitable reference object. A calibration file alone does not automatically establish a reliable centimeter scale for every object depth. The measurement script therefore asks for a known reference width and its measured pixel width.

## Recognition data

For a basic dataset, collect at least ten crops for each of three box types. Keep the folder names as the labels. The script uses a simple split of the available images for training and testing; results are only meaningful when the test crops are different from the training crops.

## Output

Generated images and tables are saved under `outputs/`. Input videos, calibration photos, and crop datasets are not included in this repository.

This is a practical student project. Results depend on lighting, camera position, background, and the quality of the input video.
