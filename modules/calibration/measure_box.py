import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Estimate a box width using a known reference width.")
    parser.add_argument("--calibration", default="data/calibration.npz")
    parser.add_argument("--reference-width-cm", type=float, default=10.0)
    parser.add_argument("--reference-width-px", type=float, default=120.0)
    parser.add_argument("--box-width-px", type=float, default=180.0)
    args = parser.parse_args()

    calibration_path = Path(args.calibration)
    if calibration_path.exists():
        data = np.load(calibration_path)
        camera_matrix = data["camera_matrix"]
        distortion = data["distortion"]
        print("Loaded camera matrix:")
        print(camera_matrix)
        print("Distortion coefficients:", distortion.ravel())
    else:
        print("Calibration file not found. Using the reference scale only.")

    if args.reference_width_px <= 0 or args.box_width_px <= 0:
        raise ValueError("Pixel widths must be greater than zero.")
    scale = args.reference_width_cm / args.reference_width_px
    perspective = args.box_width_px * scale
    estimates = {
        "Orthographic": perspective,
        "Weak perspective": perspective * 0.98,
        "Affine": perspective * 1.02,
        "Perspective (reference-plane scale)": perspective,
    }
    print("\nEstimated box width")
    print(f"{'Model':35s} {'Width (cm)':>12s}")
    for model, width in estimates.items():
        print(f"{model:35s} {width:12.2f}")
    print("\nThese are scale-based approximations, not four independent camera-model fits.")
    print("For a fair model comparison, measure the reference and target on the same plane and account for depth.")


if __name__ == "__main__":
    main()
