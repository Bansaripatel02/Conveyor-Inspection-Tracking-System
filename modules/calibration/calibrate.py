import argparse
from pathlib import Path

import cv2
import numpy as np


def calibrate(image_dir, columns, rows, square_size, output_path):
    image_paths = sorted(
        list(Path(image_dir).glob("*.jpg")) +
        list(Path(image_dir).glob("*.jpeg")) +
        list(Path(image_dir).glob("*.png"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No checkerboard images found in {image_dir}")

    object_points = []
    image_points = []
    grid = np.zeros((rows * columns, 3), np.float32)
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size
    image_size = None

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, (columns, rows), None)
        if found:
            refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            object_points.append(grid.copy())
            image_points.append(refined)
            print(f"Used {path.name}")
        else:
            print(f"Checkerboard not found in {path.name}")

    if len(object_points) < 3:
        raise ValueError("At least three readable checkerboard views are needed; 10–15 are recommended.")

    error, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path, camera_matrix=camera_matrix, distortion=distortion,
        rotation=rotations[0], translation=translations[0], rms_error=error,
        image_width=image_size[0], image_height=image_size[1]
    )
    print(f"RMS reprojection error: {error:.4f}")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients: {distortion.ravel()}")
    print(f"Saved calibration to {output_path}")

    sample = cv2.imread(str(image_paths[0]))
    h, w = sample.shape[:2]
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, (w, h), 1, (w, h))
    undistorted = cv2.undistort(sample, camera_matrix, distortion, None, new_matrix)
    preview = np.hstack([cv2.resize(sample, (w // 2, h // 2)),
                         cv2.resize(undistorted, (w // 2, h // 2))])
    preview_path = output_path.parent / "undistortion_preview.png"
    cv2.imwrite(str(preview_path), preview)
    print(f"Saved before/after preview to {preview_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate a camera from checkerboard images.")
    parser.add_argument("--images", default="data/calibration")
    parser.add_argument("--output", default="data/calibration.npz")
    parser.add_argument("--columns", type=int, default=9, help="Number of inner corners across.")
    parser.add_argument("--rows", type=int, default=6, help="Number of inner corners down.")
    parser.add_argument("--square-size", type=float, default=2.5, help="Checker square size in cm.")
    args = parser.parse_args()
    calibrate(args.images, args.columns, args.rows, args.square_size, args.output)


if __name__ == "__main__":
    main()
