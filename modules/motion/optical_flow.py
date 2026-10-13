import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def lucas_kanade_dense(previous, current, window=9):
    prev = cv2.GaussianBlur(previous.astype(np.float32), (5, 5), 0)
    curr = cv2.GaussianBlur(current.astype(np.float32), (5, 5), 0)
    ix = cv2.Sobel(prev, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(prev, cv2.CV_32F, 0, 1, ksize=3)
    it = curr - prev
    kernel = (window, window)
    sxx = cv2.boxFilter(ix * ix, -1, kernel)
    syy = cv2.boxFilter(iy * iy, -1, kernel)
    sxy = cv2.boxFilter(ix * iy, -1, kernel)
    sxt = cv2.boxFilter(ix * it, -1, kernel)
    syt = cv2.boxFilter(iy * it, -1, kernel)
    determinant = sxx * syy - sxy * sxy
    valid = determinant > 1e-5
    u = np.zeros_like(previous, dtype=np.float32)
    v = np.zeros_like(previous, dtype=np.float32)
    u[valid] = (sxy[valid] * syt[valid] - syy[valid] * sxt[valid]) / determinant[valid]
    v[valid] = (sxy[valid] * sxt[valid] - sxx[valid] * syt[valid]) / determinant[valid]
    return u, v, valid


def main():
    parser = argparse.ArgumentParser(description="Compare differential and sparse optical flow.")
    parser.add_argument("--video", default="data/conveyor.mp4")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    ok1, first = cap.read()
    ok2, second = cap.read()
    cap.release()
    if not (ok1 and ok2):
        raise FileNotFoundError(f"Could not read two frames from {args.video}")

    gray1 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    u, v, valid = lucas_kanade_dense(gray1, gray2)
    dense_vis = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    step = 18
    yy, xx = np.mgrid[step // 2:gray1.shape[0]:step, step // 2:gray1.shape[1]:step]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(dense_vis)
    axes[0].quiver(xx, yy, u[yy, xx], -v[yy, xx], color="yellow", angles="xy", scale_units="xy", scale=1)
    axes[0].set_title("Hand-built Lucas-Kanade")

    points = cv2.goodFeaturesToTrack(gray1, maxCorners=200, qualityLevel=0.01, minDistance=7)
    sparse_vis = dense_vis.copy()
    affine_vis = dense_vis.copy()
    if points is not None:
        next_points, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, points, None)
        good_old = points[status.ravel() == 1].reshape(-1, 2)
        good_new = next_points[status.ravel() == 1].reshape(-1, 2)
        for old, new in zip(good_old, good_new):
            cv2.arrowedLine(sparse_vis, tuple(old.astype(int)), tuple(new.astype(int)), (255, 0, 0), 1, tipLength=0.25)
        if len(good_old) >= 3:
            matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
            if inliers is not None:
                for old, new, inside in zip(good_old, good_new, inliers.ravel()):
                    color = (0, 255, 0) if inside else (255, 0, 0)
                    cv2.circle(affine_vis, tuple(new.astype(int)), 3, color, -1)
                print(f"Affine RANSAC inliers: {int(inliers.sum())}/{len(inliers)}")
                if matrix is not None:
                    displacement = np.linalg.norm(matrix[:, 2])
                    fps = cap_fps(args.video)
                    print(f"Approximate image displacement: {displacement:.2f} px/frame")
                    print(f"Video frame rate: {fps:.2f} fps")
                    print("Convert to cm/s using a measured pixel-to-centimeter scale.")
    axes[1].imshow(sparse_vis)
    axes[1].set_title("Sparse feature flow")
    axes[2].imshow(affine_vis)
    axes[2].set_title("RANSAC inliers (green) / outliers (red)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "optical_flow.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved flow visualizations to {path}")
    # Dense flow gives a vector at most pixels but can be unreliable on flat box surfaces.
    # Sparse flow follows corners and texture, so fewer points are available on plain surfaces.
    # Lighting changes and motion blur can violate brightness constancy.
    # Use a measured scale and frame rate before reporting speed in centimeters per second.


def cap_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 0.0


if __name__ == "__main__":
    main()
