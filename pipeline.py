import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.tracking.kalman_tracker import SimpleKalmanFilter, detect_centroid


def segment_box(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) > 500]
    if not candidates:
        return None, None
    contour = max(candidates, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None, None
    center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    return center, (x, y, w, h)


def main():
    parser = argparse.ArgumentParser(description="Run the conveyor inspection pipeline.")
    parser.add_argument("--video", default="data/conveyor.mp4")
    parser.add_argument("--scale-cm-per-pixel", type=float, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    tracker = None
    rows = []
    frame_index = 0
    first_frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if first_frame is None:
            first_frame = frame.copy()
        center, box = segment_box(frame)
        if tracker is None and center is not None:
            tracker = SimpleKalmanFilter(*center)
        if tracker is not None:
            predicted = tracker.predict()
            estimate = tracker.update(center) if center is not None else tracker.state[:2, 0].copy()
            speed_px_s = float(np.linalg.norm(tracker.state[2:4, 0]) * fps)
            speed_cm_s = speed_px_s * args.scale_cm_per_pixel if args.scale_cm_per_pixel else None
            width_cm = box[2] * args.scale_cm_per_pixel if box and args.scale_cm_per_pixel else None
            rows.append((frame_index, *estimate, speed_px_s, speed_cm_s, width_cm))
            if box:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, tuple(estimate.astype(int)), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID 1  speed {speed_px_s:.1f} px/s",
                            (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        frame_index += 1
    cap.release()

    if not rows:
        raise ValueError("No boxes detected. Use a video with a clear object and background.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = np.array(rows, dtype=object)
    plt.figure(figsize=(9, 5))
    plt.plot(result[:, 1].astype(float), result[:, 2].astype(float), label="Tracked centroid")
    plt.gca().invert_yaxis()
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.title("Pipeline tracking path")
    plt.legend()
    plt.tight_layout()
    path = output / "pipeline_trajectory.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Frame | x (px) | y (px) | speed (px/s) | speed (cm/s) | width (cm)")
    for row in rows[:20]:
        print(f"{row[0]:5d} | {row[1]:6.1f} | {row[2]:6.1f} | {row[3]:13.2f} | "
              f"{row[4] if row[4] is not None else 'n/a':>12} | {row[5] if row[5] is not None else 'n/a':>10}")
    print(f"Saved trajectory to {path}")
    print("This baseline uses a single largest contour and one track ID. Multiple-object association and recognition require additional data.")


if __name__ == "__main__":
    main()
