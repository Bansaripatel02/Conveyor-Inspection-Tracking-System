import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


class SimpleKalmanFilter:
    def __init__(self, x, y, dt=1.0):
        self.state = np.array([[x], [y], [0.0], [0.0]], dtype=float)
        self.transition = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        self.measurement = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.process_noise = np.eye(4) * 0.05
        self.measurement_noise = np.eye(2) * 8.0
        self.covariance = np.eye(4) * 10.0

    def predict(self):
        self.state = self.transition @ self.state
        self.covariance = self.transition @ self.covariance @ self.transition.T + self.process_noise
        return self.state[:2, 0].copy()

    def update(self, position):
        z = np.array(position, dtype=float).reshape(2, 1)
        innovation = z - self.measurement @ self.state
        innovation_covariance = self.measurement @ self.covariance @ self.measurement.T + self.measurement_noise
        gain = self.covariance @ self.measurement.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(4) - gain @ self.measurement) @ self.covariance
        return self.state[:2, 0].copy()


def detect_centroid(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) > 500]
    if not candidates:
        return None
    contour = max(candidates, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])


def main():
    parser = argparse.ArgumentParser(description="Track a box centroid with a Kalman filter.")
    parser.add_argument("--video", default="data/conveyor.mp4")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--dropout-start", type=int, default=20)
    parser.add_argument("--dropout-length", type=int, default=4)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    measurements, estimates, predictions = [], [], []
    kalman = None
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        measurement = detect_centroid(frame)
        if kalman is None:
            if measurement is None:
                frame_index += 1
                continue
            kalman = SimpleKalmanFilter(*measurement)
        predicted = kalman.predict()
        is_dropout = args.dropout_start <= frame_index < args.dropout_start + args.dropout_length
        if measurement is not None and not is_dropout:
            estimate = kalman.update(measurement)
            measurements.append((frame_index, *measurement))
        else:
            estimate = kalman.state[:2, 0].copy()
        predictions.append((frame_index, *predicted))
        estimates.append((frame_index, *estimate))
        if frame_index < 10:
            print(f"Frame {frame_index}: predicted={predicted.round(1)}, "
                  f"measurement={None if measurement is None else measurement.round(1)}, "
                  f"estimate={estimate.round(1)}")
        frame_index += 1
    cap.release()
    if not estimates:
        raise ValueError("No object detections were found. Try a video with a clear box and background.")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    measurements = np.array(measurements) if measurements else np.empty((0, 3))
    estimates = np.array(estimates)
    predictions = np.array(predictions)
    plt.figure(figsize=(9, 6))
    if len(measurements):
        plt.plot(measurements[:, 1], measurements[:, 2], "o", markersize=3, label="Raw detections")
    plt.plot(estimates[:, 1], estimates[:, 2], "-", label="Kalman estimate")
    plt.gca().invert_yaxis()
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.title("Box centroid tracking")
    plt.legend()
    plt.tight_layout()
    path = output / "kalman_trajectory.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved tracking plot to {path}")
    print(f"Simulated measurement dropout: frames {args.dropout_start}–{args.dropout_start + args.dropout_length - 1}")


if __name__ == "__main__":
    main()
