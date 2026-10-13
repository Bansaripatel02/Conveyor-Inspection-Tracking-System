import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import felzenszwalb, watershed
from skimage.color import rgb2gray
from scipy import ndimage as ndi


def read_frames(video_path, count=5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), min(count, max(total, 1)), dtype=int)
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError("The video did not contain readable frames.")
    return frames


def split_regions(gray, block_size=32, variance_threshold=0.012):
    height, width = gray.shape
    labels = np.zeros((height, width), dtype=np.int32)
    next_label = 1

    def split(y0, y1, x0, x1):
        nonlocal next_label
        region = gray[y0:y1, x0:x1]
        if region.size == 0:
            return
        if region.size > block_size * block_size and np.var(region) > variance_threshold:
            ym = (y0 + y1) // 2
            xm = (x0 + x1) // 2
            if ym > y0 and xm > x0:
                split(y0, ym, x0, xm)
                split(y0, ym, xm, x1)
                split(ym, y1, x0, xm)
                split(ym, y1, xm, x1)
                return
        labels[y0:y1, x0:x1] = next_label
        next_label += 1

    split(0, height, 0, width)
    return labels


def colorize_labels(labels):
    labels = labels.astype(np.int32)
    rng = np.random.default_rng(7)
    colors = rng.integers(0, 255, size=(int(labels.max()) + 1, 3), dtype=np.uint8)
    colors[0] = 0
    return colors[labels]


def segment_methods(frame):
    small = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1]))) if frame.shape[1] > 640 else frame.copy()
    smooth = cv2.GaussianBlur(small, (5, 5), 0)
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    distance_input = cv2.threshold((gray * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    distance = cv2.distanceTransform(distance_input, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(distance, 0.35 * distance.max(), 255, 0)
    markers = ndi.label(sure_fg.astype(np.uint8))[0]
    watershed_labels = watershed(-distance, markers, mask=distance_input > 0)

    graph_labels = felzenszwalb(small, scale=100, sigma=0.8, min_size=100)

    pixels = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    # Mean shift can be expensive, so sample pixels and use k-means for a practical visual comparison.
    compact = cv2.resize(small, (min(160, small.shape[1]), min(120, small.shape[0])))
    lab = cv2.cvtColor(compact, cv2.COLOR_RGB2LAB)
    samples = lab.reshape(-1, 3).astype(np.float32)
    _, small_labels, centers = cv2.kmeans(samples, 5, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        3, cv2.KMEANS_PP_CENTERS)
    mean_shift_view = cv2.resize(colorize_labels(small_labels.reshape(lab.shape[:2])), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)

    split_labels = split_regions(gray)
    return small, {
        "Watershed": colorize_labels(watershed_labels),
        "Graph-based": colorize_labels(graph_labels),
        "Color clustering": mean_shift_view,
        "Region splitting": colorize_labels(split_labels),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare four segmentation methods on five video frames.")
    parser.add_argument("--video", default="data/conveyor.mp4")
    parser.add_argument("--output", default="outputs/segmentation_comparison.png")
    args = parser.parse_args()

    frames = read_frames(args.video, 5)
    fig, axes = plt.subplots(len(frames), 5, figsize=(16, 3.2 * len(frames)))
    if len(frames) == 1:
        axes = np.array([axes])
    for row, frame in enumerate(frames):
        original, results = segment_methods(frame)
        axes[row, 0].imshow(original)
        axes[row, 0].set_title("Original")
        for col, (name, image) in enumerate(results.items(), start=1):
            axes[row, col].imshow(image)
            axes[row, col].set_title(name)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved segmentation comparison to {output}")
    # Watershed can separate touching objects when its distance-transform markers are good.
    # Graph-based segmentation usually preserves local boundaries, but texture and shadows can create extra regions.
    # Region splitting is simple and often over-segments textured backgrounds.
    # The best result depends on the footage; inspect the saved grid before choosing a method.


if __name__ == "__main__":
    main()
