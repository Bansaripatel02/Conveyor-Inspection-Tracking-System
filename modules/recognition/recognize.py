import argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_crops(root, size=(64, 64)):
    images, labels, paths = [], [], []
    for folder in sorted(Path(root).iterdir()) if Path(root).exists() else []:
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            images.append(cv2.resize(image, size))
            labels.append(folder.name)
            paths.append(path)
    return images, np.array(labels), paths


def hu_features(image):
    moments = cv2.moments(image)
    hu = cv2.HuMoments(moments).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)


def alignment_score(query, template):
    orb = cv2.ORB_create(nfeatures=300)
    kp1, des1 = orb.detectAndCompute(query, None)
    kp2, des2 = orb.detectAndCompute(template, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return float("inf")
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
    if len(matches) < 4:
        return float("inf")
    matches = sorted(matches, key=lambda item: item.distance)[:40]
    source = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
    if homography is None:
        return float("inf")
    aligned = cv2.warpPerspective(query, homography, (template.shape[1], template.shape[0]))
    return float(np.mean(np.abs(aligned.astype(float) - template.astype(float))))


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate simple box-recognition methods.")
    parser.add_argument("--templates", default="data/templates")
    parser.add_argument("--query", default=None, help="Optional query crop image.")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    images, labels, paths = load_crops(args.templates)
    if len(images) < 6 or len(set(labels)) < 2:
        raise ValueError("Add labeled crop images under data/templates/<box_type>/ (at least two types and six images total).")

    features = np.array([image.reshape(-1) for image in images], dtype=np.float32)
    test_size = max(1 / len(labels), 0.25)
    try:
        x_train, x_test, y_train, y_test, img_train, img_test = train_test_split(
            features, labels, images, test_size=test_size, random_state=42, stratify=labels
        )
    except ValueError as exc:
        raise ValueError("Each class needs enough samples for a stratified train/test split.") from exc

    components = min(20, len(x_train) - 1, x_train.shape[1])
    pca = PCA(n_components=max(1, components), whiten=True, random_state=42)
    train_pca = pca.fit_transform(x_train)
    test_pca = pca.transform(x_test)
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(train_pca, y_train)
    predicted = classifier.predict(test_pca)
    pca_accuracy = accuracy_score(y_test, predicted)

    templates = {}
    for image, label in zip(img_train, y_train):
        templates.setdefault(label, []).append(image)
    align_predictions = []
    for image in img_test:
        scores = {
            label: min(alignment_score(image, template) for template in group)
            for label, group in templates.items()
        }
        align_predictions.append(min(scores, key=scores.get))
    align_accuracy = accuracy_score(y_test, align_predictions)

    hu_vectors = np.array([hu_features(image) for image in images])
    print("Recognition accuracy")
    print(f"{'Method':30s} {'Accuracy':>10s}")
    print(f"{'Alignment (ORB + homography)':30s} {align_accuracy:9.1%}")
    print(f"{'PCA eigenspace + nearest neighbor':30s} {pca_accuracy:9.1%}")
    print(f"Hu feature vectors calculated: {len(hu_vectors)}")
    print("Hu moments describe shape, but may not separate box types with similar silhouettes.")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    eigenvectors = pca.components_[:3]
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for index, ax in enumerate(axes):
        component = eigenvectors[index].reshape(64, 64)
        ax.imshow(component, cmap="gray")
        ax.set_title(f"Component {index + 1}")
        ax.axis("off")
    fig.tight_layout()
    eigen_path = output / "eigenbox_components.png"
    fig.savefig(eigen_path, dpi=150)
    plt.close(fig)
    print(f"Saved PCA components to {eigen_path}")

    if args.query:
        query = cv2.imread(args.query, cv2.IMREAD_GRAYSCALE)
        if query is None:
            raise FileNotFoundError(args.query)
        query = cv2.resize(query, (64, 64)).reshape(1, -1).astype(np.float32)
        print("PCA prediction:", classifier.predict(pca.transform(query))[0])
    # PCA can be affected by lighting and pose changes in the training crops.
    # Alignment may handle some rotation and perspective change when enough keypoints are present.
    # The held-out scores are more useful when the crop dataset contains varied lighting and viewpoints.


if __name__ == "__main__":
    main()
