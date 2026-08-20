"""Hyperspectral image classification with classical machine learning.

This script converts hyperspectral `.dat` cubes and binary masks into NumPy
archives, builds a tabular spectral dataset, trains classical ML models, and
compares their performance using clinically relevant binary-classification
metrics.

Expected input structure:
    data/
      sample_01.dat
      sample_01_mask.png
      sample_02.dat
      sample_02_mask.png

The mask convention follows the original lab assignment:
    white pixels  -> target/positive mask region -> label 0
    black pixels  -> background/negative region  -> label 1

Because scikit-learn treats class 1 as the default positive class, the metric
functions below make the positive label explicit and default to label 0.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm


@dataclass(frozen=True)
class CubeConfig:
    """Configuration used by the course-specific Cube_Read loader."""

    wave_area: int = 100
    first_nm: int = 8
    last_nm: int = 100


TARGET_LABEL = 0      # white mask pixels in the lab assignment
BACKGROUND_LABEL = 1  # black mask pixels in the lab assignment


def get_cube_reader():
    """Import the course-specific Cube_Read loader with a clear error message."""
    try:
        from utils.hypercube_data import Cube_Read  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import Cube_Read from utils.hypercube_data. "
            "Copy the course-provided utils/hypercube_data.py file into the "
            "repository before running conversion from .dat to .npz."
        ) from exc
    return Cube_Read


def find_dat_files(data_dir: Path) -> list[Path]:
    """Return all `.dat` files from the data directory in deterministic order."""
    return sorted(data_dir.glob("*.dat"))


def read_mask(mask_path: Path, expected_shape: Tuple[int, int]) -> np.ndarray:
    """Read a binary RGB/BGR mask and validate its spatial size."""
    mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask_bgr is None:
        raise FileNotFoundError(f"Mask file could not be read: {mask_path}")

    if mask_bgr.shape[:2] != expected_shape:
        raise ValueError(
            f"Mask shape {mask_bgr.shape[:2]} does not match cube shape "
            f"{expected_shape} for mask {mask_path}"
        )

    return mask_bgr


def extract_mask_indices(
    mask_bgr: np.ndarray,
    white_threshold: int = 250,
    black_threshold: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract target and background pixel indices from a binary mask.

    Thresholds are used instead of exact 255/0 checks to make the pipeline more
    robust to small compression or export artifacts in mask images.
    """
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

    target_indices = np.all(mask_rgb >= white_threshold, axis=-1)
    background_indices = np.all(mask_rgb <= black_threshold, axis=-1)

    if not np.any(target_indices):
        raise ValueError("No white/target pixels found in the mask.")
    if not np.any(background_indices):
        raise ValueError("No black/background pixels found in the mask.")

    return target_indices, background_indices


def convert_dat_to_npz(
    data_dir: Path,
    output_dir: Path,
    cube_config: CubeConfig,
) -> int:
    """Convert all `.dat` cubes and masks into `.npz` files with X and y keys."""
    Cube_Read = get_cube_reader()
    output_dir.mkdir(parents=True, exist_ok=True)

    dat_files = find_dat_files(data_dir)
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {data_dir}")

    saved_count = 0

    for dat_path in tqdm(dat_files, desc="Converting cubes"):
        spectrum_data, _ = Cube_Read(
            str(dat_path),
            wavearea=cube_config.wave_area,
            Firstnm=cube_config.first_nm,
            Lastnm=cube_config.last_nm,
        ).cube_matrix()

        if spectrum_data.ndim != 3:
            raise ValueError(
                f"Expected cube shape (height, width, bands), got {spectrum_data.shape}"
            )

        mask_path = dat_path.with_name(f"{dat_path.stem}_mask.png")
        mask_bgr = read_mask(mask_path, expected_shape=spectrum_data.shape[:2])
        target_idx, background_idx = extract_mask_indices(mask_bgr)

        target_spectra = spectrum_data[target_idx]
        background_spectra = spectrum_data[background_idx]

        X = np.vstack([target_spectra, background_spectra]).astype(np.float32)
        y = np.concatenate(
            [
                np.full(target_spectra.shape[0], TARGET_LABEL, dtype=np.int64),
                np.full(background_spectra.shape[0], BACKGROUND_LABEL, dtype=np.int64),
            ]
        )

        if X.shape[0] != y.shape[0]:
            raise RuntimeError("Feature and label counts do not match.")

        output_path = output_dir / f"{dat_path.stem}.npz"
        np.savez_compressed(output_path, X=X, y=y)
        saved_count += 1

    return saved_count


def load_npz_dataset(npz_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load and stack all `.npz` files containing X and y arrays."""
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for npz_path in tqdm(npz_files, desc="Loading npz files"):
        with np.load(npz_path) as data:
            if "X" not in data or "y" not in data:
                raise KeyError(f"{npz_path} must contain arrays named 'X' and 'y'.")
            X_parts.append(data["X"])
            y_parts.append(data["y"])

    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int64)

    if X.shape[0] != y.shape[0]:
        raise RuntimeError("Loaded feature and label counts do not match.")

    return X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a stratified train/test split.

    Scaling is intentionally not performed here. Standardization must be fitted
    on the training data only, so it is placed inside model pipelines.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_models(random_state: int = 42) -> Dict[str, object]:
    """Create classical ML models for spectral classification."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "Linear SVM": make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                random_state=random_state,
                max_iter=20000,
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def score_for_label(model: object, X: np.ndarray, positive_label: int) -> np.ndarray:
    """Return continuous scores for the selected positive label.

    This is important because the lab maps the target mask region to label 0,
    while many scikit-learn metrics assume class 1 is the positive class.
    """
    classes = np.asarray(getattr(model, "classes_"))

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        label_index = np.where(classes == positive_label)[0]
        if label_index.size != 1:
            raise ValueError(f"Positive label {positive_label} not found in model classes.")
        return proba[:, label_index[0]]

    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        if decision.ndim != 1 or len(classes) != 2:
            raise ValueError("Only binary decision_function output is supported.")

        # For binary sklearn classifiers, positive decision values correspond
        # to classes_[1]. Invert the score if the selected positive label is
        # classes_[0].
        if classes[1] == positive_label:
            return decision
        if classes[0] == positive_label:
            return -decision

    raise TypeError("Model must expose predict_proba or decision_function.")


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    positive_label: int = TARGET_LABEL,
) -> Dict[str, float]:
    """Compute accuracy, sensitivity, specificity, F1, and ROC-AUC."""
    labels = np.unique(y_true)
    if labels.size != 2:
        raise ValueError("This evaluation function supports binary labels only.")

    negative_label_candidates = [label for label in labels if label != positive_label]
    if len(negative_label_candidates) != 1:
        raise ValueError(f"Could not infer negative label from labels {labels}.")
    negative_label = int(negative_label_candidates[0])

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[negative_label, positive_label],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1 = f1_score(y_true, y_pred, pos_label=positive_label)

    y_true_binary = (y_true == positive_label).astype(int)
    try:
        roc_auc = roc_auc_score(y_true_binary, y_score)
    except ValueError:
        roc_auc = float("nan")

    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1,
        "ROC_AUC": roc_auc,
    }


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    positive_label: int = TARGET_LABEL,
    test_size: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train models and return a sorted result table."""
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    rows = []
    for model_name, model in build_models(random_state=random_state).items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = score_for_label(model, X_test, positive_label=positive_label)
        metrics = evaluate_predictions(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            positive_label=positive_label,
        )
        rows.append({"Model": model_name, **metrics})

    results = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)
    return results.reset_index(drop=True)


def save_results(results: pd.DataFrame, output_path: Path) -> None:
    """Save model comparison results to CSV and JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    json_path = output_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(results.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify hyperspectral imaging spectra using classical ML models."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--npz-dir", type=Path, default=Path("data_npz"))
    parser.add_argument("--results", type=Path, default=Path("results/model_results.csv"))
    parser.add_argument("--convert", action="store_true", help="Convert .dat files to .npz first.")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--positive-label", type=int, default=TARGET_LABEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.convert:
        count = convert_dat_to_npz(
            data_dir=args.data_dir,
            output_dir=args.npz_dir,
            cube_config=CubeConfig(),
        )
        print(f"Converted {count} dat files to npz archives.")

    X, y = load_npz_dataset(args.npz_dir)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print("Class distribution:", dict(zip(*np.unique(y, return_counts=True))))

    results = train_and_evaluate(
        X,
        y,
        positive_label=args.positive_label,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\nModel comparison:")
    print(results.to_string(index=False))

    best_model = results.iloc[0]["Model"]
    print(f"\nBest model by ROC-AUC: {best_model}")

    save_results(results, args.results)
    print(f"Saved results to {args.results}")


if __name__ == "__main__":
    main()
