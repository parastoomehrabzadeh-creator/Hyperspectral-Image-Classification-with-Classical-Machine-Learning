# Hyperspectral Image Classification with Classical Machine Learning

A clean machine-learning pipeline for classifying hyperspectral imaging (HSI) spectra using classical ML models. The project converts hyperspectral `.dat` cubes and segmentation masks into structured NumPy datasets, trains multiple classifiers, and compares them using clinically relevant binary-classification metrics.

This repository was created from a biomedical engineering lab project and refactored into a professional, reproducible GitHub portfolio project.

---

## Project Motivation

Hyperspectral imaging captures spectral information across many wavelength bands and can reveal tissue properties that are not visible in conventional RGB images. In medical image analysis, HSI can support computer-assisted diagnosis, tissue characterization, and AI-based decision-support systems.

The goal of this project is to build a structured baseline pipeline for HSI classification using classical machine-learning methods before moving toward deeper learning approaches.

---

## What This Project Does

- Loads hyperspectral `.dat` files using the course-provided cube reader
- Reads corresponding binary mask images
- Extracts target and background spectra from mask regions
- Saves processed spectra as compressed `.npz` archives
- Builds a tabular spectral dataset from all processed files
- Uses a stratified train/test split
- Prevents data leakage by fitting scaling only on training data through scikit-learn pipelines
- Trains and compares:
  - Logistic Regression
  - Linear Support Vector Machine
  - Random Forest
- Evaluates each model using:
  - Accuracy
  - Sensitivity
  - Specificity
  - F1-score
  - ROC-AUC

---

## Repository Structure

```text
hsi-classification-classical-ml/
│
├── data/                  # Original .dat files and mask images (not included)
├── data_npz/              # Processed .npz files generated from raw data
├── results/               # Model comparison outputs
├── src/
│   └── hsi_classification.py
├── utils/
│   └── hypercube_data.py  # Course-provided loader; add it here before running
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data Format

The expected raw input structure is:

```text
data/
├── sample_01.dat
├── sample_01_mask.png
├── sample_02.dat
└── sample_02_mask.png
```

Each `.dat` file must have a corresponding mask file with the same base name and the suffix `_mask.png`.

Example:

```text
patient_001.dat
patient_001_mask.png
```

The mask convention follows the original lab assignment:

| Mask pixel | Meaning | Label |
|---|---:|---:|
| White | Target/positive mask region | 0 |
| Black | Background/negative region | 1 |

The evaluation code makes the positive label explicit because scikit-learn often treats class `1` as the default positive class, while this lab uses class `0` for the target region.

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Convert raw `.dat` files to `.npz`

Before running this step, copy the course-provided `utils/hypercube_data.py` file into the `utils/` folder.

```bash
python src/hsi_classification.py --convert
```

This creates compressed NumPy archives in `data_npz/` with two keys:

```text
X = spectral features
y = class labels
```

### 2. Train and evaluate models

```bash
python src/hsi_classification.py
```

The script prints a model comparison table and saves results to:

```text
results/model_results.csv
results/model_results.json
```

---

## Methods

### Preprocessing

The pipeline extracts spectra from the spatial regions defined by the masks. Each pixel becomes one spectral sample, where the wavelength bands are used as features.

Scaling is performed inside the scikit-learn model pipelines. This avoids a common machine-learning error: fitting `StandardScaler` on the full dataset before the train/test split.

### Models

Three classical machine-learning models are evaluated:

1. **Logistic Regression**  
   A strong linear baseline for binary classification.

2. **Linear Support Vector Machine**  
   A robust margin-based classifier suitable for high-dimensional spectral data.

3. **Random Forest**  
   A non-linear ensemble model that can capture more complex feature interactions.

---

## Evaluation Metrics

The project reports:

- **Accuracy** — overall proportion of correct predictions
- **Sensitivity** — true positive rate for the selected positive class
- **Specificity** — true negative rate
- **F1-score** — harmonic mean of precision and recall
- **ROC-AUC** — ranking quality of the model scores

The best model is selected by ROC-AUC because it is less dependent on a fixed classification threshold.

---

## Key Improvements over the Original Notebook

The original lab notebook was useful for learning, but it still contained starter-code placeholders and notebook-specific issues. This refactored version improves it by:

- Removing hard-coded local paths
- Removing `pip install` commands from notebook cells
- Replacing unfinished `None` placeholders with reusable functions
- Avoiding data leakage during scaling
- Using stratified train/test splitting
- Handling class-label direction explicitly
- Adding robust mask validation
- Adding a reusable command-line workflow
- Saving results as CSV and JSON
- Making the project suitable for a GitHub portfolio

---

## Future Improvements

- Add cross-validation for more reliable model comparison
- Add model calibration and threshold optimization
- Add spectral band importance analysis
- Add visualization of mean spectra per class
- Compare classical ML models with CNN or 1D deep-learning models
- Extend the pipeline to patient-level splitting if patient metadata is available

---

## Technologies Used

- Python
- NumPy
- OpenCV
- pandas
- scikit-learn
- Hyperspectral Imaging
- Classical Machine Learning
- Medical Image Analysis

---

## Repository Description

Classical machine-learning pipeline for hyperspectral medical image classification using Python, OpenCV, NumPy, and scikit-learn.

---

## Suggested GitHub Topics

```text
hyperspectral-imaging
medical-imaging
machine-learning
biomedical-engineering
image-processing
computer-vision
python
opencv
scikit-learn
classification
medical-ai
```
