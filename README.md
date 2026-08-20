# Hyperspectral Image Classification with Classical Machine Learning

This project presents a classical machine-learning pipeline for hyperspectral image classification using Python, OpenCV, NumPy, scikit-learn, and hyperspectral data processing techniques.

The project was developed as part of a practical university lab in Biomedical Engineering. It focuses on processing hyperspectral image data, extracting spectral features, and applying supervised machine-learning methods to classify selected regions of interest.

## Project Overview

Hyperspectral imaging (HSI) captures spectral information across many wavelength bands for each pixel in an image. Unlike standard RGB imaging, hyperspectral data provides rich spectral information that can be useful for biomedical image analysis, tissue characterization, and computer-assisted medical applications.

In this project, hyperspectral image cubes are processed and transformed into feature vectors. These features are then used to train and evaluate classical machine-learning models for supervised classification.

The main goal is to demonstrate a complete workflow from hyperspectral data handling to model evaluation.

## Key Features

- Loading and processing hyperspectral image cubes
- Mask-based region selection
- Spectral feature extraction from selected pixels
- Train/test split for supervised classification
- Feature scaling using StandardScaler
- Classification using classical machine-learning models
- Evaluation using biomedical classification metrics
- Clean project structure prepared for GitHub portfolio presentation

## Technologies Used

- Python
- NumPy
- OpenCV
- Matplotlib
- scikit-learn
- Hyperspectral image processing
- Classical machine learning

## Machine-Learning Workflow

The general workflow of the project is:

1. Load hyperspectral image data
2. Load and apply region masks
3. Extract spectral features from selected pixels
4. Split the dataset into training and test sets
5. Apply feature scaling
6. Train machine-learning classifiers
7. Evaluate model performance
8. Save results for documentation and comparison

## Dataset Notice

The original hyperspectral dataset is not included in this repository due to file size limitations and course data-sharing restrictions.

The project expects the original raw data and preprocessed files to be placed locally in the following folders:

    data/
    data_npz/

These folders are included only as placeholders using .gitkeep files so that the project structure remains visible on GitHub.

To run the project, place the required hyperspectral data files in the appropriate folders before executing the code.

## Large File Notice

Large data files such as .pkl, .npz, .dat, .npy, or raw hyperspectral files are intentionally excluded from this repository.

This keeps the repository lightweight and suitable for public portfolio presentation.

The file EL.pkl is not included because it appears to be a processed data object and is too large for a standard GitHub repository.

## Project Structure

    hsi-classification-classical-ml/
    ├── data/
    │   └── .gitkeep
    ├── data_npz/
    │   └── .gitkeep
    ├── results/
    │   └── .gitkeep
    ├── src/
    │   └── hsi_classification.py
    ├── utils/
    │   └── hypercube_data.py
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    └── LICENSE

## Installation

Clone the repository:

    git clone https://github.com/your-username/hsi-classification-classical-ml.git
    cd hsi-classification-classical-ml

Create a virtual environment:

    python -m venv .venv

Activate the virtual environment.

On Windows:

    .venv\Scripts\activate

On macOS/Linux:

    source .venv/bin/activate

Install the required dependencies:

    pip install -r requirements.txt

## Usage

Place the required hyperspectral data files locally inside:

    data/
    data_npz/

Then run the main script:

    python src/hsi_classification.py

Generated outputs such as evaluation metrics, plots, or result files can be saved in:

    results/

## Evaluation Metrics

The classification pipeline is designed to evaluate model performance using:

- Accuracy
- Sensitivity
- Specificity
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

These metrics are especially useful for biomedical classification tasks, where accuracy alone may not be sufficient.

## Notes on Reproducibility

Because the original dataset is not included, the repository does not provide fully reproducible benchmark results by default.

However, the code structure allows the pipeline to be executed once the required local hyperspectral data files are added.

This project is intended to demonstrate the implementation of a clean machine-learning workflow for hyperspectral biomedical image classification.

## Why This Project Matters

This project demonstrates how classical machine-learning methods can be applied to hyperspectral biomedical image data.

It connects core concepts from biomedical imaging, machine learning, feature extraction, and medical data analysis.

The project also highlights an important part of real-world medical AI development: preparing raw biomedical data, extracting meaningful features, evaluating model performance carefully, and documenting the workflow in a reproducible way.

## Future Improvements

Possible next steps include:

- Adding dimensionality reduction techniques such as PCA
- Comparing multiple classical machine-learning models
- Adding deep-learning models for hyperspectral image classification
- Visualizing spectral signatures of selected regions
- Adding example plots generated from a small public sample dataset
- Creating a lightweight demo dataset for reproducible execution
- Adding unit tests for preprocessing and feature extraction functions

## Repository Status

The code and project structure are prepared for portfolio presentation.

The original hyperspectral dataset and large processed data files are not included due to file size and data-sharing restrictions.

## Author

Biomedical Engineering Master's student with interests in medical imaging, hyperspectral imaging, computer vision, machine learning, and medical device development.
