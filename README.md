# Hyperspectral Image Classification with Classical Machine Learning

This project presents a classical machine-learning pipeline for hyperspectral image classification using Python, OpenCV, NumPy, scikit-learn, and hyperspectral data processing techniques.

The project was developed as part of a practical university lab in Biomedical Engineering, with a focus on applying machine-learning methods to hyperspectral image data. The goal is to classify regions of interest from hyperspectral cubes using traditional machine-learning models and clinically relevant evaluation metrics.

## Project Overview

Hyperspectral imaging (HSI) captures spectral information across many wavelength bands for each pixel in an image. Compared to standard RGB images, hyperspectral data provides richer information that can be useful for biomedical image analysis, tissue characterization, and computer-assisted medical applications.

In this project, hyperspectral image data is processed and transformed into feature vectors that can be used for supervised classification. Classical machine-learning models are then trained and evaluated to distinguish between selected regions of interest.

## Key Features

- Loading and processing hyperspectral image cubes
- Mask-based region selection
- Feature extraction from hyperspectral data
- Train/test split for supervised classification
- Data scaling using `StandardScaler`
- Classification with classical machine-learning models
- Model evaluation using accuracy, sensitivity, specificity, F1-score, ROC-AUC, and confusion matrix
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
7. Evaluate the models using classification metrics
8. Save results for documentation and comparison

## Dataset Notice

The original hyperspectral dataset is not included in this repository due to file size limitations and course data-sharing restrictions.

The project expects the original raw data and preprocessed files to be placed locally in the following folders:

```text
data/
data_npz/
