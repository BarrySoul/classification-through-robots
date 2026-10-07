# Robots.txt Ad-hoc Classifier

A lightweight machine learning project that explores whether a website’s `robots.txt` file can be used as a signal for website classification.

## Overview

This project evaluates a simple and low-cost classification pipeline based on the content of `robots.txt`. It combines a character-level TF-IDF representation with a LinearSVC model and wraps the workflow in a small Streamlit application for interactive use.

## Why this project?

Many websites expose structured directives in `robots.txt`, such as access rules, content paths, and admin endpoints. Instead of crawling the full website, this project tests whether those patterns are informative enough to support lightweight classification.

## Features

- Fetch `robots.txt` content from a website URL
- Accept pasted `robots.txt` text as input
- Load a trained model bundle from disk
- Predict the dominant category and top-k alternatives
- Run the application through a browser-based interface

## Tech stack

- Python 3
- Streamlit
- scikit-learn
- NumPy
- pandas
- joblib

## Project structure

```text
.
├── app.py                     # Streamlit interface
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore                # Git exclusions
├── models/                   # Trained model bundles
├── data/                     # Datasets and result files
│   ├── raw/
│   └── results/
├── src/
│   └── robots_classifier/    # Core ML and I/O logic
└──
```

## Setup

```bash
git clone <repository-url>
cd <repository-folder>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the app

```bash
python3 -m streamlit run app.py --server.headless true --server.port 8501
```

Then open the local URL displayed by Streamlit, usually:

```text
http://localhost:8501
```

## Model

The app expects a serialized model bundle in the `models/` directory. The bundle contains the TF-IDF vectorizer and the trained classifier used for inference.

## Notes

This project is intended as a prototype and proof of concept for `robots.txt`-based classification. The results suggest that `robots.txt` can provide a useful structural signal, but it is limited as a standalone feature for high-accuracy classification.

## Potential next steps

- improve the dataset and labeling quality
- compare additional classifiers
- add experiment tracking and model versioning
- combine `robots.txt` signals with other lightweight website features
- deploy the app as a public or internal dashboard


## Authors

This project was developed in collaboration with Léo Mafille.