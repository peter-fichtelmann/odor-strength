# Odor Strength Module

[![DOI](https://zenodo.org/badge/1100439173.svg)](https://doi.org/10.5281/zenodo.17660448)

This module contains the Python code (version 1.0) of the manuscript: Machine learning for smell: Ordinal odor strength prediction of molecular perfumery components (DOI coming soon).
It implements dataset curation, as well as training and evaluating machine learning models to predict odor strength from molecular structures via various molecular encoders and predictive algorithms.

## Overview

The odor strength prediction pipeline involves:
1. **Dataset Curation**: Web scraping and data cleaning of the curated dataset
2. **Dataset Analysis**: Exploratory data analysis of the chemical odorous space and visualization
3. **Model Training & Validation & Insights**: Hyperparameter optimization, performance evaluation and feature importance analysis
4. **Model Application**: Applying the best-performing model of the study for predictions on novel molecules

## Prerequisites

- Python 3.12.2
- Required packages are listed in requirements.txt file

## Workflow

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/peter-fichtelmann/odor-strength.git
```

### 2. Set up environment

Install Python 3.12.2 (e.g. via conda)
```bash
conda install python=3.12.2
```

Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset Curation (Required First Step)

Execute before dataset analysis and  scripts (Runtime up to 12h):

```bash
python dataset_curation.py
```

or run the Jupyter notebook:
```bash
jupyter notebook dataset_curation.ipynb
```

This script:
- Web scrapes data from Good Scents and PubChem
- Cleans and processes molecular data
- Creates the curated dataset files (`df_odor_strength.csv`, `odor_strength_groups.csv`)

### 4. Dataset Analysis (Optional)

```bash
python dataset_analysis.py
```

or run the Jupyter notebook:
```bash
jupyter notebook dataset_analysis.ipynb
```

This script:
- Performs exploratory data analysis on the curated dataset
- Creates dimensionality reduction visualizations (PCA, UMAP)
- Tests unsupervised clustering
- **Requires**: Dataset curation to be completed first

### 5. Model Training & Validation & Insights (Optional)

```bash
python model_training_validation_insights.py
```

or run the Jupyter notebook:
```bash
jupyter notebook model_training_validation_insights.ipynb
```

This script:
- Trains various molecular encoder-predictor combinations
- Performs hyperparameter optimization
- Evaluates model performance with cross-validation
- Conducts SHAP analysis for feature importance
- Tests on external validation datasets from Keller et al.
- **Requires**: Dataset curation to be completed first

### 6. Model Application (Independent)

```bash
jupyter notebook best-performing_model_application.ipynb
```

This notebook:
- Loads best-performing model from the study
- Demonstrates how to use models for prediction on new SMILES strings
- **Independent**: Can be run without running other scripts (uses pre-trained models)

## Output Files

After running the scripts, the following key files will be generated:

### Data Files
- `data/df_odor_strength.csv` - Main curated dataset
- `data/odor_strength_groups.csv` - Group assignments for cross-validation
- `data/goodscents/goodscents.csv` - GoodScents scraped data
- `data/pubchem/` - PubChem data files

### Hyperparameter databases
- `hyperparameter_optimization/hp_opt_dbs/` - Optuna trials databases for each combination of encoder-predictor

### Analysis Files
- `figures/` - Generated plots and visualizations
- `test_predictions/` - Prediction results on test sets

## Key Features

- **Multiple Molecular Encoders**: RDKit descriptors, Morgan fingerprint, RDKit fingerprint, topological torsion fingerprint, atom pair fingerprint, ChemBERTa embeddings
- **Various Predictors**: Random Forest, XGBoost, multi-layer perceptrons, CORAL, message passing neural networks
- **Cross-validation**: Group-based splitting to avoid data leakage
- **External Validation**: Testing on independent Keller 2016 dataset
- **Interpretability**: SHAP analysis for understanding predictions
- **Direct vs Indirect Prediction**: Comparison of different prediction strategies

## File Structure

```
odor_strength_module/
├── README.md                                    # This file
├── dataset_curation.py                         # Dataset preparation (run first)
├── dataset_curation.ipynb                      # Jupyter version of dataset curation
├── dataset_analysis.py                         # Exploratory data analysis
├── dataset_analysis.ipynb                      # Jupyter version of analysis
├── model_training_validation_insights.py       # Model training & validation
├── model_training_validation_insights.ipynb    # Jupyter version of training
├── best-performing_model_application.ipynb     # Model application (independent)
├── data/                                        # Raw data and generated datasets
├── models/                                      # Molecular encoder and predictor implementation and pre-trained best-performing model
├── utility/                                     # Utility functions, such as metrics, data splitting, plot colors, 
└── figures/                                     # Generated plots
```

## Usage Notes

1. **Run dataset curation before analysis and training** - Other scripts depend on the generated dataset files
2. Dataset analysis and model training can be run in any order after curation
3. Model application is independent and can be run standalone

## Citation

If you use this code in your research, please cite the corresponding publication.