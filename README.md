# Census Income Classification and Customer Segmentation Project

This project contains Python scripts converted from the original Jupyter Notebook. The workflow is split into a shared preprocessing file, a classification script, and a segmentation script.

## Project files

```text
project/
├── README.md
├── preprocessing.py
├── classification_model.py
├── segmentation_model.py
├── census-bureau.data
└── census-bureau.columns
```

### File descriptions

- `preprocessing.py`  
  Contains shared functions for loading the data, cleaning missing values, adding feature engineering, encoding classification features, and creating output folders.

- `classification_model.py`  
  Runs the income classification pipeline. It includes train/dev/test splitting, Logistic Regression, Gradient Boosting, threshold tuning, feature selection using permutation importance, final training on train + dev, and final evaluation on the test set.

- `segmentation_model.py`  
  Runs the customer segmentation pipeline. It includes preprocessing, feature selection for segmentation, scaling/encoding, dimensionality reduction, K-Means evaluation, final clustering, and segment profiling.

## Required input files

Before running the scripts, make sure these two files are in the same folder as the Python files:

```text
census-bureau.data
census-bureau.columns
```

The scripts assume these default filenames. If your files are stored somewhere else, you can pass their paths using command-line arguments.

## Environment setup

Create and activate a virtual environment first. For macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Then install the required packages:

```bash
pip install numpy pandas scikit-learn matplotlib
```

The neural network section in `classification_model.py` is optional. If you want to run it, also install PyTorch:

```bash
pip install torch
```

## How to run the classification model

Run the default classification pipeline:

```bash
python classification_model.py
```

This uses:

```text
census-bureau.data
census-bureau.columns
```

and saves outputs to:

```text
outputs_classification/
```

You can also specify file paths and an output folder:

```bash
python classification_model.py \
  --data census-bureau.data \
  --columns census-bureau.columns \
  --output-dir outputs_classification
```

To skip the feature-selection model and run only the main classification models:

```bash
python classification_model.py --skip-feature-selection
```

To run the optional neural network model:

```bash
python classification_model.py --run-nn --nn-epochs 30
```

### Classification outputs

The classification script prints model performance to the terminal and saves result files such as:

```text
outputs_classification/logistic_threshold_results.csv
outputs_classification/gb_threshold_results.csv
outputs_classification/gb_permutation_importance.csv
outputs_classification/top_feature_gb_threshold_results.csv
outputs_classification/classification_model_summary.csv
```

The most important metrics to check are class-1 precision, class-1 recall, class-1 F1, PR-AUC, and ROC-AUC. Since the positive class is rare, class-1 F1 and recall are more meaningful than overall accuracy.

## How to run the segmentation model

Run the default segmentation pipeline:

```bash
python segmentation_model.py
```

This saves outputs to:

```text
outputs_segmentation/
```

You can also specify file paths, the output folder, the number of SVD components, and the final number of clusters:

```bash
python segmentation_model.py \
  --data census-bureau.data \
  --columns census-bureau.columns \
  --output-dir outputs_segmentation \
  --n-components 30 \
  --final-k 4
```

### Segmentation outputs

The segmentation script saves files such as:

```text
outputs_segmentation/kmeans_k_evaluation.csv
outputs_segmentation/kmeans_elbow_plot.png
outputs_segmentation/kmeans_silhouette_plot.png
outputs_segmentation/segmented_data.csv
outputs_segmentation/segment_numeric_profile.csv
outputs_segmentation/segment_top_category_profiles.csv
outputs_segmentation/segment_business_labels.csv
```

Use the elbow plot and silhouette score results to justify the number of clusters. The final segment profile files can be used to interpret each customer group from a business perspective.

## Recommended execution order

Run the scripts in this order:

```bash
python classification_model.py
python segmentation_model.py
```

The two scripts are independent, but both depend on `preprocessing.py` and the two raw data files.

## Notes

- Do not delete `preprocessing.py`; both main scripts import functions from it.
- Make sure all files are in the same project directory unless you provide custom file paths.
- The Gradient Boosting feature-selection step may take a few minutes because it uses permutation importance.
- If the script cannot find the data, check that the file names match exactly: `census-bureau.data` and `census-bureau.columns`.
