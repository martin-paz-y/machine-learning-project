# Fake Job Postings Detection (NLP + ML)

Detect fraudulent job postings using a mix of **text features** (TF-IDF) and a few **structured signals** (e.g., telecommuting, has_company_logo, has_questions). This repository contains a single end-to-end notebook that loads data, cleans text, trains multiple classifiers, and compares results with an emphasis on fraud recall.

## What’s inside

* Text preprocessing:

  * Lowercasing + regex cleanup (remove punctuation / extra whitespace)
  * Build a single text field: title + description + requirements
* Feature engineering:

  * text → TF-IDF
  * Numeric/binary fields passed through:

    * telecommuting, has_company_logo, has_questions
* Models:

  * Logistic Regression (TF-IDF + structured features)
  * Decision Tree
  * Random Forest (baseline)
  * Random Forest with GridSearchCV hyperparameter tuning

## Dataset

The notebook downloads training data from:

job_train.csv (raw GitHub file):
[https://raw.githubusercontent.com/martin-paz-y/machine-learning-project/refs/heads/main/job_train.csv](https://raw.githubusercontent.com/martin-paz-y/machine-learning-project/refs/heads/main/job_train.csv)

The label is fraudulent (0 = legit, 1 = fraudulent).

## Results snapshot (from the notebook)

Your metrics may vary slightly due to random splits, but the notebook reports (test set):

Model: Logistic Regression (threshold tuned)

* Fraud Precision: 0.47
* Fraud Recall: 0.80
* Accuracy: 0.94

Model: Logistic Regression (more aggressive threshold)

* Fraud Precision: 0.30
* Fraud Recall: 0.91
* Accuracy: 0.89

Model: Decision Tree

* Fraud Precision: 0.25
* Fraud Recall: 0.77
* Accuracy: 0.87

Model: Random Forest (baseline)

* Fraud Precision: 1.00
* Fraud Recall: 0.47
* Accuracy: 0.97

Model: Random Forest (tuned)

* Fraud Precision: 0.89
* Fraud Recall: 0.54
* Accuracy: 0.97

Note: In fraud detection, recall for the fraud class can matter more than overall accuracy, depending on your business cost of missed fraud vs. false alarms.

## Repository structure

.
├── public_machine´procject-fake-jobs.ipynb   # main notebook (training + evaluation)
└── README.md

## How to run

1. Create an environment (recommended)

python -m venv .venv

macOS / Linux:
source .venv/bin/activate

Windows:
.venv\Scripts\activate

2. Install dependencies

pip install -U pip
pip install pandas numpy scikit-learn matplotlib seaborn

3. Open the notebook

jupyter notebook
or
jupyter lab

Then run: public_machine´procject-fake-jobs.ipynb

## Notes / next ideas

* Add cross-validation for more stable estimates
* Try linear SVM, calibrated probabilities, or class weights
* Perform threshold optimization based on cost (false negatives vs false positives)
* Add explainability (feature importance / SHAP) and error analysis

## License

Add a license file if you plan to distribute this project publicly (e.g., MIT, Apache-2.0).

## Acknowledgements

Dataset hosted via GitHub raw file (see link above).
