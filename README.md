# A Comparative Study of Rule-Based and Supervised Text Extraction Techniques for Detecting Cardiovascular Disease Signals in NEISS Narratives

This repository contains a notebook-based analysis pipeline for extracting a **proxy cardiovascular disease (CVD)-related lexical signal** from NEISS injury narratives and using that signal for descriptive, classification, and downstream admission modeling tasks.

The workflow combines:
- rule-based text extraction with local negation handling,
- TF-IDF + Logistic Regression,
- TF-IDF + calibrated Linear SVM,
- logistic regression for epidemiologic association,
- gradient boosting for downstream admission prediction,
- cross-validation, bootstrap confidence intervals, and threshold tuning.

## Project Goal

The notebook explores whether short free-text NEISS narratives contain enough lexical information to derive a **proxy CVD-related signal** and whether that signal is associated with hospital admission in the analytic cohort. The notebook is best interpreted as an **exploratory, weakly supervised workflow**, because the machine-learning models are trained to reproduce rule-derived labels rather than clinician-annotated gold-standard labels. Based on the accompanying manuscript draft, this framing is intentional and should be preserved in any GitHub description. fileciteturn5file0turn5file1

## Input Data

The notebook expects an Excel file named:

```bash
neiss2024.xlsx
```

The analysis uses the following columns from the NEISS dataset:
- `Diagnosis`
- `Other_Diagnosis`
- `Diagnosis_2`
- `Other_Diagnosis_2`
- `Narrative_1`
- `Disposition`
- `Age`
- `Sex`
- `Alcohol`
- `Drug`
- `Fire_Involvement`
- `Body_Part`
- `Diagnosis`
- `Location`

Narrative-bearing columns are concatenated into a lowercase `combined_text` field before rule-based and machine-learning processing begins. fileciteturn5file8turn5file11

## End-to-End Workflow

### 1. Rule-based proxy CVD extraction

The notebook first creates a rule-based extractor using regular-expression patterns for cardiovascular terms such as hypertension, CAD, MI, CHF, AFib, CVA, PAD, cardiomyopathy, valvular disease, and aortic stenosis. A local negation check looks up to 20 characters to the left of the matched term for words such as `no`, `denies`, `denied`, `without`, and `negative for`. If a term is present without a nearby negation trigger, the record is marked positive for the proxy rule-derived label. fileciteturn5file8turn5file11

A separate sensitivity-analysis section recalculates rule-derived labels using 10-, 20-, 30-, and 40-character negation windows so that the prevalence of the extracted proxy signal can be compared across negation-scope settings. This mirrors the manuscript’s sensitivity-analysis reporting. fileciteturn5file0

### 2. Admission outcome construction

Admission status is derived from the `Disposition` field:
- `1` → `Admission = 0`
- `2, 4, 5` → `Admission = 1`
- all other codes → excluded as unknown admission status

The notebook then restricts the analytic dataset to cases with known admission status, valid age, and cleaned binary sex coding. fileciteturn5file8turn5file11

### 3. Descriptive and epidemiologic analysis

The rule-derived proxy label is used to:
- summarize subgroup size,
- generate descriptive tables,
- compare age, sex, and admission prevalence,
- fit a logistic regression model relating the rule-derived signal to admission,
- estimate odds ratios and confidence intervals.

The notebook includes updated descriptive-table generation using median/IQR and bootstrap confidence intervals for proportions. fileciteturn5file12

### 4. TF-IDF + Logistic Regression

A text-classification dataset is created from non-empty `combined_text` rows. The notebook performs an 80/20 stratified train-test split and builds TF-IDF features with:
- lowercase conversion,
- English stop-word removal,
- unigram + bigram features,
- `max_features = 20000`,
- `min_df = 2`.

A logistic regression classifier is trained with class-balanced weighting and `max_iter = 1000`. The notebook then reports confusion matrix, classification metrics, ROC curve, precision-recall curve, and predictions for all rows (`CVD_ml_prob`, `CVD_ml`). fileciteturn5file8turn5file1

### 5. Five-fold stratified cross-validation

To improve robustness under class imbalance, the notebook adds **5-fold stratified cross-validation on the training set** for the text classifiers. Precision, recall, F1, and AUROC are averaged across folds for both:
- TF-IDF + Logistic Regression
- TF-IDF + calibrated Linear SVM

This supplements the held-out test split rather than replacing it. The same cross-validation strategy was later integrated into the paper’s Methods/Results sections. fileciteturn5file1

### 6. TF-IDF + calibrated Linear SVM

The notebook trains a calibrated linear SVM using:
- `LinearSVC(class_weight="balanced", random_state=42)`
- `CalibratedClassifierCV(..., method="sigmoid", cv=5)`

This section generates:
- confusion matrix,
- ROC comparison against logistic regression,
- precision-recall curve,
- predicted probabilities and binary predictions (`CVD_svm_prob`, `CVD_svm`). fileciteturn5file14turn5file1

### 7. Downstream admission modeling

The notebook includes two styles of downstream admission modeling:

#### Minimal feature admission model
A simpler logistic model is built using:
- `CVD_ml`
- `Age`
- `Sex_binary`

#### Expanded fair-feature comparison
A broader feature set is later introduced for a fairer comparison across models:
- `CVD_ml`
- `Age`
- `Sex_binary`
- `Alcohol`
- `Drug`
- `Fire_Involvement`
- `Body_Part`
- `Diagnosis`
- `Location`

Categorical features are one-hot encoded before fitting the gradient boosting classifier. The notebook also contains bootstrap-based confidence interval code for held-out test-set metrics and formatted comparison tables suitable for manuscript insertion. fileciteturn5file13turn5file1

### 8. Gradient Boosting + threshold tuning

For the downstream admission task, the notebook fits a gradient boosting classifier with:
- `n_estimators = 200`
- `learning_rate = 0.05`
- `max_depth = 3`
- `random_state = 42`

It evaluates confusion matrices, ROC curves, feature importance, fair-model comparisons, and threshold tuning based on F1. The related manuscript text reports that the tuned threshold improves recall and F1 relative to the default threshold under severe class imbalance. fileciteturn5file1turn5file13

### 9. Visualization outputs

The notebook produces multiple figures, including:
- admission rate bar plots,
- age distribution boxplots,
- confusion matrices,
- ROC curves,
- precision-recall curves,
- merged gradient-boosting feature importance plots,
- comparative performance bar charts,
- consolidated confusion-matrix panels.

Several of these figures were later aligned with manuscript revision requests, including 95% confidence intervals and consolidated confusion matrices. fileciteturn5file13turn5file14

## Key Hyperparameters

### TF-IDF vectorizer
- `lowercase=True`
- `stop_words="english"`
- `ngram_range=(1, 2)`
- `max_features=20000`
- `min_df=2`

### Logistic Regression (text classifier)
- `penalty='l2'` (default)
- `C=1.0` (default)
- `solver='lbfgs'` (default)
- `max_iter=1000`
- `class_weight='balanced'`
- `random_state=42`

### Calibrated Linear SVM
- base estimator: `LinearSVC`
- `C=1.0` (default)
- `class_weight='balanced'`
- calibration: `sigmoid`
- calibration CV folds: `5`
- `random_state=42`

### Gradient Boosting (admission model)
- `n_estimators=200`
- `learning_rate=0.05`
- `max_depth=3`
- `subsample=1.0` (default)
- `random_state=42`

These settings match the model summary later added to the manuscript. fileciteturn5file0turn5file1turn5file14

## Generated Outputs

The notebook writes several intermediate and final files, including examples such as:

```bash
step1_cvd_done.xlsx
step2_analysis_ready.xlsx
table1_descriptive.xlsx
table2_regression_results.xlsx
table_descriptive_rule_cvd_bootstrap.xlsx
table_cv_text_classifiers.xlsx
step_method3_svm_outputs.xlsx
step_method3_gb_dataset.xlsx
table_feature_importance_merged.xlsx
table_admission_model_comparison_bootstrap_numeric.xlsx
table_admission_model_comparison_bootstrap_formatted.xlsx
```

It also saves multiple PNG figures for confusion matrices, ROC curves, PR curves, and comparison charts. fileciteturn5file8turn5file13

## Environment and Dependencies

Core Python libraries used in the notebook include:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
scipy
```

The notebook was originally exported from Google Colab and assumes a notebook-style execution order. fileciteturn5file8

## How to Run

1. Place `neiss2024.xlsx` in the same working directory as the notebook.
2. Open the notebook in Jupyter or Google Colab.
3. Run cells sequentially from top to bottom.
4. Review the generated Excel tables and PNG figures in the working directory.
5. If you only want the final paper-ready outputs, make sure the later cells for:
   - cross-validation,
   - bootstrap confidence intervals,
   - fair admission-model comparison,
   - consolidated confusion matrices,
   - threshold tuning
   are also executed.

