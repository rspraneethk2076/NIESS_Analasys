# NEISS World Cup Analysis

## Purpose

This project evaluates whether actual men's FIFA World Cup tournament periods
were associated with changes in soccer-related injuries treated in United States
emergency departments and captured by the National Electronic Injury
Surveillance System (NEISS).

The principal research question is:

> Are actual men's FIFA World Cup tournament periods associated with changes in
> NEISS-estimated soccer-related emergency department injuries in the United
> States?

The analysis is observational and ecological. Results should be described as
calendar-period associations and should not be interpreted as causal effects or
as evidence that individual patients watched, attended, or participated in a
World Cup event.

## Main Notebook

`NEISS_world_cup_analysis_v4_code_only_executed.ipynb`

This is the compact, code-only version of the final executed notebook. It
contains seven partitioned code cells, their saved outputs, and no Markdown
cells or inline Python comments. This README supplies the scientific and
reproducibility context intentionally removed from that notebook.

The annotated source notebook is preserved separately:

`NEISS_world_cup_analysis_v4_final_executed.ipynb`

Do not overwrite the preserved source notebook when rerunning the analysis.

## Data Source

NEISS is maintained by the United States Consumer Product Safety Commission
(CPSC). It is a stratified probability sample of participating emergency
departments. Statistical weights permit calculation of national estimates of
emergency department-treated consumer product and recreation-related injuries.

The verified analysis uses 27 annual NEISS Excel workbooks:

- First year: 1999
- Last year: 2025
- Expected file names: `neiss1999.xlsx` through `neiss2025.xlsx`
- Verified record count after harmonization: 9,794,932

A 2026 workbook was not available and was not included in the verified
analysis.

### Download And Placement

1. Obtain each annual NEISS public-use data workbook from the CPSC NEISS data
   download service.
2. Keep one workbook per year.
3. Name the files `neissYYYY.xlsx`, where `YYYY` is the four-digit year.
4. Store all 27 files in one directory.
5. Set the `NEISS_DATA_DIR` environment variable to that directory before
   running the notebook.

Raw NEISS workbooks should not be committed to a public repository unless their
redistribution terms have been reviewed.

## Study Definitions

### Primary Outcome

The primary outcome is a soccer-related NEISS record identified by soccer
product code `1267`.

Narrative-keyword definitions and other restrictions are sensitivity analyses,
not replacements for the product-code primary outcome.

### Primary Exposure

The primary exposure consists of the actual dates of six men's FIFA World Cup
tournaments:

- 2002
- 2006
- 2010
- 2014
- 2018
- 2022

Together, these tournament windows include 186 exposed calendar days.

### Primary Comparison

Tournament dates are compared with matched dates in adjacent non-tournament
years. The verified comparison set contains 308 control days.

Secondary and sensitivity analyses include same-day-of-week matching,
tournament-specific contrasts, leave-one-tournament-out estimates, pre-2022
analysis, alternative outcome definitions, and weekly adjusted models.

## Software Environment

The verified environment was:

- Windows, 64-bit
- Python 3.13.2
- `matplotlib==3.10.9`
- `numpy==2.4.0`
- `openpyxl==3.1.5`
- `pandas==2.3.3`
- `pyarrow==24.0.0`
- `scipy==1.16.3`

The existing environment specification is:

`github_release/NIESS_Analasys/requirements.txt`

Install the core packages from the project root with:

```powershell
python -m pip install -r .\github_release\NIESS_Analasys\requirements.txt
```

JupyterLab or another Jupyter-compatible application is needed only for
interactive notebook use. The included lightweight executor can run the
notebook without Jupyter.

## Configuration

Run the analysis from the project root:

```text
D:\ai-job-agent\MIMIC_Project_3
```

The notebook uses the current working directory as the project directory.
Starting it elsewhere will redirect relative output paths to the wrong
location.

Set the raw-data directory before execution:

```powershell
$env:NEISS_DATA_DIR = "C:\Users\rendu\Downloads\NEISS Data\NEISS Data"
```

If `NEISS_DATA_DIR` is not set, the notebook checks this default location:

```text
%USERPROFILE%\Downloads\NEISS Data\NEISS Data
```

## Execution

### Option 1: Jupyter

1. Start Jupyter from the project root.
2. Open `NEISS_world_cup_analysis_v4_code_only_executed.ipynb`.
3. Confirm that `NEISS_DATA_DIR` points to the annual workbooks.
4. Select **Restart Kernel and Run All Cells**.
5. Save the rerun under a new file name.

### Option 2: Included Lightweight Executor

Run the notebook into a new output file:

```powershell
Set-Location "D:\ai-job-agent\MIMIC_Project_3"
$env:NEISS_DATA_DIR = "C:\Users\rendu\Downloads\NEISS Data\NEISS Data"
python .\execute_notebook_light.py `
  .\NEISS_world_cup_analysis_v4_code_only_executed.ipynb `
  .\NEISS_world_cup_analysis_v4_code_only_rerun.ipynb
```

The executor stops with a nonzero exit status if a code cell raises an error.
Review the output notebook and generated verification files after every full
run.

## Notebook Execution Order

The seven code cells must be run sequentially:

1. Import packages, resolve configuration, and validate annual workbooks and
   cached files.
2. Run the base analysis pipeline and load its run summary.
3. Run professor-requested analyses, including direct paired-PSU variance,
   tournament-specific estimates, leave-one-out analyses, day-of-week
   sensitivity, HAC-lag sensitivity, and the tournament forest plot.
4. Compare annual results with published CPSC calibration values for
   2021-2025.
5. Validate generated outputs and create the verified master-results dictionary,
   verification report, and SHA-256 manifest.
6. Display primary, sensitivity, tournament-specific, and HAC-lag results.
7. Display the quasi-Poisson robustness result used in the manuscript.

Do not execute cells out of order because later cells use objects and files
created by earlier cells.

## Expected Outputs

### Base Analysis

The base pipeline is:

`analysis_v2/neiss_world_cup_analysis_v2.py`

Its principal output directory is:

`outputs/neiss_world_cup_v2/`

Important manuscript-ready files include:

- `tables/table_A_cohort_data_availability.csv`
- `tables/table_B_primary_soccer_injury_burden.csv`
- `tables/table_C_tournament_period_comparison.csv`
- `tables/table_D_diagnosis_bodypart_location.csv`
- `tables/table_E_primary_model_results.csv`
- `tables/table_E_model_diagnostics.csv`
- `tables/table_F_sensitivity_analyses.csv`
- `tables/table_G_exploratory_narrative_audit.csv`
- `tables/manuscript_ready_tables_v2.xlsx`

### Professor-Revision Analyses

The revision output directory is:

`outputs/professor_revision_20260620/`

It contains:

- `tables/table_PR1_primary_and_key_sensitivity_contrasts.csv`
- `tables/table_PR2_tournament_specific_contrasts.csv`
- `tables/table_PR3_leave_one_tournament_out.csv`
- `tables/table_PR4_day_of_week_balance.csv`
- `tables/table_PR5_newey_west_lag_sensitivity.csv`
- `tables/table_PR6_cpsc_calibration_benchmark.csv`
- `figures/figure3_tournament_forest_plot.png`
- `figures/figure3_tournament_forest_plot.svg`
- `figures/figure3_tournament_forest_plot.pdf`
- `figures/figure3_tournament_forest_plot.tiff`
- `master_results_dictionary_verified_20260620.csv`
- `verification_report.json`
- `sha256_manifest.json`

The verified master-results dictionary is the authoritative numerical source
for manuscript revision. Manuscript numbers should not be copied from console
text or reconstructed manually when a verified table value is available.

## Reproduction Checks

A successful run should satisfy all of the following:

- All 27 annual workbooks from 1999-2025 are detected.
- The harmonized database contains 9,794,932 records.
- The primary soccer product-code cohort contains 170,679 unweighted records.
- The weighted primary soccer estimate is approximately 5,366,681.
- No notebook cell ends with an error output.
- `verification_report.json` reports successful output validation.
- `sha256_manifest.json` is regenerated for the current output package.
- The master-results dictionary is generated from verified tables.

Selected headline estimates from the verified run are supplied as
reproducibility checks, not as values to hard-code:

- Primary paired-PSU ratio: 1.17948
- Primary 95% CI: 1.00204 to 1.38835
- Same-day-of-week sensitivity ratio: 1.16592
- Pre-2022 sensitivity ratio: 1.22115
- Weekly adjusted HAC lag-8 ratio: 1.18205
- Quasi-Poisson/HAC ratio: 1.14630

Small formatting differences are acceptable, but unexplained numerical
differences should be investigated before manuscript values are updated.

## Statistical Notes

The weekly analysis evaluates seasonality, calendar time, COVID-era disruption,
overdispersion, and serial correlation. HAC lag 8 permits correlation across
approximately two months of weekly observations; lags 4 through 12 are examined
as sensitivity analyses.

The quasi-Poisson/HAC result is a robustness analysis because the weekly count
model exhibits overdispersion. Tournament-specific and leave-one-tournament-out
results should be interpreted with attention to limited tournament count,
heterogeneity, and confidence-interval width.

LDA, TF-IDF, SVD, t-SNE, and narrative clustering are exploratory or
supplementary analyses. They should not be presented as primary evidence for
the World Cup association.

## Reporting Language

Use this denominator description where applicable:

> NEISS-estimated emergency department-treated consumer product and
> recreation-related injuries.

Prefer cautious terms such as:

- association
- temporal association
- ecological calendar-period association
- hypothesis-generating

Avoid causal wording and avoid the undefined phrase "all injuries."

Every reported table should identify the numerator, denominator, unweighted
sample size, weighted national estimate, weighted percentage, comparison group,
and 95% confidence interval when available.

## Reproducibility And Preservation

- Preserve the original executed and annotated notebooks.
- Write reruns to new notebook and output names.
- Do not manually edit generated CSV values.
- Use the verified master-results dictionary for manuscript numbers.
- Retain the verification report and checksum manifest with shared outputs.
- Record any change to exposure dates, product codes, comparison windows, or
  model specification before rerunning the analysis.

The code-only notebook is intended for compact technical delivery. The
annotated notebook, this README, the analysis plan, and the verified output
tables together provide the complete reproducibility record.
