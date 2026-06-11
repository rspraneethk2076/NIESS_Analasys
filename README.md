# NEISS World Cup Analysis: 

## Scientific question

Are actual men's FIFA World Cup tournament periods associated with changes in
NEISS-estimated soccer-related emergency department injuries in the United
States?

The primary exposure is the actual men's FIFA World Cup tournament window. The
primary outcome is a soccer-related injury identified by NEISS product code
1267. The primary comparison uses matched same-calendar control dates. These
are ecological calendar-period comparisons and do not establish individual
World Cup exposure or causation.

## Data source

The analysis uses annual public-use National Electronic Injury Surveillance
System (NEISS) workbooks from the U.S. Consumer Product Safety Commission
(CPSC). The analytic period is 1999-2025, comprising 27 annual workbooks.

Use the approved denominator wording when reporting results:

> NEISS-estimated emergency department-treated consumer product and
> recreation-related injuries.

Official data pages:

- NEISS overview and access: <https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data>
- NEISS frequently asked questions: <https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data/Neiss-Frequently-Asked-Questions>

## Download and organize the data

1. Open the official CPSC NEISS data page and select **Access NEISS**.
2. Select a user affiliation.
3. Choose **Find Data** to download all records for each available year.
4. Download one annual workbook for every year from 1999 through 2025.
5. Name the workbooks `neiss1999.xlsx`, `neiss2000.xlsx`, and so on through
   `neiss2025.xlsx`.
6. Place them in `data/neiss/`, or set `NEISS_DATA_DIR` to another directory.

The analysis expects `neiss2025.xlsx` because its `NEISS_FMT` worksheet supplies
the current label maps. Keep the annual workbooks unchanged after download.
Raw NEISS files are intentionally not included in this reproducibility package.

## Software environment

The verified analysis run used 64-bit Python 3.13.2 on Windows. Create an
isolated environment and install the recorded dependencies:

```powershell
cd NEISS_world_cup_reproducible
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

JupyterLab is optional but convenient for running the notebook:

```powershell
python -m pip install "jupyterlab>=4,<5" "ipykernel>=6,<7"
python -m jupyter lab
```

## Configuration

All defaults are relative to this package. No user-specific local paths are
required.

| Environment variable | Default | Purpose |
|---|---|---|
| `NEISS_PROJECT_DIR` | package root | Project root used by scripts and notebook |
| `NEISS_DATA_DIR` | `data/neiss` | Annual NEISS workbook directory |
| `NEISS_OUTPUT_DIR` | `outputs/neiss_world_cup_v2` | Primary tables, figures, caches, and audits |
| `NEISS_MASTER_RESULTS_DIR` | `outputs/master_results` | Verified master results dictionary |
| `NEISS_MANUAL_REVIEW_DIR` | `outputs/manual_review` | Manual-review tables |
| `NEISS_REVISION_DIR` | `revision_plan` | Review memos and readiness checklists |
| `NEISS_FINAL_TABLE_DIR` | `tables` | Final model table |
| `NEISS_PUBLICATION_FIGURE_DIR` | `outputs/publication_quality_figures` | Vector and high-resolution figures |

Example for an external data directory:

```powershell
$env:NEISS_DATA_DIR = "D:\path\to\annual-neiss-workbooks"
```

The example above is a user-supplied configuration value, not a path embedded
in the code.

## Execution order

### Preferred notebook workflow

1. Start Jupyter from the package root.
2. Open `NEISS_world_cup_analysis_v2_portable.ipynb`.
3. Review and run the configuration cell.
4. Run all remaining cells in order.
5. Confirm that the execution summary reports `status: completed`.
6. Review the manual-verification items before using results in a manuscript.

### Equivalent command-line workflow

Run these commands from the package root:

```powershell
python analysis_v2/neiss_world_cup_analysis_v2.py
python analysis_v2/manual_review_resolution_pass.py
python analysis_v2/generate_publication_figures.py
```

The first command creates the primary analysis outputs. The second creates the
manual-review and model-resolution package. The third creates publication
figure formats and should be run after the primary tables exist.

`analysis_v2/build_executed_notebook_v2.py` can regenerate an executed portable
notebook after a complete local run. It intentionally writes a new notebook
named `NEISS_world_cup_analysis_v2_portable_executed.ipynb`.

## Expected outputs

Primary output directory: `outputs/neiss_world_cup_v2/`

- `run_summary.json`: execution status, analytic counts, runtime, and unresolved
  manual-verification items.
- `output_manifest.csv`: generated-file inventory.
- `tables/manuscript_ready_tables_v2.xlsx`: manuscript-ready workbook.
- `tables/table_A_cohort_data_availability.csv`: cohort and annual availability.
- `tables/table_B_primary_soccer_injury_burden.csv`: annual soccer injury burden.
- `tables/table_C_tournament_period_comparison.csv`: primary matched-window comparison.
- `tables/table_D_diagnosis_bodypart_location.csv`: injury characteristics.
- `tables/table_E_primary_model_results.csv`: adjusted model estimates.
- `tables/table_E_model_diagnostics.csv`: model diagnostics.
- `tables/table_F_sensitivity_analyses.csv`: sensitivity analyses.
- `tables/table_G_exploratory_narrative_audit.csv`: exploratory narrative audit.
- `figures/`: primary PNG and SVG figures.
- `audit/`: cleaning, calendar, weekly aggregation, model-fit, and plausibility audits.
- `cache/yearly/`: reusable annual Parquet caches.

Additional outputs:

- `outputs/master_results/master_results_dictionary.csv`: single value dictionary.
- `outputs/manual_review/`: adjudication and verification tables.
- `outputs/publication_quality_figures/`: SVG, PDF, 600-dpi PNG, and TIFF figures.
- `revision_plan/`: manual-review memos and manuscript-readiness checks.
- `tables/final_model_table_v3.csv`: resolved final model table.



