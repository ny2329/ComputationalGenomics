# GTEx expression analysis — READMEGTEx expression analysis and predicted-vs-actual metrics



This repository contains scripts used to reproduce figures from a presentation and to evaluate predicted vs actual gene expression on GTEx brain samples.This small workspace contains scripts and data used to reproduce figures from a presentation and to compute predicted vs actual gene expression metrics for GTEx brain samples.



Top-level filesFiles of interest

----------------

- Download `GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet` — GTEx TPM (large), original data used for targets.- `GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt` - GTEx sample attributes (metadata, TSV). 

- Download `GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt` — GTEx sample attributes (metadata, TSV).- `GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet` - GTEx TPM expression (Parquet). Large file.

- `train_pytorch_pipeline.py` — end-to-end PyTorch training pipeline (feature engineering, donor split, training, evaluation).

- `compute_pred_vs_actual.py` — aligns predictions to actuals and computes per-gene R^2 and Pearson r, optionally creating top-gene plots.Scripts

- `generate_onehot.py`, `plot_onehot.py`, `run_pipeline.py` — utilities to create the one-hot matrix, plot it, and create heatmaps / sample distribution plots.

- `methods_and_results.md`, `methods_precise.md` — manuscript-ready Methods & Results text you can reuse in your writeup.- `run_pipeline.py` - primary pipeline for generating heatmaps and related figure modes.

- `requirements.txt` — minimal Python packages used in this workspace.- `generate_onehot.py` - build one-hot encoding CSV for sample tissue labels (SMTSD).

- `plot_onehot.py` - visualise the one-hot matrix (PNG/SVG) and match style from slides.

Quick start- `compute_pred_vs_actual.py` - aligns predictions and actual TPMs, computes per-gene R^2 and Pearson r, and optionally writes top-gene scatter plots.

-----------

1. Create and activate a virtual environment (recommended):Outputs generated in this workspace



```bash- `expression_heatmap*.png/.svg` - heatmap images.

python3 -m venv .venv- `expression_heatmap_onehot_viz.png/.svg` - one-hot visualization.

source .venv/bin/activate- `expression_heatmap_sample_distribution*.png/.svg` and `_counts.csv` - sample distribution plot and counts.

```- `synthetic_predictions_brain.csv` - synthetic predictions generated during development (actual TPM + small noise). NOTE: synthetic predictions are NOT real model outputs.

- `pred_actual_gene_metrics.csv` - per-gene metrics (r2, pearson_r, pearson_p, n_samples). Generated from `synthetic_predictions_brain.csv` by default in this session.

2. Install dependencies:- `pred_vs_actual_<ENSG>.png` - example scatter plots for top genes.



```bashQuick usage

pip install -r requirements.txt

```1) Install Python dependencies (recommended in a venv):



3. Run the smoke PyTorch training (already executed in this workspace). Example command used for validation:```bash

python3 -m venv .venv

```bashsource .venv/bin/activate

python3 train_pytorch_pipeline.py \pip install -r requirements.txt

  --parquet GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet \```

  --meta GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt \

  --max-genes 200 --epochs 3 --batch-size 1024 --out-dir pytorch_run_smokeIf you don't have a `requirements.txt`, install the core packages:

```

```bash

4. Compute predicted-vs-actual metrics from your own predictions CSV:pip install pandas numpy scipy matplotlib pyarrow

```

```bash

python3 compute_pred_vs_actual.py --pred /path/to/your_predictions.csv \2) Compute predicted vs actual metrics using your own predictions CSV:

  --parquet GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet \

  --meta GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt \- Predictions CSV format: rows are samples, first column is sample ID (GTEX-style SAMPID), remaining columns are genes using the same identifiers as the TPM (ENSG with version). Example header: `SampleID,ENSG00000123456.1,ENSG00000...`.

  --out pred_actual_gene_metrics.csv --plot-top 8

``````bash

python3 compute_pred_vs_actual.py --pred /path/to/your_predictions.csv --parquet GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet --meta GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt --out pred_actual_gene_metrics.csv --plot-top 8

Expected predictions CSV format```

-------------------------------

- First column: sample ID (GTEx-style `SAMPID`).This writes `pred_actual_gene_metrics.csv` and up to `--plot-top` scatter PNGs for the top genes by R^2.

- Remaining columns: genes using the same identifiers as the TPM (ENSEMBL IDs including version; e.g. `ENSG00000123456.1`).

- The script aligns predictions and actual TPMs on sample ID and gene ID before computing metrics.3) Single-gene plot



OutputsIf your presentation specifies a particular gene (ENSG or gene symbol), paste it into the conversation and the helper script will generate `pred_vs_actual_<GENE>.png` and the exact numeric r/R^2. Alternatively, you can run a small snippet in Python to plot a gene using `pred_actual_gene_metrics.csv`.

-------

- `pytorch_run_smoke/` (or other `--out-dir`): model weights, `metrics_summary.csv`, `predictions.csv`, and figures (`pred_vs_actual_scatter.png`, `training_loss.png`, `residuals_hist.png`).Notes and caveats

- `pytorch_run_smoke/plots/`: per-gene diagnostic plots (R^2 histogram, CDF, and example gene scatter plots) if you ask the script to create them.

- The workspace currently contains a synthetic predictions file created by adding small noise to the actual TPM values. Metrics computed with that file are artificially high and should not be used as an evaluation of a real predictive model.

Notes & caveats- Alignments: the scripts align on GTEX sample IDs. Ensure your predictions CSV uses the same GTEX sample IDs as the GTEx TPM Parquet (the sample column names in the Parquet are GTEX-style SAMPIDs).

-------------- If Parquet reading fails, you can pre-export a subset of the TPM to CSV (samples x genes) and pass it with `--parquet /path/to/actuals.csv`.

- The repository contains code and a synthetic predictions file used only for development; you must supply real model predictions for final evaluation.

- Per-gene performance is more informative than a single global scatter; see `methods_precise.md` for recommended reporting (per-gene R^2 distribution, representative gene plots, cross-validation).
- The smoke-run fit scalers globally for simplicity; for publication-quality runs fit scalers on training data only.
---
Updated: May 2026
