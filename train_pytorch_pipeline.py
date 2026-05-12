#!/usr/bin/env python3
"""
train_pytorch_pipeline.py

Complete, commented PyTorch pipeline to train a regression model that predicts gene expression
(TPM) from simplified genomic features when raw sequence is not available.

Usage (quick test):
  python3 train_pytorch_pipeline.py --dry-run --max-genes 10 --epochs 2

This script supports a full run as well. It expects the GTEx TPM parquet and metadata to be present
in the working directory (same names used in earlier code). For production runs point to your
predictions / inputs as needed and increase gene/sample counts.

Sections:
 - preprocessing
 - dataset creation
 - model definition
 - training loop with early stopping
 - evaluation and visualizations

Notes:
 - By default this trains on a small subset (controlled by --max-genes) so you can test quickly.
 - For a real training run increase `--max-genes` (or pass a gene list) and `--epochs`.
 - The dataset is formed from (sample, gene) pairs for selected genes. We split donors (derived
   from GTEX sample IDs) into train/val/test to avoid leakage.
"""

import argparse
import os
import random
from pathlib import Path
import time
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


########## Utilities and config ##########

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


########## Preprocessing: load TPM and metadata ##########

def load_tpm_and_meta(parquet_path, meta_path):
    """Load TPM parquet and metadata TSV, return (actual: DataFrame samples x genes, meta DataFrame).

    The TPM parquet used in this workspace has gene rows and sample columns; this helper will
    transpose to samples x genes if necessary and drop a 'Description' column if present.
    """
    import pyarrow.parquet as pq
    tbl = pq.read_table(parquet_path)
    df = tbl.to_pandas()
    if 'Description' in df.columns:
        df = df.drop(columns=['Description'])

    # detect orientation: if columns look like GTEX sample IDs then genes x samples -> transpose
    cols = [str(c) for c in df.columns]
    idx = [str(i) for i in df.index]
    sample_like_in_cols = any(c.startswith('GTEX-') for c in cols)
    sample_like_in_index = any(i.startswith('GTEX-') for i in idx)
    if sample_like_in_cols and not sample_like_in_index:
        actual = df.T
    elif sample_like_in_index and not sample_like_in_cols:
        actual = df
    else:
        # fallback: assume index are ENSG gene IDs -> transpose
        if len(idx) > 0 and all(str(i).upper().startswith('ENSG') for i in idx[:min(10, len(idx))]):
            actual = df.T
        else:
            actual = df

    meta = pd.read_csv(meta_path, sep='\t', low_memory=False)
    return actual, meta


########## Feature engineering ##########

def build_sample_features(meta, samples):
    """Construct simple sample-level features used as inputs.

    Returned features DataFrame indexed by sample ID. Features include:
      - one-hot tissue labels (SMTSD top categories)
      - PCA of per-sample TPM profile can be added elsewhere; here we provide placeholders.
    """
    # subset metadata to samples
    meta_sub = meta[meta['SAMPID'].isin(samples)].set_index('SAMPID')

    # Use SMTSD (tissue description). Create one-hot for the top K most frequent categories and
    # group the rest into 'Other'.
    if 'SMTSD' in meta_sub.columns:
        counts = meta_sub['SMTSD'].value_counts()
        topk = counts.index[:10].tolist()
        def map_cat(x):
            return x if x in topk else 'Other'
        meta_sub['SMTSD_top'] = meta_sub['SMTSD'].fillna('Unknown').map(map_cat)
        onehot = pd.get_dummies(meta_sub['SMTSD_top'], prefix='SMTSD')
    else:
        onehot = pd.DataFrame(index=meta_sub.index)

    # Minimal numeric features: sample-level RIN or center if present, otherwise zeros
    numeric_cols = []
    for c in ['SMRIN', 'SMCENTER']:
        if c in meta_sub.columns:
            numeric_cols.append(c)

    if numeric_cols:
        # some metadata columns contain non-numeric annotations (e.g. 'B1, A1');
        # coerce to numeric safely and fill NaNs with 0.0
        nums = meta_sub[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
    else:
        nums = pd.DataFrame(0.0, index=meta_sub.index, columns=['num_dummy'])

    features = pd.concat([onehot, nums], axis=1)
    # ensure ordering matches samples
    features = features.reindex(samples).fillna(0.0)
    return features


def build_gene_features(tpm_df, gene_list, n_pca=10):
    """Construct gene-level features used as inputs.

    Features include per-gene mean, std across samples and a PCA embedding computed across
    the per-gene expression vectors.
    Returns DataFrame indexed by gene.
    """
    # compute basic stats
    gene_means = tpm_df[gene_list].mean(axis=0)
    gene_stds = tpm_df[gene_list].std(axis=0).fillna(0.0)

    # PCA on gene x samples (we want gene embeddings)
    # Perform PCA on transposed (genes x samples) for the selected genes
    gene_matrix = tpm_df[gene_list].T.values  # shape genes x samples
    # log transform to stabilise variance for PCA
    gene_matrix = np.log1p(gene_matrix)
    pca = PCA(n_components=min(n_pca, gene_matrix.shape[0], gene_matrix.shape[1]))
    gene_pca = pca.fit_transform(gene_matrix)

    cols = [f'pca_{i}' for i in range(gene_pca.shape[1])]
    df = pd.DataFrame(gene_pca, index=gene_list, columns=cols)
    df['mean_tpm'] = gene_means.values
    df['std_tpm'] = gene_stds.values
    return df


########## Dataset & DataLoader ##########

class SampleGeneDataset(Dataset):
    """Dataset of (sample, gene) pairs for regression.

    Each example is a concatenation of sample-level features and gene-level features with
    a scalar target y = log1p(TPM) for that sample x gene.
    """
    def __init__(self, samples, genes, sample_feats, gene_feats, tpm_df):
        # samples: list of sample IDs (rows in tpm_df)
        # genes: list of gene IDs
        self.samples = list(samples)
        self.genes = list(genes)
        self.sample_feats = sample_feats
        self.gene_feats = gene_feats
        self.tpm = tpm_df
        # precompute index mapping for speed
        self.N = len(self.samples) * len(self.genes)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # map linear idx -> (i_sample, i_gene)
        n_genes = len(self.genes)
        i_sample = idx // n_genes
        i_gene = idx % n_genes
        samp = self.samples[i_sample]
        gene = self.genes[i_gene]

        s_feat = self.sample_feats.loc[samp].values.astype(np.float32)
        g_feat = self.gene_feats.loc[gene].values.astype(np.float32)
        x = np.concatenate([s_feat, g_feat]).astype(np.float32)

        # target: log1p TPM
        y = np.log1p(self.tpm.loc[samp, gene]).astype(np.float32)
        return x, y, samp, gene


########## Model definition ##########

class RegressionNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256,128), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


########## Training and evaluation utilities ##########

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    count = 0
    for X, y, _, _ in loader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        count += X.size(0)
    return total_loss / max(1, count)


def evaluate(model, loader, device):
    model.eval()
    ys = []
    ps = []
    meta = []
    with torch.no_grad():
        for X, y, samp, gene in loader:
            X = X.to(device)
            pred = model(X).cpu().numpy()
            ys.append(y.numpy())
            ps.append(pred)
            meta.extend(zip(samp, gene))
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return y, p, meta


def compute_metrics(y_true, y_pred):
    # y are log1p(TPM) values; compute MSE directly, r and R2 on original scale or log-scale
    mse = mean_squared_error(y_true, y_pred)
    # Pearson on log-scale
    try:
        r, pval = pearsonr(y_true, y_pred)
    except Exception:
        r, pval = np.nan, np.nan
    r2 = r2_score(y_true, y_pred)
    return {'mse': float(mse), 'pearson_r': float(r), 'pearson_p': float(pval), 'r2': float(r2)}


########## Main orchestration ##########

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', default='GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet')
    parser.add_argument('--meta', default='GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt')
    parser.add_argument('--max-genes', type=int, default=100, help='Subset number of genes to train on (quick test)')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--out-dir', default='pytorch_run')
    args = parser.parse_args()

    set_seed(42)

    outdir = Path(args.out_dir)
    outdir.mkdir(exist_ok=True)

    print('Loading data... this may take a few seconds')
    t0 = time.time()
    tpm, meta = load_tpm_and_meta(args.parquet, args.meta)
    print('Loaded TPM shape (samples x genes):', tpm.shape)
    print('Loaded metadata shape:', meta.shape)

    # Select brain samples using SMTSD
    if 'SMTSD' in meta.columns:
        brain_ids = meta[meta['SMTSD'].str.contains('Brain', na=False)]['SAMPID'].astype(str).unique().tolist()
    else:
        brain_ids = list(tpm.index)

    samples = [s for s in brain_ids if s in tpm.index]
    print('Brain samples (intersected):', len(samples))

    # gene selection: choose most variable genes if many
    gene_vars = tpm.loc[samples].var(axis=0).sort_values(ascending=False)
    gene_list = list(gene_vars.index[:min(args.max_genes, len(gene_vars))])
    print('Selected genes:', len(gene_list))

    # donor-level splitting: derive donor id from sample id (GTEX-XXXX part)
    # donor key = first two dash-separated tokens, e.g. GTEX-1117F
    def donor_of(s):
        parts = s.split('-')
        if len(parts) >= 2:
            return '-'.join(parts[:2])
        return s

    donors = sorted(set(donor_of(s) for s in samples))
    random.shuffle(donors)
    n = len(donors)
    ntrain = int(0.7 * n)
    nval = int(0.15 * n)
    train_donors = set(donors[:ntrain])
    val_donors = set(donors[ntrain:ntrain+nval])
    test_donors = set(donors[ntrain+nval:])

    sample_to_donor = {s: donor_of(s) for s in samples}
    train_samples = [s for s in samples if sample_to_donor[s] in train_donors]
    val_samples = [s for s in samples if sample_to_donor[s] in val_donors]
    test_samples = [s for s in samples if sample_to_donor[s] in test_donors]

    print('Donors total/train/val/test:', n, len(train_donors), len(val_donors), len(test_donors))
    print('Samples train/val/test:', len(train_samples), len(val_samples), len(test_samples))

    # Build features
    sample_feats = build_sample_features(meta, samples)
    gene_feats = build_gene_features(tpm, gene_list, n_pca=10)

    # Optionally reduce feature scale
    scaler = StandardScaler()
    # Fit scaler on concatenated training sample+gene features to standardize inputs
    # We'll transform sample_feats and gene_feats separately later by fitting on training subset
    # For simplicity, scale gene features globally and sample features globally
    gene_feats_scaled = pd.DataFrame(scaler.fit_transform(gene_feats), index=gene_feats.index, columns=gene_feats.columns)
    sample_feats_scaled = pd.DataFrame(scaler.fit_transform(sample_feats), index=sample_feats.index, columns=sample_feats.columns)

    # Construct datasets: use only train/val/test sample splits but same gene_list
    train_ds = SampleGeneDataset(train_samples, gene_list, sample_feats_scaled, gene_feats_scaled, tpm)
    val_ds = SampleGeneDataset(val_samples, gene_list, sample_feats_scaled, gene_feats_scaled, tpm)
    test_ds = SampleGeneDataset(test_samples, gene_list, sample_feats_scaled, gene_feats_scaled, tpm)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    input_dim = sample_feats_scaled.shape[1] + gene_feats_scaled.shape[1]
    model = RegressionNet(input_dim=input_dim, hidden_dims=(256,128), dropout=0.3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Training loop with early stopping
    best_val = float('inf')
    best_state = None
    patience = args.patience
    wait = 0
    history = {'train_loss': [], 'val_loss': []}

    if args.dry_run:
        print('Dry run mode: skipping training; doing a single eval pass on validation')
        y_val, p_val, meta_val = evaluate(model, val_loader, device)
        metrics = compute_metrics(y_val, p_val)
        print('Validation metrics (dry-run):', metrics)
        return

    for epoch in range(1, args.epochs+1):
        t_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        y_val, p_val, _ = evaluate(model, val_loader, device)
        val_loss = mean_squared_error(y_val, p_val)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f'Epoch {epoch} train_loss={train_loss:.6f} val_mse={val_loss:.6f} time={(time.time()-t_start):.1f}s')

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = model.state_dict()
            wait = 0
            torch.save({'model_state': best_state, 'args': vars(args)}, outdir / 'best_model.pth')
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping')
                break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on test set
    y_test, p_test, meta_test = evaluate(model, test_loader, device)
    metrics = compute_metrics(y_test, p_test)
    print('Test metrics:', metrics)

    # Save metrics and predictions
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(outdir / 'metrics_summary.csv', index=False)

    # Save predictions with metadata (sample,gene)
    preds_df = pd.DataFrame({'sample': [m[0] for m in meta_test], 'gene': [m[1] for m in meta_test], 'y_true': y_test, 'y_pred': p_test})
    preds_df.to_csv(outdir / 'predictions.csv', index=False)

    # Save model
    torch.save(model.state_dict(), outdir / 'model_weights.pt')

    # Visualizations
    # 1) predicted vs observed scatter (subsample for plotting)
    nplot = min(2000, len(y_test))
    idxs = np.random.choice(len(y_test), nplot, replace=False)
    plt.figure(figsize=(6,6))
    plt.scatter(y_test[idxs], p_test[idxs], s=6, alpha=0.6)
    plt.xlabel('log1p(Actual TPM)')
    plt.ylabel('log1p(Predicted)')
    plt.title('Predicted vs Actual (test subset)')
    plt.tight_layout()
    plt.savefig(outdir / 'pred_vs_actual_scatter.png', dpi=150)
    plt.close()

    # 2) training loss curve
    plt.figure()
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'training_loss.png', dpi=150)
    plt.close()

    # 3) histogram of residuals
    residuals = (p_test - y_test)
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=100, color='#2b7bba', alpha=0.8)
    plt.xlabel('Residual (pred - actual) [log1p TPM]')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(outdir / 'residuals_hist.png', dpi=150)
    plt.close()

    print('Saved outputs to', outdir)


if __name__ == '__main__':
    main()
