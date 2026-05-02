"""
classification_benchmark.py — Independent Cell-Type Classification
===================================================================
Runs scANVI, CellTypist, and a logistic regression baseline independently
of ORBIT to establish maximum achievable F1 on the same data split.

"""

from __future__ import annotations

import os
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit


def _stratified_split(
    labels: np.ndarray,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
    return train_idx, test_idx


def _to_dense(X) -> np.ndarray:
    import scipy.sparse

    if scipy.sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _report_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    method: str,
) -> pd.DataFrame:
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    rows = []
    for ct in class_names:
        if ct in report:
            rows.append(
                {
                    "method": method,
                    "cell_type": ct,
                    "precision": report[ct]["precision"],
                    "recall": report[ct]["recall"],
                    "f1": report[ct]["f1-score"],
                    "support": report[ct]["support"],
                }
            )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    rows.append(
        {
            "method": method,
            "cell_type": "MACRO",
            "precision": acc,
            "recall": acc,
            "f1": macro_f1,
            "support": len(y_true),
        }
    )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# scANVI
# ═══════════════════════════════════════════════════════════════════

def _run_scanvi(
    adata_train,
    adata_test,
    y_test: np.ndarray,
    class_names: List[str],
    max_epochs: int = 20,
) -> Optional[pd.DataFrame]:
    try:
        import anndata
        import scvi
    except ImportError:
        print("  scANVI: scvi-tools not installed. pip install scvi-tools")
        return None

    import scanpy as sc

    t0 = time.time()
    print("  scANVI: preparing data...")

    train = adata_train.copy()
    test = adata_test.copy()
    train.obs["split"] = "train"
    test.obs["split"] = "test"

    combined = anndata.concat([train, test], label="split", keys=["train", "test"])
    combined.obs["label_scanvi"] = combined.obs["cell_type"].astype(str)
    combined.obs.loc[combined.obs["split"] == "test", "label_scanvi"] = "Unknown"

    # Normalize if needed
    X_peek = _to_dense(combined.X[:10])
    already_log = bool(np.any(X_peek < 0) or np.max(X_peek) < 20)
    if not already_log:
        sc.pp.normalize_total(combined, target_sum=1e4)
        sc.pp.log1p(combined)

    sc.pp.highly_variable_genes(
        combined,
        n_top_genes=3000,
        subset=True,
        flavor="seurat" if already_log else "seurat_v3",
    )

    scvi.model.SCANVI.setup_anndata(
        combined, labels_key="label_scanvi", unlabeled_category="Unknown"
    )
    model = scvi.model.SCANVI(combined, unlabeled_category="Unknown", n_latent=20, n_layers=2)

    print(f"  scANVI: training ({max_epochs} epochs)...")
    model.train(max_epochs=max_epochs, early_stopping=False)

    preds_all = model.predict()
    test_mask = combined.obs["split"] == "test"
    pred_labels = np.array(preds_all)[test_mask.values]

    cat_map = {ct: i for i, ct in enumerate(class_names)}
    preds_numeric = np.array([cat_map.get(lbl, -1) for lbl in pred_labels])
    valid = preds_numeric >= 0

    if valid.sum() == 0:
        print("  scANVI: no valid predictions")
        return None

    df = _report_metrics(y_test[valid], preds_numeric[valid], class_names, "scANVI")
    macro = df[df["cell_type"] == "MACRO"]["f1"].values[0]
    print(f"  scANVI: macro-F1 = {macro:.3f} ({time.time() - t0:.0f}s)")
    return df


# ═══════════════════════════════════════════════════════════════════
# CellTypist
# ═══════════════════════════════════════════════════════════════════

def _run_celltypist(
    adata_train,
    adata_test,
    y_test: np.ndarray,
    class_names: List[str],
) -> Optional[pd.DataFrame]:
    try:
        import celltypist
    except ImportError:
        print("  CellTypist: not installed. pip install celltypist>=1.6")
        return None

    import scanpy as sc

    t0 = time.time()
    print("  CellTypist: preparing data...")

    train = adata_train.copy()
    test = adata_test.copy()

    # Normalize if needed
    X_peek = _to_dense(train.X[:10])
    already_log = bool(np.any(X_peek < 0) or np.max(X_peek) < 20)
    if not already_log:
        sc.pp.normalize_total(train, target_sum=1e4)
        sc.pp.log1p(train)
        sc.pp.normalize_total(test, target_sum=1e4)
        sc.pp.log1p(test)

    print("  CellTypist: training...")
    ct_model = celltypist.train(
        train,
        labels="cell_type",
        n_jobs=4,
        feature_selection=True,
        solver="lbfgs",
        C=1.0,
        max_iter=200,
    )

    predictions = celltypist.annotate(test, model=ct_model, majority_voting=False)
    pred_labels = predictions.predicted_labels["predicted_labels"].values

    cat_map = {ct: i for i, ct in enumerate(class_names)}
    preds_numeric = np.array([cat_map.get(lbl, -1) for lbl in pred_labels])
    valid = preds_numeric >= 0

    if valid.sum() == 0:
        print("  CellTypist: no valid predictions")
        return None

    df = _report_metrics(y_test[valid], preds_numeric[valid], class_names, "CellTypist")
    macro = df[df["cell_type"] == "MACRO"]["f1"].values[0]
    print(f"  CellTypist: macro-F1 = {macro:.3f} ({time.time() - t0:.0f}s)")
    return df


# ═══════════════════════════════════════════════════════════════════
# Logistic Regression (lightweight baseline)
# ═══════════════════════════════════════════════════════════════════

def _run_logistic_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    print("  Logistic Regression: fitting...")

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_test)

    lr = LogisticRegression(
        max_iter=500, solver="lbfgs", multi_class="multinomial", C=1.0, n_jobs=-1
    )
    lr.fit(Xt, y_train)
    preds = lr.predict(Xv)

    df = _report_metrics(y_test, preds, class_names, "LogisticRegression")
    macro = df[df["cell_type"] == "MACRO"]["f1"].values[0]
    print(f"  LogisticRegression: macro-F1 = {macro:.3f} ({time.time() - t0:.0f}s)")
    return df


# ═══════════════════════════════════════════════════════════════════
# MLP on program scores (tests if attention adds value beyond scores)
# ═══════════════════════════════════════════════════════════════════

def _run_mlp_on_scores(
    program_scores_train: np.ndarray,
    program_scores_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    print("  MLP (program scores): fitting...")

    scaler = StandardScaler()
    Xt = scaler.fit_transform(program_scores_train)
    Xv = scaler.transform(program_scores_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    mlp.fit(Xt, y_train)
    preds = mlp.predict(Xv)

    df = _report_metrics(y_test, preds, class_names, "MLP_scores")
    macro = df[df["cell_type"] == "MACRO"]["f1"].values[0]
    print(f"  MLP (scores): macro-F1 = {macro:.3f} ({time.time() - t0:.0f}s)")
    return df


# ═══════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════

def run_independent_classification(
    adata_ref,
    val_fraction: float = 0.20,
    seed: int = 42,
    output_dir: str = "benchmarks/classification/",
    pathway_mask: Optional[np.ndarray] = None,
    run_scanvi: bool = True,
    run_celltypist: bool = True,
    run_logistic: bool = True,
    run_mlp_scores: bool = True,
    scanvi_epochs: int = 20,
) -> pd.DataFrame:
    """
    Run all independent classification baselines on the same stratified split.

    Parameters
    ----------
    adata_ref       : AnnData with 'cell_type' in .obs
    pathway_mask    : (G, P) binary mask — needed for MLP-on-scores baseline
    """
    os.makedirs(output_dir, exist_ok=True)

    class_names = list(adata_ref.obs["cell_type"].cat.categories)
    ct_codes = adata_ref.obs["cell_type"].cat.codes.values
    train_idx, test_idx = _stratified_split(ct_codes, val_fraction, seed)

    adata_train = adata_ref[train_idx].copy()
    adata_test = adata_ref[test_idx].copy()
    y_test = ct_codes[test_idx]
    y_train = ct_codes[train_idx]

    print(f"\n{'=' * 60}")
    print(f"Independent Classification Benchmark")
    print(f"  Train: {len(train_idx):,}  Test: {len(test_idx):,}")
    print(f"  Classes: {class_names}")
    print(f"  Split: stratified {1 - val_fraction:.0%}/{val_fraction:.0%}, seed={seed}")
    print(f"{'=' * 60}")

    all_dfs = []

    # scANVI
    if run_scanvi:
        df = _run_scanvi(adata_train, adata_test, y_test, class_names, scanvi_epochs)
        if df is not None:
            all_dfs.append(df)

    # CellTypist
    if run_celltypist:
        df = _run_celltypist(adata_train, adata_test, y_test, class_names)
        if df is not None:
            all_dfs.append(df)

    # Logistic Regression on raw expression
    if run_logistic:
        X_train = _to_dense(adata_train.X).astype(np.float32)
        X_test = _to_dense(adata_test.X).astype(np.float32)
        df = _run_logistic_regression(X_train, X_test, y_train, y_test, class_names)
        all_dfs.append(df)

    # MLP on program scores
    if run_mlp_scores and pathway_mask is not None:
        mask_np = np.asarray(pathway_mask, dtype=np.float32)
        norm = 1.0 / np.sqrt(mask_np.sum(axis=0, keepdims=True) + 1e-8)
        scores_train = (_to_dense(adata_train.X).astype(np.float32) @ mask_np) * norm
        scores_test = (_to_dense(adata_test.X).astype(np.float32) @ mask_np) * norm
        df = _run_mlp_on_scores(scores_train, scores_test, y_train, y_test, class_names)
        all_dfs.append(df)

    if not all_dfs:
        print("No methods completed.")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)

    # Summary table
    print(f"\n{'=' * 60}")
    print("Classification Summary (macro F1):")
    summary = result[result["cell_type"] == "MACRO"][["method", "f1"]].sort_values("f1", ascending=False)
    for _, row in summary.iterrows():
        print(f"  {row['method']:<25} {row['f1']:.3f}")
    print(f"{'=' * 60}")

    result.to_csv(os.path.join(output_dir, "classification_benchmark.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "classification_summary.csv"), index=False)
    print(f"Saved to: {output_dir}")

    return result
