"""
ORBIT — Main Pipeline (Alzheimer's Disease)
======================================================
"""
import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse

from model import ORBIT
from data_pipeline import (
    ADDataPipeline,
    PathwayAnalyser,
    download_reference_if_needed,
    predict_cells,
    generate_embeddings,
)
from training import train_orbit, evaluate_accuracy
from visualisation import run_all_visualisations
from dysregulation import run_dysregulation_analysis
from dysregulation_viz import run_dysregulation_visualisations

sc.settings.verbosity = 2

BASE = "."

CONFIG = {
    # ── Reference (Morabito 2021, GSE174367) ────────────────────────────
    # Upload BOTH files to ./data/reference/:
    #   GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5  (261 MB)
    #   GSE174367_snRNA-seq_cell_meta.csv.gz               (425 KB)
    "reference_h5_path":   f"{BASE}/data/reference/GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5",
    "reference_meta_path": f"{BASE}/data/reference/GSE174367_snRNA-seq_cell_meta.csv.gz",
    # True = 7 classes (Exc Inh Ast Oli OPC Mic Vasc)
    # False = separate vascular subtypes
    "consolidate_vascular": True,

    # ── Disease (GSE157827 flat AD files) ────────────────────────────────
    "disease_folder": f"{BASE}/data/disease",
    # None = all 12 donors; or e.g. ["AD1","AD2"] for a quick test
    "disease_donors": None,

    # ── Control (GSE189819 .h5 files) ───────────────────────────────────
    "control_folder": f"{BASE}/data/control",

    # ── Query (optional unannotated cells) ──────────────────────────────
    "query_path": None,

    # ── Outputs ──────────────────────────────────────────────────────────
    "output_dir":     f"{BASE}/outputs/",
    "checkpoint_dir": f"{BASE}/outputs/",

    # ── Architecture ─────────────────────────────────────────────────────
    "embed_dim": 128, "num_heads": 8, "min_pathway_genes": 5,

    # ── Training ─────────────────────────────────────────────────────────
    "epochs": 50, "batch_size": 128, "lr": 1e-3,
    "weight_decay": 0.01, "grad_clip": 1.0,
    "early_stop_patience": 7,
    "w_class": 1.0, "w_diversity": 0.1, "w_entropy": 0.01,
    "confidence_threshold": 0.7,

    # ── AD-relevant genes for visualisation ─────────────────────────────
    "gene_list": [
        "APP", "PSEN1", "PSEN2",
        "APOE", "TREM2", "CLU",
        "MAPT", "SNCA",
        "BDNF", "SYP", "SNAP25",
        "GFAP", "AQP4",
        "TMEM119", "P2RY12", "CX3CR1",
    ],

    # ── Dysregulation ────────────────────────────────────────────────────
    "condition_col":   "condition",
    "healthy_label":   "Control",
    "disease_label":   "AD",
    "run_permutation": False,
    "n_permutations":  1000,
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 1. Ensure reference is downloaded
    download_reference_if_needed(CONFIG["reference_h5_path"])

    # 2. Load all datasets
    pipeline = ADDataPipeline(
        reference_h5_path    = CONFIG["reference_h5_path"],
        reference_meta_path  = CONFIG["reference_meta_path"],
        disease_folder       = CONFIG["disease_folder"],
        control_folder       = CONFIG["control_folder"],
        consolidate_vascular = CONFIG["consolidate_vascular"],
        disease_donors       = CONFIG["disease_donors"],
        query_path           = CONFIG["query_path"],
    )
    adata = pipeline.load_and_process()

    # 3. Pathway mask
    print("\n[3] Building pathway mask...")
    analyser = PathwayAnalyser(min_genes=CONFIG["min_pathway_genes"])
    pathway_mask, pathway_names = analyser.create_pathway_mask(adata)
    assert pathway_mask.shape[0] == adata.n_vars

    # 4. Model
    print("\n[4] Initialising model...")
    adata_ref   = adata[adata.obs["dataset"] == "reference"].copy()
    valid_cats  = adata_ref.obs["cell_type"].cat.categories.tolist()
    print(f"  {len(valid_cats)} cell types: {valid_cats}")
    model = ORBIT(
        num_genes     = adata.n_vars,
        num_classes   = len(valid_cats),
        pathway_mask  = pathway_mask,
        pathway_names = pathway_names,
        embed_dim     = CONFIG["embed_dim"],
        num_heads     = CONFIG["num_heads"],
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Train
    print("\n[5] Training...")
    model, history = train_orbit(
        model, adata_ref,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        grad_clip=CONFIG["grad_clip"],
        checkpoint_dir=CONFIG["checkpoint_dir"],
        device=device,
        w_class=CONFIG["w_class"],
        w_diversity=CONFIG["w_diversity"],
        w_entropy=CONFIG["w_entropy"],
        early_stop_patience=CONFIG["early_stop_patience"],
    )
    pd.DataFrame(history).to_csv(
        os.path.join(CONFIG["output_dir"], "training_history.csv"), index=False
    )

    # 6. Evaluate
    print("\n[6] Evaluating...")
    acc, per_class = evaluate_accuracy(model, adata_ref, device=device)
    pd.Series(per_class).to_csv(
        os.path.join(CONFIG["output_dir"], "per_class_accuracy.csv"),
        header=["accuracy"],
    )

    # 7. Predict disease + control
    print("\n[7] Predicting cell types...")
    target_ds = ["disease", "control"] + (
        ["query"] if CONFIG["query_path"] else []
    )
    adata = predict_cells(
        model, adata, valid_cats,
        target_datasets=target_ds,
        confidence_threshold=CONFIG["confidence_threshold"],
        device=device,
    )
    (
        adata.obs.groupby(["dataset", "predicted"]).size()
        .unstack(fill_value=0)
        .to_csv(os.path.join(CONFIG["output_dir"], "predictions_by_dataset.csv"))
    )

    # 8. Embeddings
    print("\n[8] Generating embeddings...")
    adata = generate_embeddings(model, adata, device=device)

    # 9. Mean attention
    print("\n[9] Computing mean attention matrix...")
    model.eval()
    X_full = np.nan_to_num(
        adata.X.toarray() if scipy.sparse.issparse(adata.X)
        else adata.X.copy()
    ).astype(np.float32)
    Xt = torch.tensor(X_full, dtype=torch.float32).to(device)
    mean_attn = model.compute_mean_attention(Xt, batch_size=256)
    np.save(os.path.join(CONFIG["output_dir"], "mean_attention_matrix.npy"),
            mean_attn)

    # 10. Top co-activations
    print("\n[10] Top program co-activations:")
    P = len(pathway_names)
    for src, tgt, w in sorted(
        [(pathway_names[i], pathway_names[j], mean_attn[i, j])
         for i in range(P) for j in range(P) if i != j],
        key=lambda x: -x[2],
    )[:10]:
        print(f"  {src[:40]:<40} -> {tgt[:40]:<40}  {w:.4f}")

    # 11. Core visualisations
    print("\n[11] Core visualisations...")
    run_all_visualisations(
        model, adata, mean_attn,
        out_dir=CONFIG["output_dir"],
        gene_list=CONFIG["gene_list"],
        device=device,
    )

    # 12. Dysregulation
    print("\n[12] AD vs Control dysregulation analysis...")
    dysreg_results = run_dysregulation_analysis(
        model=model, adata=adata,
        output_dir=os.path.join(CONFIG["output_dir"], "dysregulation/"),
        condition_col=CONFIG["condition_col"],
        celltype_col="combined_cell_type",
        healthy_label=CONFIG["healthy_label"],
        disease_label=CONFIG["disease_label"],
        run_permutation=CONFIG["run_permutation"],
        n_permutations=CONFIG["n_permutations"],
        batch_size=256, device=device,
        gene_names=list(adata.var_names),
    )

    # 13. Dysregulation visualisations
    print("\n[13] Dysregulation visualisations...")
    run_dysregulation_visualisations(
        stratified_results=dysreg_results["stratified"],
        dysreg_results=dysreg_results,
        pathway_names=pathway_names,
        gene_names=list(adata.var_names),
        out_dir=os.path.join(CONFIG["output_dir"], "dysregulation_figures/"),
    )

    # 14. Save AnnData
    print("\n[14] Saving annotated AnnData...")
    out_h5ad = os.path.join(CONFIG["output_dir"],
                            "orbit_AD_annotated.h5ad")
    adata.write_h5ad(out_h5ad)
    print(f"  Saved: {out_h5ad}")

    print(f"\n{'='*55}")
    print(f"ORBIT complete!  Accuracy: {acc:.3f}")
    print(f"Outputs: {CONFIG['output_dir']}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
