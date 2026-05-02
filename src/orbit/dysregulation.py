"""
ORBIT — Dysregulation Analyser
==========================================

Core idea
---------
The model's attention mechanism learns which cell programs co-activate together.
By comparing these co-activation patterns between healthy controls and disease
patients — matched by cell type — we can identify which program relationships
are gained, lost, or reversed in disease.

This is fundamentally different from standard differential expression because:
  - It operates at the level of program RELATIONSHIPS, not individual genes
  - It is cell-type-stratified (each cell type compared independently)
  - It captures REWIRING: relationships that exist in health but not disease,
    and vice versa, not just upregulation/downregulation

"""

import numpy as np
import pandas as pd
import scipy.stats
import scipy.sparse
from typing import Dict, List, Optional, Tuple
import torch


# ---------------------------------------------------------------------------
# Core: compute condition-stratified attention matrices
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_stratified_attention(
    model,
    adata,
    condition_col: str = "condition",
    celltype_col: str = "combined_cell_type",
    healthy_label: str = "Control",
    disease_label: str = "MDD",
    batch_size: int = 256,
    min_cells: int = 30,
    device=None,
) -> Dict:
    """
    Compute per-condition, per-cell-type mean attention matrices.

    For each cell type with at least min_cells in BOTH conditions, computes:
      - attn_healthy[cell_type]  : (P, P) mean attention, healthy cells only
      - attn_disease[cell_type]  : (P, P) mean attention, disease cells only
      - attn_delta[cell_type]    : attn_disease - attn_healthy  (signed delta)

    Parameters
    ----------
    model : ORBIT (trained, eval mode)
    adata : AnnData with condition_col and celltype_col in .obs
    condition_col : column in adata.obs indicating condition
    celltype_col  : column in adata.obs indicating cell type
    healthy_label : value in condition_col for healthy cells
    disease_label : value in condition_col for disease cells
    batch_size : int, for streaming inference
    min_cells  : minimum cells per (condition, cell_type) to include
    device : torch.device

    Returns
    -------
    results : dict with keys:
        'attn_healthy'      : dict {cell_type: (P,P) array}
        'attn_disease'      : dict {cell_type: (P,P) array}
        'attn_delta'        : dict {cell_type: (P,P) array}
        'attn_healthy_all'  : (P,P) pooled across all healthy cells
        'attn_disease_all'  : (P,P) pooled across all disease cells
        'attn_delta_all'    : (P,P) pooled delta
        'cell_types_used'   : list of cell types with sufficient cells
        'cell_counts'       : dict {cell_type: {'healthy': n, 'disease': n}}
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    P = model.num_pathways
    obs = adata.obs

    cell_types = obs[celltype_col].unique().tolist()
    results_h, results_d, results_delta = {}, {}, {}
    cell_counts = {}
    used_types = []

    for ct in cell_types:
        mask_h = (obs[condition_col] == healthy_label) & (obs[celltype_col] == ct)
        mask_d = (obs[condition_col] == disease_label) & (obs[celltype_col] == ct)

        n_h = mask_h.sum()
        n_d = mask_d.sum()
        cell_counts[ct] = {"healthy": int(n_h), "disease": int(n_d)}

        if n_h < min_cells or n_d < min_cells:
            continue

        used_types.append(ct)

        for condition_mask, target_dict, label in [
            (mask_h, results_h, "healthy"),
            (mask_d, results_d, "disease"),
        ]:
            adata_sub = adata[condition_mask]
            X = adata_sub.X.toarray() if scipy.sparse.issparse(adata_sub.X) \
                else adata_sub.X.copy()
            X = np.nan_to_num(X).astype(np.float32)
            Xt = torch.tensor(X, dtype=torch.float32)

            attn_sum = torch.zeros(P, P, device=device)
            count = 0
            for s in range(0, len(Xt), batch_size):
                xb = Xt[s:s+batch_size].to(device)
                out = model(xb, return_attention=True)
                aw = out[2]  # last_attn always at index 2
                attn_sum += aw.mean(dim=(0, 1))
                count += 1
            target_dict[ct] = (attn_sum / count).cpu().numpy()

        results_delta[ct] = results_d[ct] - results_h[ct]

    # Pooled across all used cell types (weighted by cell count)
    def _pool(d):
        if not d:
            return np.zeros((P, P), dtype=np.float32)
        weights = np.array([cell_counts[ct]["healthy"] + cell_counts[ct]["disease"]
                            for ct in used_types if ct in d], dtype=np.float32)
        weights /= weights.sum()
        return sum(w * d[ct] for ct, w in zip(used_types, weights))

    attn_h_all = _pool(results_h)
    attn_d_all = _pool(results_d)

    print(f"\nStratified attention computed:")
    print(f"  Cell types used: {len(used_types)}")
    print(f"  Cell types skipped (too few cells): "
          f"{len(cell_types) - len(used_types)}")

    return {
        "attn_healthy":     results_h,
        "attn_disease":     results_d,
        "attn_delta":       results_delta,
        "attn_healthy_all": attn_h_all,
        "attn_disease_all": attn_d_all,
        "attn_delta_all":   attn_d_all - attn_h_all,
        "cell_types_used":  used_types,
        "cell_counts":      cell_counts,
    }


# ---------------------------------------------------------------------------
# Permutation testing for statistical significance
# ---------------------------------------------------------------------------

def permutation_test_delta(
    model,
    adata,
    stratified_results: Dict,
    condition_col: str = "condition",
    celltype_col:  str = "combined_cell_type",
    healthy_label: str = "Control",
    disease_label: str = "MDD",
    n_permutations: int = 1000,
    batch_size: int = 256,
    fdr_method: str = "fdr_bh",
    device=None,
    seed: int = 42,
) -> Dict:
    """
    Permutation test for each (P1, P2) program pair.

    For each cell type, shuffles condition labels n_permutations times,
    recomputes the delta attention matrix under the null, and reports
    empirical p-values and FDR-corrected q-values.

    This is computationally expensive. For 254 programs and 1000 permutations,
    run time is approximately n_permutations × per-cell-type-inference-time.
    For typical datasets, allow 30–90 minutes on GPU.

    A faster alternative for exploratory analysis: use n_permutations=100.

    Parameters
    ----------
    n_permutations : int
        Number of label shuffles per cell type (1000 for publication, 100 for exploration)
    fdr_method : str
        Passed to statsmodels multipletests ('fdr_bh' = Benjamini-Hochberg)

    Returns
    -------
    stats : dict {cell_type: DataFrame with columns [P1, P2, delta, pval, qval, signed]}
    pooled_stats : DataFrame for the pooled delta
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        raise ImportError("pip install statsmodels")

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    rng = np.random.default_rng(seed)

    P = model.num_pathways
    obs = adata.obs.copy()
    used_types = stratified_results["cell_types_used"]
    stats = {}

    for ct in used_types:
        print(f"  Permutation test: {ct} ...", end=" ", flush=True)

        ct_mask = obs[celltype_col] == ct
        adata_ct = adata[ct_mask]
        cond_labels = obs.loc[ct_mask, condition_col].values

        X = adata_ct.X.toarray() if scipy.sparse.issparse(adata_ct.X) \
            else adata_ct.X.copy()
        X = np.nan_to_num(X).astype(np.float32)
        Xt = torch.tensor(X, dtype=torch.float32)

        # Cache all attention matrices once
        all_attn = []
        with torch.no_grad():
            for s in range(0, len(Xt), batch_size):
                xb = Xt[s:s+batch_size].to(device)
                out = model(xb, return_attention=True)
                aw = out[2]  # last_attn always at index 2
                all_attn.append(aw.mean(dim=1).cpu().numpy())  # (B, P, P)
        all_attn = np.concatenate(all_attn, axis=0)  # (N, P, P)

        # Observed delta
        h_idx = cond_labels == healthy_label
        d_idx = cond_labels == disease_label
        obs_delta = all_attn[d_idx].mean(0) - all_attn[h_idx].mean(0)  # (P, P)

        # Null distribution via label permutation
        n_h = h_idx.sum()
        null_deltas = np.zeros((n_permutations, P, P), dtype=np.float32)
        for i in range(n_permutations):
            perm = rng.permutation(len(cond_labels))
            null_h = perm[:n_h]
            null_d = perm[n_h:]
            null_deltas[i] = all_attn[null_d].mean(0) - all_attn[null_h].mean(0)

        # Empirical two-tailed p-values: fraction of null |delta| >= |observed|
        pvals_mat = (np.abs(null_deltas) >= np.abs(obs_delta)[None]).mean(0)  # (P, P)
        pvals_flat = pvals_mat.ravel()

        # FDR correction
        reject, qvals_flat, _, _ = multipletests(pvals_flat, method=fdr_method)
        qvals_mat = qvals_flat.reshape(P, P)

        # Build output DataFrame
        rows = []
        for p1 in range(P):
            for p2 in range(P):
                if p1 == p2:
                    continue
                rows.append({
                    "P1":    model.pathway_names[p1],
                    "P2":    model.pathway_names[p2],
                    "delta": float(obs_delta[p1, p2]),
                    "pval":  float(pvals_mat[p1, p2]),
                    "qval":  float(qvals_mat[p1, p2]),
                    "direction": "gained" if obs_delta[p1, p2] > 0 else "lost",
                    "cell_type": ct,
                })
        stats[ct] = pd.DataFrame(rows).sort_values("qval")
        print(f"done  ({(qvals_mat < 0.05).sum()} sig pairs at q<0.05)")

    return stats


# ---------------------------------------------------------------------------
# Dysregulation scores
# ---------------------------------------------------------------------------

def compute_dysregulation_scores(
    stratified_results: Dict,
    pathway_names: List[str],
) -> pd.DataFrame:
    """
    Per-program dysregulation score aggregated across cell types.

    Score for program P in cell type C:
        dysreg_score[P, C] = mean |delta_attn[P, :]| + mean |delta_attn[:, P]|
        (outgoing + incoming attention change, averaged)

    Returns DataFrame of shape (num_programs, num_cell_types + ['mean', 'max'])
    """
    used = stratified_results["cell_types_used"]
    P = len(pathway_names)
    score_dict = {}

    for ct in used:
        delta = stratified_results["attn_delta"][ct]             # (P, P)
        abs_delta = np.abs(delta)
        outgoing = abs_delta.mean(axis=1)   # P: how much P1's outgoing edges changed
        incoming = abs_delta.mean(axis=0)   # P: how much P2's incoming edges changed
        score_dict[ct] = (outgoing + incoming) / 2.0

    df = pd.DataFrame(score_dict, index=pathway_names)
    df["mean_across_types"] = df.mean(axis=1)
    df["max_across_types"]  = df.max(axis=1)
    return df.sort_values("mean_across_types", ascending=False)


def compute_vulnerability_ranking(
    stratified_results: Dict,
) -> pd.DataFrame:
    """
    Cell-type vulnerability: which cell types show the most pathway rewiring?

    Vulnerability = mean absolute delta across all (P1, P2) off-diagonal pairs.

    Returns DataFrame sorted by vulnerability score descending.
    """
    rows = []
    for ct in stratified_results["cell_types_used"]:
        delta = stratified_results["attn_delta"][ct]
        np.fill_diagonal(delta, 0)           # exclude self-attention
        n_h = stratified_results["cell_counts"][ct]["healthy"]
        n_d = stratified_results["cell_counts"][ct]["disease"]

        score = np.abs(delta).mean()
        n_gained = (delta > 0.05).sum()      # pairs with meaningful gain
        n_lost   = (delta < -0.05).sum()     # pairs with meaningful loss

        rows.append({
            "cell_type":        ct,
            "vulnerability":    float(score),
            "n_gained_edges":   int(n_gained),
            "n_lost_edges":     int(n_lost),
            "n_healthy_cells":  n_h,
            "n_disease_cells":  n_d,
            "max_gain":         float(delta.max()),
            "max_loss":         float(delta.min()),
        })
    return pd.DataFrame(rows).sort_values("vulnerability", ascending=False)


def compute_gene_dysregulation(
    model,
    stratified_results: Dict,
    cell_types: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Extended backtracing: project program dysregulation onto individual genes.

    gene_dysreg[G, P2] = Σ_P1 mask[G,P1] × delta_attn[P1,P2]

    Positive value: gene G is involved in programs whose outgoing attention
    to P2 INCREASED in disease (gene implicated in gained co-activation with P2).
    Negative value: gene implicated in lost co-activation.

    Parameters
    ----------
    cell_types : list of cell types to compute for, or None for all used types

    Returns
    -------
    Dict {cell_type: (num_genes, P) float32 array}
    """
    mask = model.pathway_mask.cpu().numpy()   # (G, P)
    if cell_types is None:
        cell_types = stratified_results["cell_types_used"]

    gene_dysreg = {}
    for ct in cell_types:
        if ct not in stratified_results["attn_delta"]:
            continue
        delta = stratified_results["attn_delta"][ct]          # (P, P)
        gene_dysreg[ct] = (mask @ delta).astype(np.float32)  # (G, P)
    return gene_dysreg


def find_consistent_dysregulation(
    stratified_results: Dict,
    min_cell_types: int = 3,
    delta_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Identify (P1, P2) pairs that are consistently dysregulated across
    multiple cell types (consistent = same direction, above threshold).

    These are candidates for disease-wide mechanisms rather than
    cell-type-specific vulnerabilities.

    Parameters
    ----------
    min_cell_types : int
        Minimum number of cell types showing consistent dysregulation
    delta_threshold : float
        Minimum |delta| to count as dysregulated in a given cell type

    Returns
    -------
    DataFrame sorted by n_consistent_types descending
    """
    used = stratified_results["cell_types_used"]
    P = len(next(iter(stratified_results["attn_delta"].values())))
    strat_pathway_names = stratified_results.get(
        "pathway_names",
        [str(i) for i in range(P)]
    )

    # Stack deltas across cell types: (num_types, P, P)
    deltas = np.stack([stratified_results["attn_delta"][ct] for ct in used], axis=0)

    rows = []
    for p1 in range(P):
        for p2 in range(P):
            if p1 == p2:
                continue
            d_vec = deltas[:, p1, p2]           # one value per cell type
            gained = (d_vec >  delta_threshold).sum()
            lost   = (d_vec < -delta_threshold).sum()
            consistent = max(gained, lost)

            if consistent >= min_cell_types:
                direction = "gained" if gained >= lost else "lost"
                rows.append({
                    "P1":               strat_pathway_names[p1],
                    "P2_name":          strat_pathway_names[p2],
                    "P1_idx":           p1,
                    "P2_idx":           p2,
                    "P1_name":          strat_pathway_names[p1],
                    "P2_name_full":     strat_pathway_names[p2],
                    "n_consistent":     int(consistent),
                    "n_gained_types":   int(gained),
                    "n_lost_types":     int(lost),
                    "direction":        direction,
                    "mean_delta":       float(d_vec.mean()),
                    "max_abs_delta":    float(np.abs(d_vec).max()),
                })

    if not rows:
        # No pairs met the threshold (common when attention is near-uniform).
        # Return empty DataFrame with correct columns so downstream code
        # (len, to_csv, sort) still works without KeyError.
        return pd.DataFrame(columns=[
            "P1", "P2_name", "P1_idx", "P2_idx",
            "P1_name", "P2_name_full",
            "n_consistent", "n_gained_types", "n_lost_types",
            "direction", "mean_delta", "max_abs_delta",
        ])
    df = pd.DataFrame(rows).sort_values("n_consistent", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Main orchestrator: run full dysregulation analysis
# ---------------------------------------------------------------------------

def run_dysregulation_analysis(
    model,
    adata,
    output_dir: str,
    condition_col: str = "condition",
    celltype_col:  str = "combined_cell_type",
    healthy_label: str = "Control",
    disease_label: str = "MDD",
    run_permutation: bool = True,
    n_permutations:  int = 1000,
    batch_size: int = 256,
    device=None,
    gene_names: Optional[List[str]] = None,
) -> Dict:
    """
    Full dysregulation pipeline. Saves all CSVs and returns result dict.

    Steps:
      1. Compute stratified attention (per cell type, per condition)
      2. Compute dysregulation scores + vulnerability ranking
      3. Find consistent cross-cell-type dysregulation
      4. Gene-level dysregulation projection
      5. Permutation testing (if run_permutation=True)
      6. Save all results to output_dir

    Returns
    -------
    Full results dict for use in visualisation.
    """
    import os
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("ORBIT — Dysregulation Analysis")
    print(f"  Conditions: '{healthy_label}' vs '{disease_label}'")
    print("="*60)

    # Step 1: Stratified attention
    print("\n[Dysreg 1] Computing stratified attention matrices...")
    strat = compute_stratified_attention(
        model, adata,
        condition_col=condition_col, celltype_col=celltype_col,
        healthy_label=healthy_label, disease_label=disease_label,
        batch_size=batch_size, device=device,
    )
    # Inject pathway names so find_consistent_dysregulation can label rows
    strat["pathway_names"] = model.pathway_names
    np.save(os.path.join(output_dir, "attn_healthy_pooled.npy"), strat["attn_healthy_all"])
    np.save(os.path.join(output_dir, "attn_disease_pooled.npy"), strat["attn_disease_all"])
    np.save(os.path.join(output_dir, "attn_delta_pooled.npy"),   strat["attn_delta_all"])
    for ct in strat["cell_types_used"]:
        safe_ct = ct.replace(" ", "_").replace("/", "-")
        np.save(os.path.join(output_dir, f"attn_delta_{safe_ct}.npy"),
                strat["attn_delta"][ct])

    # Step 2: Scores
    print("\n[Dysreg 2] Computing dysregulation scores...")
    score_df = compute_dysregulation_scores(strat, model.pathway_names)
    score_df.to_csv(os.path.join(output_dir, "program_dysregulation_scores.csv"))
    print(f"  Top 5 dysregulated programs (pooled):")
    for prog, row in score_df.head(5).iterrows():
        print(f"    {prog:<45} mean score: {row['mean_across_types']:.4f}")

    vuln_df = compute_vulnerability_ranking(strat)
    vuln_df.to_csv(os.path.join(output_dir, "celltype_vulnerability_ranking.csv"), index=False)
    print(f"\n  Top 5 vulnerable cell types:")
    for _, row in vuln_df.head(5).iterrows():
        print(f"    {row['cell_type']:<45} vuln: {row['vulnerability']:.4f}  "
              f"gained: {row['n_gained_edges']}  lost: {row['n_lost_edges']}")

    # Step 3: Consistent dysregulation
    print("\n[Dysreg 3] Finding consistent cross-cell-type dysregulation...")
    consist_df = find_consistent_dysregulation(strat, min_cell_types=3)
    consist_df.to_csv(os.path.join(output_dir, "consistent_dysregulation.csv"), index=False)
    print(f"  Found {len(consist_df)} consistently dysregulated pairs "
          f"(>=3 cell types, |delta|>=0.05)")

    # Step 4: Gene dysregulation projection
    print("\n[Dysreg 4] Gene-level dysregulation projection...")
    gene_dysreg = compute_gene_dysregulation(model, strat)
    if gene_names is not None:
        # Save top dysregulated genes per cell type
        gene_arr = np.array(gene_names)
        for ct, gd in gene_dysreg.items():
            gene_score = np.abs(gd).mean(axis=1)   # mean across target programs
            top_idx = np.argsort(gene_score)[::-1][:200]
            top_df = pd.DataFrame({
                "gene": gene_arr[top_idx],
                "mean_abs_dysreg": gene_score[top_idx],
            })
            safe_ct = ct.replace(" ", "_").replace("/", "-")
            top_df.to_csv(os.path.join(output_dir, f"top_dysreg_genes_{safe_ct}.csv"),
                          index=False)
    print(f"  Gene dysregulation computed for {len(gene_dysreg)} cell types")

    # Step 5: Permutation testing
    perm_stats = None
    if run_permutation:
        print(f"\n[Dysreg 5] Permutation testing ({n_permutations} permutations)...")
        print("  (This is the most time-intensive step — see runtime estimates)")
        perm_stats = permutation_test_delta(
            model, adata, strat,
            condition_col=condition_col, celltype_col=celltype_col,
            healthy_label=healthy_label, disease_label=disease_label,
            n_permutations=n_permutations, batch_size=batch_size, device=device,
        )
        for ct, df in perm_stats.items():
            safe_ct = ct.replace(" ", "_").replace("/", "-")
            df.to_csv(os.path.join(output_dir, f"perm_stats_{safe_ct}.csv"), index=False)
            sig = df[df["qval"] < 0.05]
            print(f"  {ct}: {len(sig)} significant pairs at q<0.05")

    # Step 6: Cell counts summary
    cc_df = pd.DataFrame(strat["cell_counts"]).T
    cc_df.to_csv(os.path.join(output_dir, "cell_counts_by_condition.csv"))

    print("\n" + "="*60)
    print(f"Dysregulation analysis complete. Outputs: {output_dir}")
    print("="*60)

    return {
        "stratified":    strat,
        "score_df":      score_df,
        "vuln_df":       vuln_df,
        "consist_df":    consist_df,
        "gene_dysreg":   gene_dysreg,
        "perm_stats":    perm_stats,
    }
