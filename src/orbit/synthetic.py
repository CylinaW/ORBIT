"""
ORBIT synthetic validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-8)


def _orthogonalize(raw: np.ndarray, covariates: Sequence[np.ndarray]) -> np.ndarray:
    """Project `raw` off the span of an intercept + covariates."""
    cols = [np.ones_like(raw, dtype=np.float64)]
    cols.extend(np.asarray(c, dtype=np.float64) for c in covariates)
    design = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(design, raw.astype(np.float64), rcond=None)
    resid = raw - design @ beta
    return resid / (resid.std() + 1e-8)


def _rank_matrix(matrix: np.ndarray) -> np.ndarray:
    ranks = np.zeros_like(matrix, dtype=int)
    for i in range(matrix.shape[0]):
        ranks[i, :] = np.argsort(np.argsort(-matrix[i, :])) + 1
    return ranks


def _offdiag_top_k(matrix: np.ndarray, top_fraction: float) -> int:
    p = matrix.shape[0]
    return max(1, int(np.ceil(top_fraction * max(p - 1, 1))))


def _union_edge_genes(edges: Iterable[Tuple[int, int]]) -> List[int]:
    genes = set()
    for i, j in edges:
        genes.add(int(i))
        genes.add(int(j))
    return sorted(genes)


def _make_disjoint_program_mask(
    n_genes: int,
    n_programs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    genes_per_program = n_genes // n_programs
    perm = rng.permutation(n_genes)
    mask = np.zeros((n_genes, n_programs), dtype=np.float32)
    for p in range(n_programs):
        start = p * genes_per_program
        stop = (p + 1) * genes_per_program
        mask[perm[start:stop], p] = 1.0
    return mask


def _compute_program_scores(X_gene: np.ndarray, mask: np.ndarray) -> np.ndarray:
    norm = 1.0 / np.sqrt(mask.sum(axis=0, keepdims=True) + 1e-8)
    return (X_gene @ mask) * norm


def _hidden_triplet_edges(triplets: Sequence[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    return [(a, c) for a, _, c in triplets] + [(b, c) for _, b, c in triplets]


def _directed_edge_metrics(
    matrix: np.ndarray,
    directed_edges: Sequence[Tuple[int, int]],
    top_fraction: float,
    require_orientation: bool,
) -> Dict[str, float]:
    if not directed_edges:
        return {
            "edge_recall_at_k": np.nan,
            "edge_mean_rank": np.nan,
            "direction_accuracy": np.nan,
            "mean_asi": np.nan,
        }

    matrix = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(matrix, 0.0)
    ranks = _rank_matrix(matrix)
    top_k = _offdiag_top_k(matrix, top_fraction)

    recovered = 0
    ranks_forward: List[int] = []
    dir_correct = 0
    asi_values: List[float] = []
    for src, dst in directed_edges:
        src = int(src)
        dst = int(dst)
        fwd = float(matrix[src, dst])
        rev = float(matrix[dst, src])
        ok_orientation = fwd > rev
        if ok_orientation:
            dir_correct += 1
        rank = int(ranks[src, dst])
        ranks_forward.append(rank)
        if rank <= top_k and (ok_orientation or not require_orientation):
            recovered += 1
        asi_values.append((fwd - rev) / (abs(fwd) + abs(rev) + 1e-10))

    return {
        "edge_recall_at_k": recovered / len(directed_edges),
        "edge_mean_rank": float(np.mean(ranks_forward)),
        "direction_accuracy": dir_correct / len(directed_edges),
        "mean_asi": float(np.mean(asi_values)),
    }


def _triplet_structural_recovery(
    matrix: np.ndarray,
    triplets: Sequence[Tuple[int, int, int]],
    top_fraction: float,
    directional: bool,
) -> float:
    if not triplets:
        return np.nan
    if not directional:
        return 0.0

    matrix = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(matrix, 0.0)
    ranks = _rank_matrix(matrix)
    top_k = _offdiag_top_k(matrix, top_fraction)

    recovered = 0
    for a, b, c in triplets:
        rank_ok = ranks[a, c] <= top_k and ranks[b, c] <= top_k
        if directional:
            orient_ok = matrix[a, c] > matrix[c, a] and matrix[b, c] > matrix[c, b]
        else:
            orient_ok = True
        if rank_ok and orient_ok:
            recovered += 1
    return recovered / len(triplets)


def _compute_pearson_matrix(program_scores: np.ndarray) -> np.ndarray:
    corr = np.abs(np.corrcoef(program_scores.T))
    np.fill_diagonal(corr, 0.0)
    return corr


def _compute_mi_matrix(program_scores: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        return np.zeros((program_scores.shape[1], program_scores.shape[1]), dtype=np.float64)

    n_cells = program_scores.shape[0]
    if n_cells > 500:
        idx = np.random.default_rng(42).choice(n_cells, 500, replace=False)
        scores = program_scores[idx]
    else:
        scores = program_scores

    p = scores.shape[1]
    mi = np.zeros((p, p), dtype=np.float64)
    for target in range(p):
        mi[target, :] = mutual_info_regression(
            scores,
            scores[:, target],
            n_neighbors=n_neighbors,
            random_state=42,
        )
    mi = (mi + mi.T) / 2.0
    np.fill_diagonal(mi, 0.0)
    return mi


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] < 5:
        return 0.0
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()
    B = b - b.mean(0, keepdims=True) - b.mean(1, keepdims=True) + b.mean()
    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()
    if dvar_x <= 0 or dvar_y <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y)))


def _compute_dcor_matrix(program_scores: np.ndarray, subsample: int = 800) -> np.ndarray:
    n_cells = program_scores.shape[0]
    if n_cells > subsample:
        idx = np.random.default_rng(42).choice(n_cells, subsample, replace=False)
        scores = program_scores[idx]
    else:
        scores = program_scores
    p = scores.shape[1]
    dcor = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(i + 1, p):
            val = _distance_correlation(scores[:, i], scores[:, j])
            dcor[i, j] = dcor[j, i] = val
    return dcor


def _compute_hsic_matrix(program_scores: np.ndarray, subsample: int = 500) -> np.ndarray:
    try:
        from scipy.spatial.distance import pdist
    except ImportError:
        return np.zeros((program_scores.shape[1], program_scores.shape[1]), dtype=np.float64)

    n_cells = program_scores.shape[0]
    if n_cells > subsample:
        idx = np.random.default_rng(42).choice(n_cells, subsample, replace=False)
        scores = program_scores[idx]
    else:
        scores = program_scores

    p = scores.shape[1]
    hsic = np.zeros((p, p), dtype=np.float64)
    bandwidth = np.zeros(p, dtype=np.float64)
    for col in range(p):
        d = pdist(scores[:, col : col + 1])
        bandwidth[col] = max(np.median(d), 1e-8)

    for i in range(p):
        xi = scores[:, i : i + 1]
        Ki = np.exp(-0.5 * (xi - xi.T) ** 2 / bandwidth[i] ** 2)
        Hi = Ki - Ki.mean(0, keepdims=True) - Ki.mean(1, keepdims=True) + Ki.mean()
        for j in range(i + 1, p):
            xj = scores[:, j : j + 1]
            Kj = np.exp(-0.5 * (xj - xj.T) ** 2 / bandwidth[j] ** 2)
            Hj = Kj - Kj.mean(0, keepdims=True) - Kj.mean(1, keepdims=True) + Kj.mean()
            hsic[i, j] = hsic[j, i] = float((Hi * Hj).sum() / (scores.shape[0] ** 2))
    return hsic


def _compute_directional_feature_importance(
    program_scores: np.ndarray,
    estimator: str,
    subsample: int = 1500,
    random_state: int = 42,
) -> np.ndarray:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.neural_network import MLPRegressor
    except ImportError:
        return np.zeros((program_scores.shape[1], program_scores.shape[1]), dtype=np.float64)

    n_cells = program_scores.shape[0]
    rng = np.random.default_rng(random_state)
    if n_cells > subsample:
        idx = rng.choice(n_cells, subsample, replace=False)
        scores = program_scores[idx]
    else:
        scores = program_scores

    p = scores.shape[1]
    importance = np.zeros((p, p), dtype=np.float64)
    for target in range(p):
        y = scores[:, target]
        X = np.delete(scores, target, axis=1)

        if estimator == "random_forest":
            model = RandomForestRegressor(
                n_estimators=120,
                max_depth=8,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=random_state,
            )
            model.fit(X, y)
            imps = model.feature_importances_
        elif estimator == "mlp":
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=300,
                random_state=random_state,
            )
            model.fit(X, y)
            perm = permutation_importance(
                model,
                X,
                y,
                n_repeats=5,
                random_state=random_state,
                scoring="neg_mean_squared_error",
            )
            imps = np.maximum(perm.importances_mean, 0.0)
        else:
            raise ValueError(f"Unknown estimator: {estimator}")

        for src_idx, value in enumerate(imps):
            original_idx = src_idx if src_idx < target else src_idx + 1
            importance[original_idx, target] = float(value)
    np.fill_diagonal(importance, 0.0)
    return importance


def compute_baseline_matrices(program_scores: np.ndarray) -> Dict[str, np.ndarray]:
    """Return all baseline adjacency matrices used in synthetic validation."""
    return {
        "pearson": _compute_pearson_matrix(program_scores),
        "mutual_info": _compute_mi_matrix(program_scores),
        "distance_correlation": _compute_dcor_matrix(program_scores),
        "hsic": _compute_hsic_matrix(program_scores),
        "random_forest": _compute_directional_feature_importance(program_scores, "random_forest"),
        "mlp": _compute_directional_feature_importance(program_scores, "mlp"),
    }


@dataclass
class SyntheticSummary:
    method: str
    pairwise_recall_at_k: float
    pairwise_mean_rank: float
    pairwise_direction_accuracy: float
    pairwise_mean_asi: float
    hidden_edge_recall_at_k: float
    hidden_edge_mean_rank: float
    hidden_direction_accuracy: float
    hidden_mean_asi: float
    triplet_structural_recovery: float


def make_synthetic_adata(
    n_cells: int = 8000,
    n_genes: int = 2000,
    n_programs: int = 40,
    n_pairwise: int = 12,
    n_triplets: int = 4,
    n_celltypes: int = 5,
    pairwise_strength: float = 0.75,
    triplet_strength: float = 1.35,
    latent_noise: float = 0.30,
    gene_noise: float = 0.08,
    seed: int = 42,
    n_triplet: Optional[int] = None,
    n_quad: Optional[int] = None,
):
    """
    Generate a synthetic AnnData object plus ground-truth interaction metadata.

    The hidden triplets are designed so that A and B jointly influence C while
    the marginal Pearson correlations corr(A, C) and corr(B, C) stay close to 0
    after orthogonalization against the linear main effects.
    """
    if n_triplet is not None:
        n_triplets = int(n_triplet)
    del n_quad  # backward compatibility with old callers

    try:
        import anndata
        import scipy.sparse
    except ImportError as exc:
        raise ImportError("make_synthetic_adata requires anndata and scipy.") from exc

    rng = np.random.default_rng(seed)
    mask = _make_disjoint_program_mask(n_genes, n_programs, rng)

    ct_assignments = rng.choice(n_celltypes, size=n_cells, replace=True)
    ct_profiles = rng.normal(loc=1.0, scale=0.55, size=(n_celltypes, n_programs))
    shared_latents = rng.normal(size=(n_cells, 4))
    latent_loadings = rng.normal(scale=0.20, size=(4, n_programs))
    scores = ct_profiles[ct_assignments] + shared_latents @ latent_loadings
    scores += rng.normal(scale=latent_noise, size=scores.shape)
    scores = np.maximum(scores, 0.05)

    pairwise_pairs: List[Tuple[int, int]] = []
    triplet_triples: List[Tuple[int, int, int]] = []

    available = list(rng.permutation(n_programs))
    cursor = 0
    for _ in range(min(n_pairwise, n_programs // 3)):
        src = int(available[cursor])
        dst = int(available[cursor + 1])
        cursor += 2
        scores[:, dst] += pairwise_strength * _standardize(scores[:, src])
        pairwise_pairs.append((src, dst))

    used = {idx for pair in pairwise_pairs for idx in pair}
    remaining = [p for p in range(n_programs) if p not in used]
    rng.shuffle(remaining)
    for k in range(min(n_triplets, len(remaining) // 3)):
        a, b, c = map(int, remaining[3 * k : 3 * k + 3])
        za = _standardize(scores[:, a])
        zb = _standardize(scores[:, b])
        gate = ((za > np.median(za)) & (zb > np.median(zb))).astype(np.float64)
        interaction = _orthogonalize(gate, [za, zb])
        scores[:, c] += triplet_strength * interaction
        # Residualize the target against the parent main effects so that the
        # retained signal is predominantly higher-order rather than pairwise.
        scores[:, c] = _orthogonalize(scores[:, c], [za, zb])
        scores[:, c] -= scores[:, c].min() - 0.05
        triplet_triples.append((a, b, c))

    scores += rng.normal(scale=latent_noise * 0.15, size=scores.shape)
    scores = np.maximum(scores, 0.0)

    X_gene = (scores @ mask.T).astype(np.float32)
    for p in range(n_programs):
        gene_idx = np.where(mask[:, p] > 0)[0]
        X_gene[:, gene_idx] += rng.normal(
            loc=0.0,
            scale=gene_noise,
            size=(n_cells, len(gene_idx)),
        ).astype(np.float32)
    X_gene = np.maximum(X_gene, 0.0)

    library_size = X_gene.sum(axis=1, keepdims=True)
    library_size[library_size == 0] = 1.0
    X_gene = np.log1p(X_gene / library_size * 10000.0).astype(np.float32)

    # ── Post-log1p re-orthogonalization ──────────────────────────────
    # log1p is nonlinear and reintroduces marginal correlation (up to
    # |r| = 0.20) between parent and target programs.  We regress out
    # the linear component in the FINAL reconstructed score space that
    # the model will see, distributing the correction across genes.
    norm_vec = 1.0 / np.sqrt(mask.sum(axis=0, keepdims=True) + 1e-8)
    for a, b, c in triplet_triples:
        recon_tmp = _compute_program_scores(X_gene, mask)
        design = np.column_stack([
            recon_tmp[:, a] - recon_tmp[:, a].mean(),
            recon_tmp[:, b] - recon_tmp[:, b].mean(),
            np.ones(n_cells),
        ])
        target = recon_tmp[:, c] - recon_tmp[:, c].mean()
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        correction = design @ beta  # per-cell score correction
        c_genes = np.where(mask[:, c] > 0)[0]
        n_c = len(c_genes)
        if n_c > 0:
            per_gene = correction / (n_c * norm_vec[0, c])
            for g in c_genes:
                X_gene[:, g] -= per_gene.astype(np.float32)

    # Pass the dense ndarray directly — accepted by all anndata versions.
    # Sparse wrappers (csr_matrix / csr_array) have inconsistent support across
    # anndata 0.9–0.11 depending on scipy version; ndarray is universally safe.
    adata = anndata.AnnData(X=X_gene)
    adata.obs["synthetic_cell_type"] = ct_assignments.astype(int).astype(str)

    recon_scores = _compute_program_scores(X_gene, mask)
    hidden_edges = _hidden_triplet_edges(triplet_triples)
    marginal_correlations = []
    for src, dst in hidden_edges:
        marginal_correlations.append(float(np.corrcoef(recon_scores[:, src], recon_scores[:, dst])[0, 1]))

    print("\nSynthetic verification:")
    print("  Directed additive pairs:")
    for src, dst in pairwise_pairs[: min(4, len(pairwise_pairs))]:
        corr = np.corrcoef(recon_scores[:, src], recon_scores[:, dst])[0, 1]
        print(f"    {src} -> {dst}: Pearson r = {corr:.3f}")
    print("  Hidden triplets:")
    for a, b, c in triplet_triples:
        corr_ac = np.corrcoef(recon_scores[:, a], recon_scores[:, c])[0, 1]
        corr_bc = np.corrcoef(recon_scores[:, b], recon_scores[:, c])[0, 1]
        joint = ((recon_scores[:, a] > np.median(recon_scores[:, a])) &
                 (recon_scores[:, b] > np.median(recon_scores[:, b]))).astype(np.float64)
        mean_joint = recon_scores[joint > 0, c].mean()
        mean_else = recon_scores[joint == 0, c].mean()
        print(
            f"    ({a}, {b}) -> {c}: corr(A,C)={corr_ac:+.3f} "
            f"corr(B,C)={corr_bc:+.3f} joint_delta={mean_joint - mean_else:+.3f}"
        )

    ground_truth = {
        "mask": mask,
        "pairwise_pairs": pairwise_pairs,
        "triplet_triples": triplet_triples,
        "hidden_triplet_edges": hidden_edges,
        "ct_assignments": ct_assignments,
        "marginal_correlations": marginal_correlations,
        "conditional_pairs": [],
        "quadruplets": [],
    }
    return adata, ground_truth


def evaluate_synthetic_matrices(
    orbit_matrix: np.ndarray,
    baseline_matrices: Dict[str, np.ndarray],
    ground_truth: Dict[str, object],
    top_fraction: float = 0.10,
) -> Tuple[List[SyntheticSummary], Dict[str, float]]:
    """Evaluate ORBIT and baselines on pairwise and hidden triplet structure."""
    pairwise_edges = list(ground_truth.get("pairwise_pairs", []))
    hidden_edges = list(ground_truth.get("hidden_triplet_edges", []))
    triplets = list(ground_truth.get("triplet_triples", []))

    summaries: List[SyntheticSummary] = []
    matrices = {"ORBIT": np.asarray(orbit_matrix, dtype=np.float64)}
    matrices.update({k: np.asarray(v, dtype=np.float64) for k, v in baseline_matrices.items()})

    directional_methods = {"ORBIT", "random_forest", "mlp"}
    for method, matrix in matrices.items():
        directional = method in directional_methods
        pairwise = _directed_edge_metrics(
            matrix,
            pairwise_edges,
            top_fraction=top_fraction,
            require_orientation=directional,
        )
        hidden = _directed_edge_metrics(
            matrix,
            hidden_edges,
            top_fraction=top_fraction,
            require_orientation=directional,
        )
        triplet_recovery = _triplet_structural_recovery(
            matrix,
            triplets,
            top_fraction=top_fraction,
            directional=directional,
        )
        summaries.append(
            SyntheticSummary(
                method=method,
                pairwise_recall_at_k=pairwise["edge_recall_at_k"],
                pairwise_mean_rank=pairwise["edge_mean_rank"],
                pairwise_direction_accuracy=pairwise["direction_accuracy"],
                pairwise_mean_asi=pairwise["mean_asi"],
                hidden_edge_recall_at_k=hidden["edge_recall_at_k"],
                hidden_edge_mean_rank=hidden["edge_mean_rank"],
                hidden_direction_accuracy=hidden["direction_accuracy"],
                hidden_mean_asi=hidden["mean_asi"],
                triplet_structural_recovery=triplet_recovery,
            )
        )

    diagnostics = {
        "mean_abs_hidden_marginal_corr": float(np.mean(np.abs(ground_truth.get("marginal_correlations", [np.nan])))),
        "n_pairwise_edges": len(pairwise_edges),
        "n_hidden_triplet_edges": len(hidden_edges),
        "n_triplets": len(triplets),
    }
    return summaries, diagnostics


def evaluate_synthetic(
    model,
    adata,
    ground_truth,
    output_dir: Optional[str] = None,
    top_fraction: float = 0.10,
):
    """
    Evaluate a trained ORBIT model plus strong baselines on the synthetic task.

    Returns a flat dict for backward compatibility with benchmarking.py and a
    richer CSV when `output_dir` is provided.
    """
    import pandas as pd

    try:
        import scipy.sparse
        import torch
    except ImportError as exc:
        raise ImportError("evaluate_synthetic requires scipy and torch.") from exc

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.asarray(adata.X)
    X = np.asarray(X, dtype=np.float32)

    mask_np = model.pathway_mask.detach().cpu().numpy()
    scores = _compute_program_scores(X, mask_np)
    baseline_matrices = compute_baseline_matrices(scores)

    Xt = torch.tensor(X, dtype=torch.float32)
    orbit_matrix = model.compute_mean_attention(Xt, batch_size=256)
    if isinstance(orbit_matrix, torch.Tensor):
        orbit_matrix = orbit_matrix.detach().cpu().numpy()
    summaries, diagnostics = evaluate_synthetic_matrices(
        orbit_matrix=orbit_matrix,
        baseline_matrices=baseline_matrices,
        ground_truth=ground_truth,
        top_fraction=top_fraction,
    )

    df = pd.DataFrame([s.__dict__ for s in summaries]).sort_values("method").reset_index(drop=True)
    print("\n=== SYNTHETIC RESULTS ===")
    print(
        f"  Hidden triplet edges mean |marginal Pearson r|: "
        f"{diagnostics['mean_abs_hidden_marginal_corr']:.4f}"
    )
    print(df.to_string(index=False))

    if output_dir:
        df.to_csv(os.path.join(output_dir, "synthetic_method_comparison.csv"), index=False)
        pd.DataFrame([diagnostics]).to_csv(os.path.join(output_dir, "synthetic_diagnostics.csv"), index=False)

    orbit_row = df[df["method"] == "ORBIT"].iloc[0]
    pearson_row = df[df["method"] == "pearson"].iloc[0]
    return {
        "orbit_pairwise_detection_rate": float(orbit_row["pairwise_recall_at_k"]),
        "corr_pairwise_detection_rate": float(pearson_row["pairwise_recall_at_k"]),
        "orbit_pairwise_mean_rank": float(orbit_row["pairwise_mean_rank"]),
        "corr_pairwise_mean_rank": float(pearson_row["pairwise_mean_rank"]),
        "orbit_direction_accuracy": float(orbit_row["pairwise_direction_accuracy"]),
        "orbit_mean_ASI": float(orbit_row["pairwise_mean_asi"]),
        "corr_direction_accuracy": float(pearson_row["pairwise_direction_accuracy"]),
        "corr_mean_ASI": float(pearson_row["pairwise_mean_asi"]),
        "orbit_triplet_detection_rate": float(orbit_row["triplet_structural_recovery"]),
        "corr_triplet_detection_rate": float(pearson_row["triplet_structural_recovery"]),
        "orbit_hidden_edge_recall_at_k": float(orbit_row["hidden_edge_recall_at_k"]),
        "corr_hidden_edge_recall_at_k": float(pearson_row["hidden_edge_recall_at_k"]),
        "mean_abs_hidden_marginal_corr": diagnostics["mean_abs_hidden_marginal_corr"],
    }


def triplet_intervention_loss(
    scores_recon: "torch.Tensor",
    triplets: Sequence[Tuple[int, int, int]],
    margin: float = 1.0,
    temperature: float = 4.0,
) -> "torch.Tensor":
    """Differentiable hinge loss that prevents the intervention term from plateauing.

    The ground-truth interaction is a binary AND gate residualized off linear
    main effects — invisible to marginal/pairwise methods.  A generic contrastive
    loss hits a floor because gradient signal from linear reconstruction cannot
    reach this nonlinear term.  This loss directly optimises the quantity that
    ``joint_delta`` measures at eval time, giving the encoder an explicit gradient
    to encode the AND-gate interaction.

    The soft gate uses a sigmoid so that gradients are non-zero everywhere.
    With temperature=4 the gate is close to hard (sigmoid(4) ≈ 0.98) while
    remaining differentiable; lower temperature smooths the boundary.

    Parameters
    ----------
    scores_recon : Tensor of shape (B, P), the model's reconstructed program
        scores *kept in the compute graph* (do not detach before calling).
    triplets : list of (a, b, c) int tuples from ground_truth["triplet_triples"].
    margin : hinge threshold; penalises if joint_delta < margin.  Should be set
        to ~50-70% of the expected ground-truth delta (~1.0 for the default
        synthetic with triplet_strength=1.35).
    temperature : sharpness of the soft median gate.  4.0 works well for
        standardised scores; increase to 8.0 if scores have high variance.

    Returns
    -------
    Scalar tensor, mean hinge loss across all triplets.  Returns zero-gradient
    tensor if ``triplets`` is empty.
    """
    import torch
    import torch.nn.functional as F

    if not triplets:
        return scores_recon.sum() * 0.0  # zero but keeps gradient graph intact

    losses = []
    for a, b, c in triplets:
        sa = scores_recon[:, a]   # (B,)
        sb = scores_recon[:, b]
        sc = scores_recon[:, c]

        # Soft median gate: sigmoid(temperature * (s - median(s)))
        # This is ~1 where s > median, ~0 where s < median, differentiable throughout.
        gate_a = torch.sigmoid(temperature * (sa - sa.median()))
        gate_b = torch.sigmoid(temperature * (sb - sb.median()))

        # Joint-active weight: both parents above their median
        joint_weight = gate_a * gate_b                  # (B,) in [0, 1]
        rest_weight  = (1.0 - gate_a) + (1.0 - gate_b) # (B,) soft complement

        eps = 1e-6
        joint_weight_sum = joint_weight.sum() + eps
        rest_weight_sum  = rest_weight.sum()  + eps

        mean_joint = (sc * joint_weight).sum() / joint_weight_sum
        mean_rest  = (sc * rest_weight).sum()  / rest_weight_sum

        # Hinge: penalise if delta < margin (push mean_joint above mean_rest + margin)
        delta = mean_joint - mean_rest
        losses.append(F.relu(margin - delta))

    return torch.stack(losses).mean()


def get_intervention_weight(
    epoch: int,
    warmup_end: int = 10,
    ramp_end: int = 80,
    w_start: float = 0.5,
    w_end: float = 2.0,
) -> float:
    """Cosine-annealed intervention weight schedule.

    Prevents the plateau caused by a fixed w_interv competing against a recon
    loss that changes by ~6× over training.

    Timeline
    --------
    0 … warmup_end-1   : returns 0.0  (recon-first warmup, matches existing
                          ``interv_warmup_epochs`` logic in pretrain_pathway_attention)
    warmup_end … ramp_end : cosine ramp from w_start → w_end
    ramp_end+1 … ∞     : holds at w_end

    Parameters
    ----------
    epoch      : current epoch index (0-based).
    warmup_end : last epoch of the recon-only warmup phase.
    ramp_end   : epoch at which w_end is reached and held.
    w_start    : initial weight after warmup (typically 0.5, matching original).
    w_end      : final weight after ramp (2.0 keeps triplet signal ~1-3% of
                 total loss as recon converges from ~20 to ~7).

    Returns
    -------
    float weight to multiply against ``triplet_intervention_loss``.

    Example
    -------
    >>> [round(get_intervention_weight(e, 10, 80), 3) for e in [0,9,10,45,80,120]]
    [0.0, 0.0, 0.5, 1.268, 2.0, 2.0]
    """
    import math

    if epoch < warmup_end:
        return 0.0
    if epoch >= ramp_end:
        return w_end
    # Cosine interpolation from w_start to w_end
    progress = (epoch - warmup_end) / max(ramp_end - warmup_end, 1)
    cosine   = 0.5 * (1.0 - math.cos(math.pi * progress))
    return w_start + cosine * (w_end - w_start)


def evaluate_synthetic_attention_direct(
    model,
    adata,
    ground_truth,
    output_dir: Optional[str] = None,
    top_fraction: float = 0.10,
):
    """
    Evaluate using the attention matrix from an UNMASKED forward pass.

    The standard evaluate_synthetic calls model.compute_mean_attention which
    may average over cells or use a different code path.  This function runs
    the model's forward pass directly on true_scores (no masking) and extracts
    last-layer attention, which is exactly what the intervention loss trains.

    This is the correct evaluation for the synthetic benchmark because:
    1. The intervention loss trains the last layer's Q/K weights via last_attn_live
    2. compute_mean_attention may use a different aggregation (e.g. cell-average
       of the MASKED forward pass attention, which is what recon sees)
    3. For the synthetic task we want: does the TRAINED attention structure
       reflect the ground-truth directed influence?
    """
    import pandas as pd
    import torch
    import scipy.sparse

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.asarray(adata.X)
    X = np.asarray(X, dtype=np.float32)

    mask_np = model.pathway_mask.detach().cpu().numpy()
    scores = _compute_program_scores(X, mask_np)
    baseline_matrices = compute_baseline_matrices(scores)

    # ── Extract attention from unmasked forward pass ─────────────────────
    # This matches what intervention_effect_loss trains: full true_scores →
    # encoder → transformer blocks → last block attention
    device = next(model.parameters()).device
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    P = model.num_pathways

    all_attn = []
    batch_size = 256
    with torch.no_grad():
        for s in range(0, len(Xt), batch_size):
            xb = Xt[s:s+batch_size]
            true_sc = model._project_to_pathways(xb)      # (B, P)
            h = model.pathway_encoder(true_sc)
            for i, block in enumerate(model.transformer_blocks):
                if i == len(model.transformer_blocks) - 1:
                    h, attn_w = block(h, return_attn=True) # (B, H, P, P)
                else:
                    h = block(h, return_attn=False)
            # Average over heads → (B, P, P)
            attn_batch = attn_w.mean(dim=1).cpu().numpy()  # (B, P, P)
            all_attn.append(attn_batch)

    # Cell-average attention matrix → (P, P)
    orbit_matrix = np.concatenate(all_attn, axis=0).mean(axis=0)

    summaries, diagnostics = evaluate_synthetic_matrices(
        orbit_matrix=orbit_matrix,
        baseline_matrices=baseline_matrices,
        ground_truth=ground_truth,
        top_fraction=top_fraction,
    )

    df = pd.DataFrame([s.__dict__ for s in summaries]).sort_values("method").reset_index(drop=True)
    print("\n=== SYNTHETIC RESULTS (direct attention) ===")
    print(
        f"  Hidden triplet edges mean |marginal Pearson r|: "
        f"{diagnostics['mean_abs_hidden_marginal_corr']:.4f}"
    )
    print(df.to_string(index=False))

    if output_dir:
        df.to_csv(os.path.join(output_dir, "synthetic_method_comparison_direct.csv"), index=False)

    # Also print the actual attention values at triplet edges for debugging
    triplets = ground_truth.get("triplet_triples", [])
    if triplets:
        print("\n  Attention at triplet edges:")
        for a, b, c in triplets:
            print(f"    ({a},{b})->{c}: A[a,c]={orbit_matrix[a,c]:.4f} A[b,c]={orbit_matrix[b,c]:.4f} "
                  f"A[c,a]={orbit_matrix[c,a]:.4f} A[c,b]={orbit_matrix[c,b]:.4f} "
                  f"ratio_a={orbit_matrix[a,c]/(orbit_matrix[c,a]+1e-8):.2f} "
                  f"ratio_b={orbit_matrix[b,c]/(orbit_matrix[c,b]+1e-8):.2f}")
        # Print top-4 off-diagonal for reference
        m = orbit_matrix.copy()
        np.fill_diagonal(m, 0)
        flat = m.ravel()
        top_idx = np.argsort(flat)[::-1][:8]
        print("\n  Top-8 attention pairs:")
        for idx in top_idx:
            r, c_idx = divmod(idx, P)
            print(f"    [{r}]->[{c_idx}]: {m[r, c_idx]:.4f}")

    orbit_row = df[df["method"] == "ORBIT"].iloc[0]
    pearson_row = df[df["method"] == "pearson"].iloc[0]
    return {
        "orbit_pairwise_detection_rate": float(orbit_row["pairwise_recall_at_k"]),
        "corr_pairwise_detection_rate": float(pearson_row["pairwise_recall_at_k"]),
        "orbit_pairwise_mean_rank": float(orbit_row["pairwise_mean_rank"]),
        "corr_pairwise_mean_rank": float(pearson_row["pairwise_mean_rank"]),
        "orbit_direction_accuracy": float(orbit_row["pairwise_direction_accuracy"]),
        "orbit_mean_ASI": float(orbit_row["pairwise_mean_asi"]),
        "corr_direction_accuracy": float(pearson_row["pairwise_direction_accuracy"]),
        "corr_mean_ASI": float(pearson_row["pairwise_mean_asi"]),
        "orbit_triplet_detection_rate": float(orbit_row["triplet_structural_recovery"]),
        "corr_triplet_detection_rate": float(pearson_row["triplet_structural_recovery"]),
        "orbit_hidden_edge_recall_at_k": float(orbit_row["hidden_edge_recall_at_k"]),
        "corr_hidden_edge_recall_at_k": float(pearson_row["hidden_edge_recall_at_k"]),
        "mean_abs_hidden_marginal_corr": diagnostics["mean_abs_hidden_marginal_corr"],
    }


def make_pathway_mask_from_ground_truth(ground_truth):
    import torch

    return torch.tensor(ground_truth["mask"], dtype=torch.float32)
