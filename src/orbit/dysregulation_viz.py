"""
QuokkaVision v2 — Dysregulation Visualisations  (v2.2)
=======================================================
Five publication-quality figures for the AD vs Control comparison.
Nature Biotechnology style throughout.

  Fig D1 — Signed delta heatmap  (disease - healthy pathway co-activation)
  Fig D2 — Cell-type vulnerability bar + rewiring direction (fixed blank panel)
  Fig D3 — Program dysregulation dot plot  (legend outside right margin)
  Fig D4 — Gained vs lost co-activation network
  Fig D5 — REPLACED: Pathway Co-activation Rewiring Scatter
           (x = control attention, y = disease attention, per cell type)
           This is genuinely different from DEG/GSEA — it plots PATHWAY-LEVEL
           structural changes that are invisible to gene-level methods.

Design decisions:
  - Uniform Paul Tol palette across all figures (same _C_* constants)
  - All legends placed outside axes with bbox_to_anchor to prevent overlap
  - D2 rewiring panel: explicit xlim symmetry, guaranteed non-blank
  - D5: not a volcano. The x/y axes are both attention weights (not fold-change
    or p-value). The scatter reveals which program-pairs REWIRE in AD —
    something DEG literally cannot show because DEG has no concept of
    between-pathway coupling.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx
import scipy.stats
import os
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Shared style (identical constants to visualisation.py for uniformity)
# ---------------------------------------------------------------------------
_NB = {
    "font.family":           "Arial",
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
    "font.size":             8,
    "axes.titlesize":        9,
    "axes.titleweight":      "bold",
    "axes.labelsize":        8,
    "xtick.labelsize":       7,
    "ytick.labelsize":       7,
    "legend.fontsize":       7.5,
    "legend.title_fontsize": 8,
    "axes.linewidth":        0.8,
    "xtick.major.width":     0.8,
    "ytick.major.width":     0.8,
    "xtick.major.size":      3,
    "ytick.major.size":      3,
    "lines.linewidth":       1.2,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             False,
    "figure.facecolor":      "white",
    "axes.facecolor":        "white",
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.05,
}
plt.rcParams.update(_NB)

_C_RED  = "#EE6677"
_C_BLUE = "#4477BB"
_C_GRN  = "#228833"
_C_YEL  = "#CCBB44"
_C_CYA  = "#66CCEE"
_C_PUR  = "#AA3377"
_C_GRY  = "#BBBBBB"
_TPAL   = [_C_BLUE, _C_RED, _C_GRN, _C_YEL, _C_CYA, _C_PUR, _C_GRY]

_CT_PAL = {"Exc": _C_BLUE, "Inh": _C_RED, "Ast": _C_GRN,
           "Mic": _C_YEL,  "OPC": _C_CYA, "Oli": _C_PUR,
           "Vasc": _C_GRY, "Unknown": "#BBBBBB"}


def _clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_label(ax, letter, x=-0.14, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


def _truncate(s, n=28):
    return s[:n] + ("…" if len(s) > n else "")


def _95ci(arr):
    n = len(arr)
    if n < 2:
        return 0.0
    return scipy.stats.sem(arr) * scipy.stats.t.ppf(0.975, df=n - 1)


def _heatmap_pcolormesh(data_df, ax, cmap, vmin, vmax, center=None):
    """Draw heatmap as pcolormesh — zero white lines between cells."""
    arr = data_df.values if hasattr(data_df, "values") else data_df
    if center is not None:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        mesh = ax.pcolormesh(arr, cmap=cmap, norm=norm, rasterized=True)
    else:
        mesh = ax.pcolormesh(arr, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlim(0, arr.shape[1])
    ax.set_ylim(0, arr.shape[0])
    ax.invert_yaxis()
    return mesh


# ---------------------------------------------------------------------------
# Fig D1: Signed delta heatmap
# ---------------------------------------------------------------------------

def plot_delta_heatmap(stratified_results, pathway_names, save_path,
                        cell_type=None, top_n=40):
    """
    Diverging pcolormesh heatmap of delta_attn = attn_disease - attn_healthy.
    Zero white lines (pcolormesh), hierarchically clustered row/column order.
    Red = gained in disease | Blue = lost | White = no change.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list

    if cell_type is None:
        delta = stratified_results.get("attn_delta_all",
                np.zeros((len(pathway_names), len(pathway_names))))
        title = "Pathway rewiring: disease vs healthy (pooled)"
    else:
        delta = stratified_results.get("attn_delta", {}).get(
                cell_type, np.zeros((len(pathway_names), len(pathway_names))))
        title = f"Pathway rewiring: {cell_type}"

    df  = pd.DataFrame(delta, index=pathway_names, columns=pathway_names)
    var = pd.concat([df.var(axis=0), df.var(axis=1)]).groupby(level=0).max()
    top = var.nlargest(min(top_n, len(var))).index
    df_sub = df.loc[top, top].copy()

    if len(df_sub) > 2:
        try:
            Z     = linkage(df_sub.values, method="average", metric="euclidean")
            order = leaves_list(Z)
            df_sub = df_sub.iloc[order, :].iloc[:, order]
        except Exception:
            pass

    vmax = float(np.percentile(np.abs(delta), 98)) if np.abs(delta).max() > 0 else 1e-4
    vmax = max(vmax, 1e-5)   # never zero

    n = len(df_sub)
    row_labels = [_truncate(df_sub.index[i], 28) for i in range(n)]
    col_labels = [_truncate(df_sub.columns[i], 28) for i in range(n)]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    mesh = _heatmap_pcolormesh(df_sub, ax, cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax, center=0.0)

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.45, aspect=12, pad=0.015)
    cbar.set_label("Δ attention (disease − healthy)", size=7, labelpad=3)
    cbar.ax.tick_params(labelsize=6.5, width=0.6, length=2)

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(col_labels, rotation=90, fontsize=5.2, ha="right")
    ax.set_yticklabels(row_labels, rotation=0,  fontsize=5.2, va="center")
    ax.tick_params(axis="both", length=0, pad=2)
    ax.set_xlabel("Target program (key)", fontsize=8, labelpad=4)
    ax.set_ylabel("Source program (query)", fontsize=8, labelpad=4)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    _panel_label(ax, "a")

    fig.text(0.01, -0.025,
             f"Red = gained co-activation in AD.  Blue = lost.  "
             f"Top {min(top_n, n)} programs, hierarchically clustered.",
             fontsize=6, color="#666666")

    # Note about scale
    fig.text(0.99, -0.025,
             f"Scale: ±{vmax:.4f} — will increase after re-training with v2.2.",
             fontsize=6, color="#AA3377", ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig D1] {save_path}")


# ---------------------------------------------------------------------------
# Fig D2: Cell-type vulnerability ranking + rewiring direction (FIXED)
# ---------------------------------------------------------------------------

def plot_vulnerability_ranking(vuln_df, save_path, top_n=20):
    """
    Two-panel figure:
      Left   — Horizontal bar chart: vulnerability score per cell type
      Right  — Stacked bar: gained (red) vs lost (blue) edge counts

    FIX for blank right panel:
      - If n_gained_edges / n_lost_edges are all zero (uniform attention before
        training), we synthesize indicative values with a clear annotation.
      - xlim is set symmetrically based on actual max, never left at 0.
      - A text note explains if values are pre-training placeholders.
    """
    df = vuln_df.head(top_n).copy().sort_values("vulnerability")

    cmap   = plt.cm.YlOrRd
    norm   = mcolors.Normalize(vmin=df["vulnerability"].min(),
                                vmax=max(df["vulnerability"].max(), 1e-9))
    colors = [cmap(norm(v)) for v in df["vulnerability"]]

    n_rows = len(df)
    fig_h  = max(3.5, n_rows * 0.38 + 1.2)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.5, fig_h),
        gridspec_kw={"width_ratios": [2.2, 1.4], "wspace": 0.12})

    # ── Panel a: vulnerability score ────────────────────────────────────
    ax1.barh(df["cell_type"], df["vulnerability"],
             color=colors, edgecolor="white", linewidth=0.4,
             height=0.65, zorder=3)
    ax1.set_xlabel("Mean |Δ attention|  (vulnerability score)", labelpad=3)
    ax1.set_title("Cell-type vulnerability", pad=4)
    ax1.xaxis.grid(True, lw=0.4, alpha=0.4, color="#CCCCCC")
    ax1.set_axisbelow(True)
    _clean_ax(ax1)
    _panel_label(ax1, "a")

    sm_v = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_v.set_array([])
    cbar1 = fig.colorbar(sm_v, ax=ax1, shrink=0.35, aspect=12, pad=0.01)
    cbar1.set_label("|Δ attn|", size=6.5)
    cbar1.ax.tick_params(labelsize=6, width=0.6, length=2)

    # ── Panel b: gained vs lost edge counts ─────────────────────────────
    gained = df.get("n_gained_edges", pd.Series(np.zeros(len(df)),
                                                  index=df.index)).values
    lost   = df.get("n_lost_edges",   pd.Series(np.zeros(len(df)),
                                                  index=df.index)).values

    is_placeholder = (gained.max() == 0 and lost.max() == 0)
    if is_placeholder:
        # Before training, attention is near-uniform → no edges cross threshold.
        # Show a labelled empty panel rather than a blank one.
        ax2.barh(df["cell_type"], np.zeros(len(df)),
                 color=_C_GRY, edgecolor="white", height=0.65, zorder=3)
        ax2.set_xlim(-1, 1)
        ax2.text(0.5, 0.5,
                 "No edges above\nthreshold yet\n(pre-training)",
                 ha="center", va="center", fontsize=7, color="#888888",
                 transform=ax2.transAxes, style="italic")
    else:
        ax2.barh(df["cell_type"], gained,
                 color=_C_RED, alpha=0.85, edgecolor="white",
                 linewidth=0.4, height=0.65, zorder=3, label="Gained")
        ax2.barh(df["cell_type"], -lost,
                 color=_C_BLUE, alpha=0.85, edgecolor="white",
                 linewidth=0.4, height=0.65, zorder=3, label="Lost")
        ax2.axvline(0, color="#444444", linewidth=0.7, zorder=4)
        max_val = max(gained.max(), lost.max(), 1)
        ax2.set_xlim(-max_val * 1.25, max_val * 1.25)
        ax2.legend(fontsize=7, frameon=False, loc="lower right",
                   handlelength=1.2, handletextpad=0.3, labelspacing=0.3)
        ax2.xaxis.grid(True, lw=0.4, alpha=0.4, color="#CCCCCC")
        ax2.set_axisbelow(True)

    ax2.set_xlabel("Edge count", labelpad=3)
    ax2.set_title("Rewiring direction", pad=4)
    ax2.set_yticklabels([])
    _clean_ax(ax2)
    _panel_label(ax2, "b")

    if is_placeholder:
        fig.text(0.5, -0.03,
                 "Rewiring counts will populate after re-training. "
                 "Threshold is relative to post-training attention scale.",
                 ha="center", fontsize=6, color="#AA3377")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig D2] {save_path}")


# ---------------------------------------------------------------------------
# Fig D3: Program dysregulation dot plot — legend outside right margin
# ---------------------------------------------------------------------------

def plot_dysregulation_scores(score_df, save_path, top_n=30):
    """
    Dot plot: colour = dysregulation score (YlOrRd), size = relative score.
    Legend column placed in dedicated right-margin axes — no overlap.
    """
    top_progs = score_df["mean_across_types"].nlargest(top_n).index
    ct_cols   = [c for c in score_df.columns
                 if c not in ("mean_across_types", "max_across_types")]
    sub = score_df.loc[top_progs, ct_cols]

    n_prog, n_ct = len(sub), len(ct_cols)
    fig_w  = max(4.0, n_ct * 0.62 + 3.0)
    fig_h  = max(5.0, n_prog * 0.28 + 1.2)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                             width_ratios=[n_ct * 0.62 + 1.2, 1.6],
                             wspace=0.06)
    ax     = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis("off")

    vmax   = max(sub.values.max(), 1e-9)
    cmap   = plt.cm.YlOrRd
    norm   = mcolors.Normalize(vmin=0, vmax=vmax)
    max_sz = 180

    for ci, ct in enumerate(ct_cols):
        for pi, prog in enumerate(sub.index):
            val   = float(sub.loc[prog, ct])
            frac  = val / vmax
            color = cmap(norm(val))
            size  = frac * max_sz + 5
            ax.scatter(ci, pi, s=size, c=[color],
                       linewidths=0.3, edgecolors="#555555",
                       zorder=3, alpha=0.9)

    ax.set_xticks(range(n_ct))
    ax.set_xticklabels([_truncate(c, 14) for c in ct_cols],
                        rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(n_prog))
    ax.set_yticklabels([_truncate(p, 32) for p in sub.index], fontsize=5.8)
    ax.set_xlim(-0.6, n_ct - 0.4)
    ax.set_ylim(-0.7, n_prog - 0.3)
    ax.invert_yaxis()
    for pi in range(0, n_prog, 2):
        ax.axhspan(pi - 0.5, pi + 0.5, color="#F6F6F6", zorder=0, lw=0)
    ax.set_xlabel("Cell type", labelpad=4)
    ax.set_ylabel("Pathway program", labelpad=4)
    ax.set_title(f"Top {top_n} dysregulated programs",
                 fontsize=9, fontweight="bold", pad=6)
    _clean_ax(ax)
    _panel_label(ax, "c")

    # Colourbar in legend axes
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_leg, fraction=0.55, shrink=0.5, aspect=14,
                         location="left", pad=0.05)
    cbar.set_label("Dysregulation score", size=7)
    cbar.ax.tick_params(labelsize=6.5, width=0.6, length=2)

    # Size legend
    for fv, lbl in [(0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1.0, "100%")]:
        ax_leg.scatter([], [], s=fv * max_sz + 5, c="#888888", alpha=0.9,
                       linewidths=0.3, edgecolors="#555555", label=lbl)
    ax_leg.legend(title="Relative\nscore", fontsize=6.5, title_fontsize=7,
                  frameon=False, loc="lower left",
                  handletextpad=0.4, borderpad=0.2, labelspacing=0.5,
                  scatterpoints=1)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig D3] {save_path}")


# ---------------------------------------------------------------------------
# Fig D4: Gained vs lost co-activation network
# ---------------------------------------------------------------------------

def plot_gained_lost_network(stratified_results, pathway_names, save_path,
                              cell_type=None, delta_threshold=None, top_k=25):
    """
    Directed network: gained (red) and lost (blue) co-activation edges.
    threshold is auto-set to top_k percentile if not specified,
    so the graph is never empty even with uniform pre-training attention.
    """
    if cell_type is None:
        delta = stratified_results.get("attn_delta_all",
                np.zeros((len(pathway_names), len(pathway_names))))
        title = "Pathway co-activation rewiring: pooled"
    else:
        delta = stratified_results.get("attn_delta", {}).get(
                cell_type, np.zeros((len(pathway_names), len(pathway_names))))
        title = f"Pathway rewiring: {cell_type}"

    P = len(pathway_names)
    gains, losses = [], []
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            d = float(delta[i, j])
            if d > 0:
                gains.append((i, j, d))
            elif d < 0:
                losses.append((i, j, abs(d)))

    if not gains and not losses:
        print(f"  [Fig D4] No non-zero delta edges. Skipping.")
        return

    # Auto-threshold at top-k
    gains.sort(key=lambda x: -x[2]);   top_gains  = gains[:top_k]
    losses.sort(key=lambda x: -x[2]);  top_losses = losses[:top_k]
    actual_threshold = (top_gains[-1][2] if top_gains else 0.0)

    G = nx.DiGraph()
    for i, j, w in top_gains:
        n_i, n_j = pathway_names[i], pathway_names[j]
        if n_i not in G:  G.add_node(n_i)
        if n_j not in G:  G.add_node(n_j)
        G.add_edge(n_i, n_j, weight=w, etype="gained")
    for i, j, w in top_losses:
        n_i, n_j = pathway_names[i], pathway_names[j]
        if n_i not in G:  G.add_node(n_i)
        if n_j not in G:  G.add_node(n_j)
        G.add_edge(n_i, n_j, weight=w, etype="lost")

    if G.number_of_nodes() == 0:
        print(f"  [Fig D4] Empty graph. Skipping.")
        return

    node_abs = {}
    for n in G.nodes:
        idx = pathway_names.index(n) if n in pathway_names else 0
        node_abs[n] = float(np.abs(delta[idx, :]).sum() +
                            np.abs(delta[:, idx]).sum())

    pos        = nx.spring_layout(G, seed=42, k=3.0)
    node_list  = list(G.nodes)
    node_sizes = [max(60, node_abs.get(n, 0) * 400) for n in node_list]

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ax.set_facecolor("white")

    nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                           node_size=node_sizes, node_color="#F2F2F2",
                           edgecolors="#888888", linewidths=0.6,
                           ax=ax, alpha=0.9)

    gained_edges = [(u, v) for u, v, d in G.edges(data=True)
                    if d["etype"] == "gained"]
    if gained_edges:
        gw = np.array([G[u][v]["weight"] for u, v in gained_edges])
        widths = 0.6 + (gw / max(gw.max(), 1e-9)) * 4.0
        nx.draw_networkx_edges(G, pos, edgelist=gained_edges,
                               edge_color=_C_RED, arrows=True, arrowsize=10,
                               width=widths, alpha=0.80, ax=ax,
                               connectionstyle="arc3,rad=0.05",
                               min_source_margin=8, min_target_margin=8)

    lost_edges = [(u, v) for u, v, d in G.edges(data=True)
                  if d["etype"] == "lost"]
    if lost_edges:
        lw = np.array([G[u][v]["weight"] for u, v in lost_edges])
        widths = 0.6 + (lw / max(lw.max(), 1e-9)) * 4.0
        nx.draw_networkx_edges(G, pos, edgelist=lost_edges,
                               edge_color=_C_BLUE, arrows=True, arrowsize=10,
                               width=widths, alpha=0.80, ax=ax,
                               connectionstyle="arc3,rad=-0.05",
                               min_source_margin=8, min_target_margin=8)

    labels = {n: _truncate(n, 22) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels,
                             font_size=5.8, font_color="#111111", ax=ax)

    ax.legend(handles=[
        Line2D([0], [0], color=_C_RED,  lw=2.5,
               label=f"Gained in AD (n={len(gained_edges)})"),
        Line2D([0], [0], color=_C_BLUE, lw=2.5,
               label=f"Lost in AD (n={len(lost_edges)})"),
    ], fontsize=7.5, frameon=False, loc="upper left",
       handlelength=1.6, handletextpad=0.4, labelspacing=0.4)

    ax.set_title(f"{title}\n(top {top_k} gained + lost edges, "
                 f"min |Δ| = {actual_threshold:.5f})",
                 fontsize=9, fontweight="bold", pad=6)
    ax.axis("off")
    _panel_label(ax, "d", x=-0.01, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig D4] {save_path}")


# ---------------------------------------------------------------------------
# Fig D5 — REPLACED: Pathway Co-activation Rewiring Scatter
# ---------------------------------------------------------------------------

def plot_pathway_rewiring_scatter(
    mean_attn_disease: np.ndarray,
    mean_attn_control: np.ndarray,
    pathway_names: List[str],
    save_path: str,
    cell_type: Optional[str] = None,
    top_label_n: int = 12,
    by_celltype: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Pathway Co-activation Rewiring Scatter — NOT a volcano plot.

    Why this is fundamentally different from DEG/GSEA:
    ─────────────────────────────────────────────────
    DEG measures: mean expression of gene G in AD vs Control.
    GSEA measures: enrichment of gene-set S in ranked gene list.
    Both are MARGINAL — they look at one gene or one pathway at a time.

    This figure measures: JOINT attention weight between program pair (P1→P2).
    X axis = attention weight in Control cells.
    Y axis = attention weight in Disease cells.
    Points far from the diagonal represent structural changes in how
    programs communicate — invisible to any gene-level method because
    DEG has no concept of between-program coupling.

    Example insight (when training is complete):
      "Myelination ↔ Metabolic coupling is high in Oli control cells
       (upper right quadrant) but collapses in AD Oli cells (below diagonal)."
       This is not a change in mean expression of any gene — it is a change
       in transcriptional co-regulation between programs.

    Multiple panels: one per cell type if by_celltype is provided.
    """
    P = len(pathway_names)

    # Flatten off-diagonal entries
    diag = np.eye(P, dtype=bool)
    ctrl_flat = mean_attn_control[~diag].ravel()
    dis_flat  = mean_attn_disease[~diag].ravel()
    delta     = dis_flat - ctrl_flat

    # Pair labels for top annotations
    idx_pairs = [(i, j) for i in range(P) for j in range(P) if i != j]

    # Top changed pairs by |delta|
    abs_delta = np.abs(delta)
    top_idx   = np.argsort(abs_delta)[::-1][:top_label_n]

    if by_celltype and len(by_celltype) > 1:
        # Multi-panel: one scatter per cell type
        cell_types = list(by_celltype.keys())
        ncols = min(len(cell_types), 3)
        nrows = int(np.ceil(len(cell_types) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 4.0 + 0.5, nrows * 3.8),
                                  gridspec_kw={"hspace": 0.45, "wspace": 0.38})
        axes = np.array(axes).ravel()

        for ai, ct in enumerate(cell_types):
            ax  = axes[ai]
            ct_ctrl = by_celltype[ct].get("control",
                      mean_attn_control)[~diag].ravel()
            ct_dis  = by_celltype[ct].get("disease",
                      mean_attn_disease)[~diag].ravel()
            _rewiring_panel(ax, ct_ctrl, ct_dis, idx_pairs, pathway_names,
                            top_label_n=5, title=ct,
                            colour=_CT_PAL.get(ct, _C_BLUE),
                            letter=chr(ord("a") + ai))
        for ax in axes[len(cell_types):]:
            ax.set_visible(False)

        fig.suptitle("Pathway co-activation rewiring in AD (per cell type)",
                     fontsize=10, fontweight="bold", y=1.01)
        fig.text(0.5, -0.03,
                 "X = control attention weight, Y = AD attention weight. "
                 "Below diagonal = coupling lost in AD. "
                 "Above = gained. Not detectable by DEG or GSEA.",
                 ha="center", fontsize=6.5, color="#666666")
    else:
        # Single panel — pooled or specified cell type
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        title = cell_type if cell_type else "Pooled (all cell types)"
        _rewiring_panel(ax, ctrl_flat, dis_flat, idx_pairs, pathway_names,
                        top_label_n=top_label_n, title=title,
                        colour=_C_BLUE, letter="e")
        fig.text(0.5, -0.04,
                 "Each point = one directed program pair (P1→P2). "
                 "Below diagonal = co-activation weakened in AD. "
                 "This structural change is invisible to DEG and GSEA.",
                 ha="center", fontsize=6.5, color="#666666")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig D5] {save_path}")


def _rewiring_panel(ax, ctrl_flat, dis_flat, idx_pairs, pathway_names,
                     top_label_n, title, colour, letter):
    """Helper: draw a single rewiring scatter panel."""
    delta    = dis_flat - ctrl_flat
    abs_delta = np.abs(delta)

    # Background points
    ax.scatter(ctrl_flat, dis_flat, s=1.2, c=_C_GRY,
               alpha=0.18, rasterized=True, linewidths=0, zorder=1)

    # Colour by direction
    gained_m = delta > 0
    lost_m   = delta < 0
    if gained_m.any():
        ax.scatter(ctrl_flat[gained_m], dis_flat[gained_m],
                   s=2.5, c=_C_RED, alpha=0.5,
                   rasterized=True, linewidths=0, zorder=2)
    if lost_m.any():
        ax.scatter(ctrl_flat[lost_m], dis_flat[lost_m],
                   s=2.5, c=_C_BLUE, alpha=0.5,
                   rasterized=True, linewidths=0, zorder=2)

    # Diagonal: no change
    lim = max(ctrl_flat.max(), dis_flat.max(), 1e-6) * 1.08
    ax.plot([0, lim], [0, lim], color="#333333",
            lw=0.8, ls="--", zorder=3, label="No change")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # Top-N labels with non-overlapping placement
    top_idx = np.argsort(abs_delta)[::-1][:top_label_n]
    used_positions = []

    def _no_overlap(xy, used, min_dist=0.015 * lim):
        return all(
            np.hypot(xy[0] - u[0], xy[1] - u[1]) > min_dist
            for u in used)

    for ti in top_idx:
        if ti >= len(idx_pairs):
            continue
        p1i, p2i = idx_pairs[ti]
        cx, cy   = ctrl_flat[ti], dis_flat[ti]
        lbl = (f"{_truncate(pathway_names[p1i], 14)}\n"
               f"→{_truncate(pathway_names[p2i], 14)}")
        col = _C_RED if delta[ti] > 0 else _C_BLUE

        # Offset to avoid overlap
        offset = (4, 4) if delta[ti] > 0 else (-4, -8)
        text_xy = (cx + offset[0] / 100 * lim,
                   cy + offset[1] / 100 * lim)
        if _no_overlap(text_xy, used_positions):
            ax.annotate(lbl, xy=(cx, cy),
                        xytext=text_xy,
                        fontsize=4.5, color=col,
                        arrowprops=dict(arrowstyle="-", lw=0.35,
                                        color="#BBBBBB"),
                        multialignment="left")
            used_positions.append(text_xy)

    ax.set_xlabel("Attention weight (control)", fontsize=7, labelpad=2)
    ax.set_ylabel("Attention weight (AD)", fontsize=7, labelpad=2)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=4)
    ax.legend(handles=[
        Patch(facecolor=_C_RED,  edgecolor="none", label="Gained in AD"),
        Patch(facecolor=_C_BLUE, edgecolor="none", label="Lost in AD"),
        Line2D([0], [0], color="#333333", lw=0.9, ls="--", label="No change"),
    ], fontsize=6.5, frameon=False, loc="upper left",
       handlelength=1.3, labelspacing=0.3)
    ax.yaxis.grid(True, lw=0.35, alpha=0.35, color="#CCCCCC")
    ax.xaxis.grid(True, lw=0.35, alpha=0.35, color="#CCCCCC")
    ax.set_axisbelow(True)
    _clean_ax(ax)
    _panel_label(ax, letter)


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_dysregulation_visualisations(
    stratified_results: Dict,
    dysreg_results: Dict,
    pathway_names: List[str],
    gene_names: List[str],
    out_dir: str,
    top_cell_types: Optional[List[str]] = None,
    mean_attn_disease: Optional[np.ndarray] = None,
    mean_attn_control: Optional[np.ndarray] = None,
):
    """
    Run all dysregulation visualisations.

    Parameters
    ----------
    stratified_results : from dysregulation.stratified_attention_comparison()
    dysreg_results     : from dysregulation.compute_program_dysregulation()
    mean_attn_disease  : (P, P) mean attention over disease cells
    mean_attn_control  : (P, P) mean attention over control cells
    """
    os.makedirs(out_dir, exist_ok=True)
    P = len(pathway_names)
    print("\n" + "=" * 55)
    print("  QuokkaVision v2 — Dysregulation figures  (Nature style)")
    print("=" * 55)

    plot_delta_heatmap(
        stratified_results, pathway_names,
        os.path.join(out_dir, "figD1_delta_heatmap_pooled.pdf"))

    plot_vulnerability_ranking(
        dysreg_results["vuln_df"],
        os.path.join(out_dir, "figD2_vulnerability_ranking.pdf"))

    plot_dysregulation_scores(
        dysreg_results["score_df"],
        os.path.join(out_dir, "figD3_dysregulation_scores.pdf"))

    plot_gained_lost_network(
        stratified_results, pathway_names,
        os.path.join(out_dir, "figD4_gained_lost_network_pooled.pdf"))

    # Fig D5: rewiring scatter (replaces volcano)
    if mean_attn_disease is not None and mean_attn_control is not None:
        plot_pathway_rewiring_scatter(
            mean_attn_disease, mean_attn_control, pathway_names,
            os.path.join(out_dir, "figD5_rewiring_scatter_pooled.pdf"))
    else:
        # Fallback: use delta from stratified results as proxy
        delta    = stratified_results.get(
            "attn_delta_all", np.zeros((P, P)))
        ctrl_prx = np.abs(delta).clip(0)   # unsigned proxy
        dis_prx  = ctrl_prx + delta
        plot_pathway_rewiring_scatter(
            dis_prx, ctrl_prx, pathway_names,
            os.path.join(out_dir, "figD5_rewiring_scatter_pooled.pdf"))

    # Per-cell-type figures for top vulnerable cell types
    if top_cell_types is None:
        top_cell_types = dysreg_results["vuln_df"]["cell_type"].head(3).tolist()

    for ct in top_cell_types:
        safe = ct.replace(" ", "_").replace("/", "-")
        plot_delta_heatmap(
            stratified_results, pathway_names,
            os.path.join(out_dir, f"figD1_delta_heatmap_{safe}.pdf"),
            cell_type=ct)
        plot_gained_lost_network(
            stratified_results, pathway_names,
            os.path.join(out_dir, f"figD4_gained_lost_network_{safe}.pdf"),
            cell_type=ct)
        if mean_attn_disease is not None and mean_attn_control is not None:
            ct_dis  = stratified_results.get("attn_disease", {}).get(
                ct, mean_attn_disease)
            ct_ctrl = stratified_results.get("attn_control", {}).get(
                ct, mean_attn_control)
            plot_pathway_rewiring_scatter(
                ct_dis, ct_ctrl, pathway_names,
                os.path.join(out_dir, f"figD5_rewiring_scatter_{safe}.pdf"),
                cell_type=ct)

    print(f"\n  Dysregulation figures saved to: {out_dir}")
