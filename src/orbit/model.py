"""
ORBIT — Model Architecture
=============================================

Two-stage design:
─────────────────
Stage 1 (self-supervised, no labels):
  Masked Pathway Expression Pre-training (MPEP).
  Trains the pathway encoder + transformer via masked reconstruction.

Stage 2 (end-to-end, lightweight classifier):
  A two-layer linear head (D→D→C) is attached after Stage 1 finishes.
  Purpose: provide a biological anchor for interpretation.
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pathway encoder
# ---------------------------------------------------------------------------

class PathwayEncoder(nn.Module):
    """(B, P) → (B, P, D): scalar expression + pathway identity embedding."""
    def __init__(self, num_pathways: int, embed_dim: int = 128,
                 identity_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        backbone_dim = embed_dim - identity_dim
        self.backbone = nn.Sequential(
            nn.Linear(1, backbone_dim),
            nn.LayerNorm(backbone_dim),
            nn.GELU(),
        )
        self.identity_embeddings = nn.Embedding(num_pathways, identity_dim)
        nn.init.normal_(self.identity_embeddings.weight, std=0.02)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.register_buffer("pathway_indices",
                             torch.arange(num_pathways, dtype=torch.long))

    def forward(self, pathway_scores: torch.Tensor) -> torch.Tensor:
        B = pathway_scores.shape[0]
        x = self.backbone(pathway_scores.unsqueeze(-1))
        identity = self.identity_embeddings(self.pathway_indices)
        x = torch.cat([x, identity.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class PathwayTransformerBlock(nn.Module):
    """Pre-norm transformer block (LayerNorm before attention)."""
    def __init__(self, embed_dim: int, num_heads: int,
                 ff_dim: int, attn_dropout: float = 0.1,
                 ff_dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                            dropout=attn_dropout,
                                            batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(ff_dropout),
        )

    def forward(self, x: torch.Tensor,
                return_attn: bool = False) -> Tuple:
        normed   = self.norm1(x)
        attn_out, attn_w = self.attn(normed, normed, normed,
                                      need_weights=True,
                                      average_attn_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return (x, attn_w) if return_attn else x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ORBIT(nn.Module):
    def __init__(
        self,
        num_genes:        int,
        pathway_mask:     torch.Tensor,
        pathway_names:    List[str],
        embed_dim:        int   = 128,
        num_heads:        int   = 8,
        depth:            int   = 2,
        ff_multiplier:    int   = 4,
        attn_dropout:     float = 0.1,
        ff_dropout:       float = 0.1,
        encoder_dropout:  float = 0.3,
        num_classes:      Optional[int]   = None,   # set None for Stage 1
        cell_type_names:  Optional[List[str]] = None,
    ):
        super().__init__()
        assert pathway_mask.shape[0] == num_genes
        assert embed_dim % num_heads == 0

        self.num_genes     = num_genes
        self.num_pathways  = pathway_mask.shape[1]
        self.embed_dim     = embed_dim
        self.num_heads     = num_heads
        self.depth         = depth
        self.pathway_names = pathway_names
        self.num_classes   = num_classes
        self.cell_type_names = cell_type_names

        self.register_buffer("pathway_mask", pathway_mask.float())
        pathway_sizes = pathway_mask.sum(dim=0).clamp(min=1.0)
        self.register_buffer("pathway_size_norm", 1.0 / pathway_sizes.sqrt())

        # ── Stage 1 components ────────────────────────────────────────
        self.pathway_encoder = PathwayEncoder(
            self.num_pathways, embed_dim, 32, encoder_dropout)

        ff_dim = embed_dim * ff_multiplier
        self.transformer_blocks = nn.ModuleList([
            PathwayTransformerBlock(embed_dim, num_heads, ff_dim,
                                    attn_dropout, ff_dropout)
            for _ in range(depth)
        ])
        self.final_norm    = nn.LayerNorm(embed_dim)
        self.pathway_decoder = nn.Linear(embed_dim, 1)  # reconstruction head
        # Intervention-effect prediction head (Stage 1 auxiliary loss)
        # Registered here so optimizer sees its params from the start.
        self.interv_token_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # ── Stage 2 classifier (optional) ────────────────────────────
        # Thin: D → D → C  (~16k params vs ~400k in the transformer)
        # Purpose: biological anchor, not benchmark accuracy.
        self.classifier = None
        if num_classes is not None:
            self._build_classifier(num_classes)

        self._init_weights()

    def _build_classifier(self, num_classes: int):
        """Build (or rebuild) the thin classification head."""
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, num_classes),
        )
        self.num_classes = num_classes
        # Initialise head weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def attach_classifier(self, num_classes: int,
                           cell_type_names: Optional[List[str]] = None):
        """
        Called after Stage 1 to add the classification head.
        Only the new head weights are randomly initialised;
        Stage 1 weights are untouched.
        """
        self._build_classifier(num_classes)
        self.cell_type_names = cell_type_names
        device = next(self.parameters()).device
        self.classifier = self.classifier.to(device)
        n_cls_params = sum(p.numel() for p in self.classifier.parameters())
        print(f"  Classifier attached: {num_classes} classes  "
              f"({n_cls_params:,} params)  "
              f"→ {cell_type_names}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not getattr(
                self, "classifier", None
            ):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Core pathway projection ────────────────────────────────────────

    def _project_to_pathways(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(x, self.pathway_mask)
        return scores * self.pathway_size_norm.unsqueeze(0)

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                return_attention: bool = False) -> Tuple:
        """
        Returns
        -------
        cell_emb  : (B, D)   mean-pooled cell embedding
        h         : (B, P, D) per-pathway embeddings (after final_norm)
        last_attn : (B, H, P, P)  — only if return_attention=True
        logits    : (B, C)   — only if classifier is attached
        """
        h = self.pathway_encoder(self._project_to_pathways(x))
        last_attn = None
        for i, block in enumerate(self.transformer_blocks):
            if i == len(self.transformer_blocks) - 1:
                h, last_attn = block(h, return_attn=True)
            else:
                h = block(h, return_attn=False)
        h        = self.final_norm(h)
        cell_emb = h.mean(dim=1)                              # (B, D)

        if self.classifier is not None:
            logits = self.classifier(cell_emb)               # (B, C)
            if return_attention:
                return cell_emb, h, last_attn, logits
            return cell_emb, h, logits

        if return_attention:
            return cell_emb, h, last_attn
        return cell_emb, h

    # ── Inference helpers ─────────────────────────────────────────────

    @torch.no_grad()
    def get_pathway_scores(self, x: torch.Tensor) -> np.ndarray:
        return self._project_to_pathways(x).cpu().numpy()

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> np.ndarray:
        out = self.forward(x)
        return out[0].cpu().numpy()   # cell_emb is always first

    @torch.no_grad()
    def predict(self, x: torch.Tensor,
                confidence_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (predicted_labels, confidence) arrays.
        Cells below confidence_threshold are labelled 'Unknown'.
        Only valid after attach_classifier() has been called.
        """
        assert self.classifier is not None, \
            "Call attach_classifier() before predict()"
        out     = self.forward(x)
        logits  = out[-1]                                    # last element
        probs   = torch.softmax(logits, dim=-1).cpu().numpy()
        conf    = probs.max(axis=1)
        idx     = probs.argmax(axis=1)
        names   = self.cell_type_names or [str(i) for i in range(self.num_classes)]
        labels  = np.array([names[i] if conf[j] >= confidence_threshold
                             else "Unknown"
                             for j, i in enumerate(idx)])
        return labels, conf

    @torch.no_grad()
    def compute_mean_attention(self, x: torch.Tensor,
                                batch_size: int = 256) -> np.ndarray:
        """Mean attention (P, P) over all cells in x."""
        device   = next(self.parameters()).device
        attn_sum = torch.zeros(self.num_pathways, self.num_pathways,
                               device=device)
        count = 0
        for s in range(0, x.shape[0], batch_size):
            xb  = x[s:s + batch_size].to(device)
            out = self.forward(xb, return_attention=True)
            # last_attn is at index 2 (with classifier) or 2 (without)
            last_attn = out[2]
            attn_sum += last_attn.mean(dim=(0, 1))
            count    += 1
        return (attn_sum / count).cpu().numpy()

    @torch.no_grad()
    def compute_mean_attention_by_celltype(
            self, adata, batch_size: int = 256) -> dict:
        """
        {cell_type: (P, P)} using obs['cell_type'] as stratification key.
        Labels used only as index filters — never as training targets.
        """
        device = next(self.parameters()).device
        result = {}
        cell_types = adata.obs["cell_type"].unique().tolist()
        for ct in cell_types:
            idx = (adata.obs["cell_type"] == ct).values
            if idx.sum() < 10:
                continue
            X  = adata[idx].X
            X  = X.toarray() if scipy.sparse.issparse(X) else X.copy()
            Xt = torch.tensor(np.nan_to_num(X).astype(np.float32))
            result[ct] = self.compute_mean_attention(Xt, batch_size)
        return result

    @torch.no_grad()
    def compute_gene_pathway_influence(self,
                                        mean_attn: np.ndarray) -> np.ndarray:
        """influence[G, P2] = Σ_P1  membership[G, P1] × attention[P1, P2]"""
        return (self.pathway_mask.cpu().numpy() @ mean_attn).astype("float32")
