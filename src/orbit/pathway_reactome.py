"""
ORBIT Reactome Pathway Mask Builder
=====================================================================
Builds the binary (G × P) pathway mask from Reactome gene sets.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("pathway_reactome")

# ── Enrichr endpoint ────────────────────────────────────────────────────────
_ENRICHR_GMT_URL = (
    "https://maayanlab.cloud/Enrichr/geneSetLibrary"
    "?mode=text&libraryName={library}"
)

_REACTOME_LIBRARY_NAMES = [
    "Reactome_2022",
    "Reactome_2016",
    "Reactome_2015",
]

_DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/orbit")


# ══════════════════════════════════════════════════════════════════════════════
# CORE DOWNLOAD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def _download_enrichr_gmt(
    library_name: str,
    timeout: int = 60,
) -> Dict[str, List[str]]:
    """
    Download a gene-set library from the Enrichr GMT endpoint.

    Returns {pathway_name: [gene1, gene2, ...]} dict.
    Bypasses gseapy — stdlib only. Explicit bytes decode avoids the bug.
    """
    url = _ENRICHR_GMT_URL.format(library=library_name)
    print(f"    Fetching: {url[:80]}...")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ORBIT/2.0 (pathway_reactome.py; direct download)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_bytes = response.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"HTTP {e.code} fetching {library_name} from Enrichr: {e.reason}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Network error fetching {library_name}: {e.reason}\n"
            f"Supply a local GMT cache at: {_DEFAULT_CACHE_DIR}/reactome_human.gmt"
        ) from e

    text = raw_bytes.decode("utf-8", errors="replace")

    gene_sets: Dict[str, List[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        pathway_name = parts[0].strip().lower()   # Reactome names → lowercase for consistency
        genes = [g.split(",")[0].strip().upper() for g in parts[2:] if g.strip()]
        if genes:
            gene_sets[pathway_name] = genes

    if not gene_sets:
        raise RuntimeError(
            f"Downloaded {library_name} but parsed 0 gene sets.\n"
            f"First 500 bytes of response:\n{text[:500]}"
        )

    print(f"    ✓ Parsed {len(gene_sets)} pathways from {library_name}")
    return gene_sets


def _parse_gmt_file(gmt_path: str) -> Dict[str, List[str]]:
    """Parse a local .gmt file into {pathway_name: [genes]} dict."""
    gene_sets: Dict[str, List[str]] = {}
    with open(gmt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip().lower()
            genes = [g.split(",")[0].strip().upper() for g in parts[2:] if g.strip()]
            if genes:
                gene_sets[name] = genes
    return gene_sets


# ══════════════════════════════════════════════════════════════════════════════
# ReactomePathwayAnalyser
# ══════════════════════════════════════════════════════════════════════════════

class ReactomePathwayAnalyser:
    """
    Build a binary (G × P) pathway mask from Reactome Human gene sets.

    Parameters
    ----------
    min_genes     : minimum genes per pathway to include (default 5)
    max_genes     : maximum genes per pathway to include (default 500)
    organism      : "Human" (default) — Reactome is human-only via Enrichr
    cache_dir     : directory for caching downloaded gene sets
    """

    def __init__(
        self,
        min_genes:  int = 5,
        max_genes:  int = 500,
        organism:   str = "Human",
        cache_dir:  str = _DEFAULT_CACHE_DIR,
    ) -> None:
        self.min_genes = min_genes
        self.max_genes = max_genes
        self.organism  = organism
        self.cache_dir = cache_dir
        self._db: Optional[Dict[str, List[str]]] = None

    # ── Internal: load / cache ─────────────────────────────────────────────

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, "reactome_human.json")

    def _gmt_fallback_path(self) -> str:
        return os.path.join(self.cache_dir, "reactome_human.gmt")

    def _load_db(self) -> Dict[str, List[str]]:
        """
        Load Reactome gene sets via fallback chain:
          1. In-memory cache (self._db)
          2. Local JSON cache file
          3. Direct Enrichr GMT download (no gseapy)
          4. Local GMT file (user-supplied)
        """
        if self._db is not None:
            return self._db

        os.makedirs(self.cache_dir, exist_ok=True)
        cache = self._cache_path()

        # ── Step 1: JSON cache ─────────────────────────────────────────
        if os.path.exists(cache):
            print(f"  Loading Reactome from cache: {cache}")
            with open(cache, "r") as f:
                self._db = json.load(f)
            print(f"  Loaded {len(self._db)} Reactome pathways from cache")
            return self._db

        # ── Step 2: Direct Enrichr download ───────────────────────────
        print(f"  Building Reactome pathway mask...")
        print(f"  Downloading Reactome Human pathways (direct Enrichr download)...")

        last_error: Optional[Exception] = None
        for lib_name in _REACTOME_LIBRARY_NAMES:
            try:
                self._db = _download_enrichr_gmt(lib_name, timeout=60)
                with open(cache, "w") as f:
                    json.dump(self._db, f)
                print(f"  Cached to: {cache}")
                return self._db
            except RuntimeError as e:
                print(f"    ✗ {lib_name} failed: {str(e)[:120]}")
                last_error = e

        # ── Step 3: Local GMT fallback ─────────────────────────────────
        gmt_path = self._gmt_fallback_path()
        if os.path.exists(gmt_path):
            print(f"  Loading Reactome from local GMT: {gmt_path}")
            self._db = _parse_gmt_file(gmt_path)
            with open(cache, "w") as f:
                json.dump(self._db, f)
            return self._db

        # ── Step 4: Failure with instructions ─────────────────────────
        raise RuntimeError(
            f"Failed to load Reactome pathway data.\n\n"
            f"Last error: {last_error}\n\n"
            f"Fix options (choose one):\n"
            f"  A) Check internet connection — Enrichr may be temporarily down\n"
            f"  B) Download Reactome GMT manually from:\n"
            f"       https://maayanlab.cloud/Enrichr/#libraries\n"
            f"     Look for 'Reactome_2022', download, save as:\n"
            f"       {gmt_path}\n"
            f"  C) Try upgrading gseapy (may not fix all environments):\n"
            f"       pip install 'gseapy>=1.1.0' --upgrade\n"
            f"     Then clear cache: rm -f {self._cache_path()}"
        ) from last_error

    # ── Public API ─────────────────────────────────────────────────────────

    def create_pathway_mask(
        self,
        adata,
        normalize: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build binary (G × P) pathway membership mask for Reactome.

        Parameters
        ----------
        adata     : AnnData — gene names from adata.var_names (HGNC symbols)
        normalize : if True, divide each column by its L2 norm (default False)

        Returns
        -------
        mask          : (G, P) float32 ndarray — binary pathway membership
        pathway_names : list of P pathway name strings (lowercase, with R-HSA ID)
        """
        db = self._load_db()

        gene_index = pd.Index(adata.var_names.str.upper())
        G = len(gene_index)

        filtered_pathways: Dict[str, List[int]] = {}
        for pathway_name, genes in db.items():
            gene_idx = gene_index.get_indexer(genes)   # genes already uppercased in _load_db
            valid_idx = gene_idx[gene_idx >= 0].tolist()
            if self.min_genes <= len(valid_idx) <= self.max_genes:
                filtered_pathways[pathway_name] = valid_idx

        if not filtered_pathways:
            raise RuntimeError(
                f"No Reactome pathways survived filtering "
                f"(min_genes={self.min_genes}, max_genes={self.max_genes}).\n"
                f"Check that adata.var_names contains HGNC gene symbols.\n"
                f"Sample adata.var_names: {list(adata.var_names[:10])}"
            )

        pathway_names = sorted(filtered_pathways.keys())
        P = len(pathway_names)
        mask = np.zeros((G, P), dtype=np.float32)

        for p_idx, pname in enumerate(pathway_names):
            for g_idx in filtered_pathways[pname]:
                mask[g_idx, p_idx] = 1.0

        if normalize:
            col_norms = np.linalg.norm(mask, axis=0, keepdims=True) + 1e-8
            mask = mask / col_norms

        print(f"  Reactome mask: {G:,} genes × {P} pathways "
              f"(filtered from {len(db)} total, "
              f"min={self.min_genes}, max={self.max_genes})")
        print(f"  Genes covered: {int(mask.sum(axis=1).astype(bool).sum()):,} / {G:,}")
        print(f"  Mean genes/pathway: {mask.sum(axis=0).mean():.1f}")

        # ORBIT.register_buffer requires torch.Tensor, not np.ndarray.
        try:
            import torch
            mask = torch.from_numpy(mask)
        except ImportError:
            pass

        return mask, pathway_names

    def get_pathway_gene_dict(self, adata) -> Dict[str, set]:
        """
        Returns {pathway_name: set(gene_symbols)} for all filtered pathways.
        Useful for biological_validation.py.
        """
        _, pnames = self.create_pathway_mask(adata)
        db = self._load_db()
        gene_set_upper = set(g.upper() for g in adata.var_names)
        return {
            pname: {g for g in db.get(pname, []) if g in gene_set_upper}
            for pname in pnames
        }

    def clear_cache(self) -> None:
        """Delete local cache to force re-download on next call."""
        cache = self._cache_path()
        if os.path.exists(cache):
            os.remove(cache)
            print(f"  Cache cleared: {cache}")
        else:
            print(f"  No cache found at: {cache}")

    def get_reactome_id(self, pathway_name: str) -> Optional[str]:
        """
        Extract the R-HSA numeric ID from a Reactome pathway name string.
        e.g. "translocation of zap-70 ... r-hsa-202430" → "R-HSA-202430"
        Returns None if no R-HSA ID found.
        """
        match = re.search(r"r-hsa-(\d+)", pathway_name, re.IGNORECASE)
        if match:
            return f"R-HSA-{match.group(1)}"
        return None
