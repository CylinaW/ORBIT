"""
ORBIT — Data Pipeline
====================================
"""

import os
import glob
import shutil
import tempfile
import subprocess
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import torch
import anndata as ad
import gseapy as gp
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Reference: Morabito 2021 — CELLxGENE h5ad (cell_type in obs by schema)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Reference: Morabito 2021 GSE174367 -- two GEO files required
# ---------------------------------------------------------------------------
#
# File 1: GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5  (261 MB)
#   10x feature-barcode count matrix. Load with sc.read_10x_h5().
#
# File 2: GSE174367_snRNA-seq_cell_meta.csv.gz  (425 KB)
#   Per-barcode metadata. Row index = Barcode column.
#   Cell type column: 'cell_type' or 'CellType'.
#   Labels: EX, INH, ODC, ASC, MG, OPC, PER.END
#
# Upload BOTH files to: ./data/reference/
#
# Cell type abbreviations (Swarup lab, paper Fig 1c):
#   EX  = excitatory neurons  -> Exc
#   INH = inhibitory neurons  -> Inh
#   ODC = oligodendrocytes    -> Oli
#   ASC = astrocytes          -> Ast
#   OPC = OPC                 -> OPC
#   MG  = microglia           -> Mic
#   PER.END / PER/END         -> Vasc

REFERENCE_H5_PATH   = "./data/reference/GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5"
REFERENCE_META_PATH = "./data/reference/GSE174367_snRNA-seq_cell_meta.csv.gz"

MORABITO_CT_MAP = {
    "EX":       "Exc",
    "INH":      "Inh",
    "ODC":      "Oli",
    "ASC":      "Ast",
    "OPC":      "OPC",
    "MG":       "Mic",
    "PER.END":  "Vasc",
    "PER/END":  "Vasc",
    "END":      "Vasc",
    "PER":      "Vasc",
    "Excitatory":     "Exc",
    "Inhibitory":     "Inh",
    "Oligodendrocyte":"Oli",
    "Oligodendrocytes":"Oli",
    "Astrocyte":      "Ast",
    "Astrocytes":     "Ast",
    "Microglia":      "Mic",
    "Pericyte":       "Vasc",
    "Endothelial":    "Vasc",
}

CONSOLIDATION_MAP = {
    "Exc": "Exc", "Inh": "Inh", "Ast": "Ast",
    "Oli": "Oli", "OPC": "OPC", "Mic": "Mic", "Vasc": "Vasc",
}


def standard_qc_normalize(adata, label="data", min_genes=200, max_pct_mt=20):
    """
    Standard QC + normalisation pipeline:
      - Filter cells: min_genes detected, max mitochondrial %
      - Normalise: total-count normalise to 10,000 counts/cell
      - Log-transform: log1p
    Operates in-place and returns adata.
    """
    import re
    # Mitochondrial genes
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    before = adata.n_obs
    adata = adata[
        (adata.obs["n_genes_by_counts"] >= min_genes) &
        (adata.obs["pct_counts_mt"] <= max_pct_mt)
    ].copy()
    dropped = before - adata.n_obs
    if dropped > 0:
        print(f"    [{label}] QC: dropped {dropped:,} low-quality cells "
              f"({adata.n_obs:,} remain)")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def load_morabito2021_reference(
    h5_path: str = REFERENCE_H5_PATH,
    meta_path: str = REFERENCE_META_PATH,
    consolidate_vascular: bool = True,
) -> ad.AnnData:
    """
    Load Morabito et al. 2021 PFC snRNA-seq reference from GEO raw files.

    Requires two files from GSE174367 supplementary data:
      h5_path   -- GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5 (261 MB)
      meta_path -- GSE174367_snRNA-seq_cell_meta.csv.gz              (425 KB)

    Both must be uploaded to ./data/reference/.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Expression matrix not found: {h5_path}\n"
            "Upload GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5 "
            "to: ./data/reference/"
        )
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Cell metadata not found: {meta_path}\n"
            "Upload GSE174367_snRNA-seq_cell_meta.csv.gz "
            "to: ./data/reference/"
        )

    sz_mb = os.path.getsize(h5_path) / 1e6
    print(f"  Loading Morabito 2021 reference ({os.path.basename(h5_path)}, {sz_mb:.0f} MB)...")
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    print(f"    Raw: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    print(f"  Loading cell metadata ({os.path.basename(meta_path)})...")
    # Accept both .csv.gz and plain .csv (handles renamed uploads)
    _meta_path = meta_path
    if not os.path.exists(_meta_path):
        # Try plain .csv variant
        _meta_path_plain = meta_path.replace('.csv.gz', '.csv')
        if os.path.exists(_meta_path_plain):
            _meta_path = _meta_path_plain
            print(f'    Note: using plain CSV: {os.path.basename(_meta_path)}')
        else:
            raise FileNotFoundError(
                f'Cell metadata not found: {meta_path}\n'
                f'Also tried: {_meta_path_plain}\n'
                'Upload GSE174367_snRNA-seq_cell_meta.csv.gz (or .csv) '
                'to: ./data/reference/'
            )
    meta = pd.read_csv(_meta_path, index_col=0)
    print(f"    Meta: {len(meta):,} rows, columns: {list(meta.columns)}")

    # Normalise barcode index (strip 10x -1 suffix if present)
    def _strip_suffix(idx):
        import re
        return pd.Index([re.sub(r'-\d+$', '', b) for b in idx.astype(str)])

    adata_bc = _strip_suffix(adata.obs_names)
    meta_bc  = _strip_suffix(meta.index)
    bc_to_obs  = dict(zip(adata_bc, adata.obs_names))
    bc_to_meta = dict(zip(meta_bc,  meta.index))

    # Find cell type column
    ct_col = None
    for candidate in ["Cell.Type", "cell_type", "CellType", "celltype",
                      "Cell_Type", "annotation", "cluster_label", "Annotation"]:
        if candidate in meta.columns:
            ct_col = candidate
            break
    if ct_col is None:
        for col in meta.columns:
            if 4 <= meta[col].nunique() <= 12:  # SampleID has 18 -> excluded
                ct_col = col
                print(f"    Auto-detected cell type column: '{ct_col}'")
                break
    if ct_col is None:
        raise ValueError(
            f"Cannot find cell type column.\n"
            f"Columns: {list(meta.columns)}"
        )
    print(f"    Cell type column: '{ct_col}'")
    print(f"    Labels: {sorted(meta[ct_col].dropna().unique().tolist())}")

    # Join metadata
    common_bc = list(set(adata_bc) & set(meta_bc))
    obs_idx   = [bc_to_obs[bc]  for bc in common_bc]
    meta_idx  = [bc_to_meta[bc] for bc in common_bc]
    adata_sub = adata[obs_idx].copy()
    adata_sub.obs['raw_cell_type'] = meta.loc[meta_idx, ct_col].values
    # Also carry Diagnosis (AD / Control) as condition label if present
    if 'Diagnosis' in meta.columns:
        adata_sub.obs['diagnosis'] = meta.loc[meta_idx, 'Diagnosis'].values
    if 'SampleID' in meta.columns:
        adata_sub.obs['donor_id'] = meta.loc[meta_idx, 'SampleID'].values
    matched_n = adata_sub.obs['raw_cell_type'].notna().sum()
    print(f"    Barcode match: {matched_n:,} / {adata.n_obs:,} cells ({matched_n/adata.n_obs*100:.1f}%)")

    if matched_n < adata.n_obs * 0.5:
        raise ValueError(
            f"Only {matched_n}/{adata.n_obs} barcodes matched.\n"
            "Ensure both files are from the same GEO submission (GSE174367)."
        )
    adata = adata_sub

    # Map to major classes
    adata.obs['cell_type'] = adata.obs['raw_cell_type'].astype(str).map(MORABITO_CT_MAP)
    if consolidate_vascular:
        adata.obs['cell_type'] = adata.obs['cell_type'].map(
            lambda x: CONSOLIDATION_MAP.get(x, x) if pd.notna(x) else np.nan
        )

    before = adata.n_obs
    adata  = adata[adata.obs['cell_type'].notna() & (adata.obs['cell_type'] != 'nan')].copy()
    dropped = before - adata.n_obs
    if dropped > 0:
        print(f"    Dropped {dropped:,} cells (unmatched or unrecognised labels)")

    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
    counts = adata.obs['cell_type'].value_counts()
    print(f"    Final: {adata.n_obs:,} cells, {adata.obs['cell_type'].nunique()} types:")
    for ct, n in sorted(counts.items()):
        print(f"      {ct:<12} {n:>8,} cells")

    adata.obs['dataset']   = 'reference'
    adata.obs['condition'] = 'mixed'
    return adata


def download_reference_if_needed(
    local_path: str = REFERENCE_H5_PATH,
    url: str = '',
) -> str:
    """
    For the GEO-based workflow the reference is uploaded locally.
    This function just validates both required files are present.
    """
    meta_path = local_path.replace(
        'filtered_feature_bc_matrix.h5',
        'cell_meta.csv.gz'
    )
    missing = []
    if not os.path.exists(local_path): missing.append(local_path)
    if not os.path.exists(meta_path):  missing.append(meta_path)
    if missing:
        raise FileNotFoundError(
            'Reference file(s) not found:\n' +
            '\n'.join(f'  {p}' for p in missing) + '\n'
            'Upload both files to ./data/reference/'
        )
    sz = os.path.getsize(local_path) / 1e6
    print(f"Reference found: {os.path.basename(local_path)} ({sz:.0f} MB)")
    return local_path

# ---------------------------------------------------------------------------
# Donor manifests (from series matrix files)
# ---------------------------------------------------------------------------

# GSE157827 (Lau 2020, PNAS) — PFC snRNA-seq, 12 AD + 9 NC donors
GSE157827_DONORS = {
    "AD": {
        "AD1":  "GSM4775561",
        "AD2":  "GSM4775562",
        "AD4":  "GSM4775563",
        "AD5":  "GSM4775564",
        "AD6":  "GSM4775565",
        "AD8":  "GSM4775566",
        "AD9":  "GSM4775567",
        "AD10": "GSM4775568",
        "AD13": "GSM4775569",
        "AD19": "GSM4775570",
        "AD20": "GSM4775571",
        "AD21": "GSM4775572",
    },
    "NC": {
        "NC3":  "GSM4775573",
        "NC7":  "GSM4775574",
        "NC11": "GSM4775575",
        "NC12": "GSM4775576",
        "NC14": "GSM4775577",
        "NC15": "GSM4775578",
        "NC16": "GSM4775579",
        "NC17": "GSM4775580",
        "NC18": "GSM4775581",
    },
}

# GSE189819 (Bhatt 2022) — PFC BA9, healthy control neurons (.h5 format)
GSE189819_CONTROL_FILES = [
    ("GSM6261344", "Control-1",
     "GSM6261344_Control-1-MAP2_filtered_feature_bc_matrix.h5"),
    ("GSM6261345", "Control-2",
     "GSM6261345_Control-2-MAP2_filtered_feature_bc_matrix.h5"),
    ("GSM6261346", "Control-3",
     "GSM6261346_Control-3-MAP2_filtered_feature_bc_matrix.h5"),
    ("GSM6261347", "Control-4",
     "GSM6261347_Control-4-MAP2_filtered_feature_bc_matrix.h5"),
    ("GSM6261348", "Control-5",
     "GSM6261348_Control-5-MAP2_filtered_feature_bc_matrix.h5"),
    ("GSM6261349", "Control-6",
     "GSM6261349_Control-6-MAP2_filtered_feature_bc_matrix.h5"),
    ("GSM6261350", "Control-7",
     "GSM6261350_Control-7-MAP2_filtered_feature_bc_matrix.h5"),
    ("GSM6261351", "Control-8",
     "GSM6261351_Control-8-MAP2_filtered_feature_bc_matrix.h5"),
]


def load_gse157827_donor(folder: str, donor_name: str, gsm_id: str) -> ad.AnnData:
    prefix = f"{gsm_id}_{donor_name}_"
    src_files = {
        "barcodes.tsv.gz": os.path.join(folder, f"{prefix}barcodes.tsv.gz"),
        "features.tsv.gz": os.path.join(folder, f"{prefix}features.tsv.gz"),
        "matrix.mtx.gz":   os.path.join(folder, f"{prefix}matrix.mtx.gz"),
    }
    for dst, src in src_files.items():
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Missing: {src}\n"
                f"All three files ({prefix}barcodes/features/matrix) must be in {folder}"
            )

    tmpdir = tempfile.mkdtemp(prefix=f"qv_{donor_name}_")
    try:
        for dst_name, src_path in src_files.items():
            dst_path = os.path.join(tmpdir, dst_name)
            try:
                os.symlink(os.path.abspath(src_path), dst_path)
            except (OSError, NotImplementedError):
                shutil.copy2(src_path, dst_path)
        adata = sc.read_10x_mtx(tmpdir, var_names="gene_symbols", make_unique=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    adata.obs["donor_id"]    = donor_name
    adata.obs["gsm_id"]      = gsm_id
    adata.obs["source_file"] = prefix
    return adata


def load_gse157827_ad_donors(
    folder: str,
    donors: Optional[List[str]] = None,
) -> ad.AnnData:
    """
    Load AD donors from GSE157827.

    Parameters
    ----------
    folder : str
        Path containing flat GSE157827 files.
        e.g. ./data/disease/
    donors : list of donor IDs or None for all 12.
        e.g. ["AD1", "AD2", "AD4"]
    """
    adatas, ids = [], []
    for donor_name, gsm_id in GSE157827_DONORS["AD"].items():
        if donors is not None and donor_name not in donors:
            continue
        print(f"    Loading {donor_name} ({gsm_id})...", end=" ", flush=True)
        try:
            a = load_gse157827_donor(folder, donor_name, gsm_id)
            a.obs["condition"] = "AD"
            a.obs["dataset"]   = "disease"
            adatas.append(a)
            ids.append(donor_name)
            print(f"{a.n_obs} cells")
        except FileNotFoundError as e:
            print("SKIPPED")
            print(f"      -> {e}")

    if not adatas:
        raise ValueError(
            f"No GSE157827 AD files loaded from: {folder}\n"
            "Check that all flat barcodes/features/matrix files are in this folder."
        )

    combined = ad.concat(adatas, join="outer", label="donor",
                         keys=ids, index_unique="-")
    combined.obs_names_make_unique()
    print(f"  GSE157827 AD: {combined.n_obs} cells from {len(adatas)} donors")
    return combined


# ---------------------------------------------------------------------------
# Control loader: GSE189819 .h5 files
# ---------------------------------------------------------------------------

GSE189819_CONTROL_FILES = [
    ("GSM6261344", "Control-1",
     "GSM6261344_Control-1-MAP2_filtered_feature_bc_matrix.h5"),
    ("GSM6261345", "Control-2",
     "GSM6261345_Control-2-MAP2_filtered_feature_bc_matrix.h5"),
    ("GSM6261346", "Control-3",
     "GSM6261346_Control-3-MAP2_filtered_feature_bc_matrix.h5"),
    ("GSM6261347", "Control-4",
     "GSM6261347_Control-4-MAP2_filtered_feature_bc_matrix.h5"),
    ("GSM6261348", "Control-5",
     "GSM6261348_Control-5-MAP2_filtered_feature_bc_matrix.h5"),
    ("GSM6261349", "Control-6",
     "GSM6261349_Control-6-MAP2_filtered_feature_bc_matrix.h5"),
    ("GSM6261350", "Control-7",
     "GSM6261350_Control-7-MAP2_filtered_feature_bc_matrix.h5"),
    ("GSM6261351", "Control-8",
     "GSM6261351_Control-8-MAP2_filtered_feature_bc_matrix.h5"),
]


def load_gse189819_controls(folder: str) -> ad.AnnData:
    """
    Load 8 GSE189819 healthy control donors from .h5 files.

    Parameters
    ----------
    folder : str
        e.g. ./data/control/
    """
    adatas, ids = [], []
    for gsm_id, donor_name, filename in GSE189819_CONTROL_FILES:
        fpath = os.path.join(folder, filename)
        if not os.path.exists(fpath):
            print(f"    {donor_name}: NOT FOUND at {fpath} — skipping")
            continue
        print(f"    Loading {donor_name} ({gsm_id})...", end=" ", flush=True)
        a = sc.read_10x_h5(fpath)
        a.var_names_make_unique()
        a.obs["donor_id"]    = donor_name
        a.obs["gsm_id"]      = gsm_id
        a.obs["source_file"] = filename
        a.obs["condition"]   = "Control"
        a.obs["dataset"]     = "control"
        adatas.append(a)
        ids.append(donor_name)
        print(f"{a.n_obs} cells")

    if not adatas:
        raise ValueError(
            f"No GSE189819 control files found in: {folder}\n"
            "Upload the 8 Control-MAP2 .h5 files (GSM6261344-6261351)."
        )

    combined = ad.concat(adatas, join="outer", label="donor",
                         keys=ids, index_unique="-")
    combined.obs_names_make_unique()
    print(f"  GSE189819 controls: {combined.n_obs} cells from {len(adatas)} donors")
    return combined


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

class ADDataPipeline:
    """
    AD data pipeline: reference + disease + control.

    Parameters
    ----------
    reference_h5_path : str
        Path to Morabito 2021 h5ad (downloaded by notebook Cell 3).
        Default: /content/morabito2021_PFC_snRNA.h5ad
    disease_folder : str
        Folder with flat GSE157827 AD donor files.
        Upload to: ./data/disease/
    control_folder : str
        Folder with 8 GSE189819 Control-MAP2 .h5 files.
        Upload to: ./data/control/
    consolidate_vascular : bool
        True = 7 classes (recommended).
    disease_donors : list or None
        Subset of AD donor IDs. None = all 12.
    query_path : str or None
        Optional unannotated cells for prediction.
    """

    def __init__(
        self,
        reference_h5_path: str = REFERENCE_H5_PATH,
        reference_meta_path: str = REFERENCE_META_PATH,
        disease_folder: str = "./data/disease",
        control_folder: str = "./data/control",
        consolidate_vascular: bool = True,
        disease_donors: Optional[List[str]] = None,
        query_path: Optional[str] = None,
    ):
        self.reference_h5_path    = reference_h5_path
        self.reference_meta_path  = reference_meta_path
        self.disease_folder       = disease_folder
        self.control_folder       = control_folder
        self.consolidate_vascular = consolidate_vascular
        self.disease_donors       = disease_donors
        self.query_path           = query_path

    def load_and_process(self) -> ad.AnnData:
        """
        Load all datasets, align genes, apply QC + normalisation.

        Returns AnnData with obs:
          dataset   : reference | disease | control | query
          condition : mixed | AD | Control | unknown
          cell_type : annotated for reference; Unknown elsewhere
          donor_id  : per-donor identifier
        """
        print("\n[Pipeline] Loading all datasets...")

        print("\n  [1/3] Reference — Morabito 2021 PFC snRNA-seq (GSE174367)...")
        ref = load_morabito2021_reference(
            h5_path=self.reference_h5_path,
            meta_path=self.reference_meta_path,
            consolidate_vascular=self.consolidate_vascular,
        )

        print("\n  [2/3] Disease — GSE157827 AD donors...")
        dis = load_gse157827_ad_donors(
            self.disease_folder,
            donors=self.disease_donors,
        )

        print("\n  [3/3] Controls — GSE189819 healthy donors...")
        ctrl = load_gse189819_controls(self.control_folder)

        parts = [ref, dis, ctrl]
        if self.query_path is not None:
            print("\n  [opt] Query...")
            qry = _load_any(self.query_path)
            qry.obs["dataset"]   = "query"
            qry.obs["condition"] = "unknown"
            parts.append(qry)

        # Align to common genes
        common = parts[0].var_names
        for a in parts[1:]:
            common = common.intersection(a.var_names)
        common = common.sort_values()
        print(f"\n  Common genes: {len(common):,}")
        if len(common) < 1000:
            raise ValueError(
                f"Only {len(common)} common genes. "
                "All datasets must use hg38 HGNC gene symbols."
            )

        aligned = [a[:, common].copy() for a in parts]
        adata = ad.concat(aligned, join="outer", merge="first")
        adata.obs_names_make_unique()
        adata.var = aligned[0].var.loc[common].copy()

        print(f"  Combined: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        print(f"  {adata.obs['dataset'].value_counts().to_dict()}")

        print("\n  Running QC + normalisation...")
        adata = standard_qc_normalize(adata, label="combined")

        ref_mask = adata.obs["dataset"] == "reference"
        ref_cats = sorted(
            adata.obs.loc[ref_mask, "cell_type"].dropna().unique().tolist()
        )
        adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
        adata.obs.loc[~ref_mask, "cell_type"] = "Unknown"
        adata.obs["cell_type"] = pd.Categorical(
            adata.obs["cell_type"], categories=ref_cats + ["Unknown"]
        )
        print(f"  Reference cell types: {ref_cats}")
        print(f"\n  Final: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        return adata


def _load_any(path: str) -> ad.AnnData:
    if path.endswith(".h5ad"):
        return sc.read_h5ad(path)
    elif path.endswith(".h5"):
        a = sc.read_10x_h5(path)
        a.var_names_make_unique()
        return a
    elif os.path.isdir(path):
        return sc.read_10x_mtx(path, var_names="gene_symbols", make_unique=True)
    else:
        raise ValueError(f"Unrecognised format: {path}")


# ---------------------------------------------------------------------------
# Allen Brain Atlas pathway mask
# ---------------------------------------------------------------------------

class PathwayAnalyser:
    def __init__(self, min_genes: int = 5):
        self.min_genes = min_genes
        self._db = None

    def _load_db(self):
        if self._db is None:
            cache = "/content/allen_brain_atlas_cache.json"
            if os.path.exists(cache):
                import json
                with open(cache) as f:
                    self._db = json.load(f)
                print(f"  Allen Brain Atlas from cache ({len(self._db)} programs)")
            else:
                print("  Downloading Allen Brain Atlas...")
                self._db = gp.get_library(name="Allen_Brain_Atlas_10x_scRNA_2021")
                import json
                with open(cache, "w") as f:
                    json.dump(self._db, f)
                print(f"  Saved to {cache}")
        return self._db

    def create_pathway_mask(
        self, adata, human_only: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Build binary pathway mask.

        human_only : bool (default True)
            If True, exclude programs whose names start with 'Mouse '
            (Allen Brain Atlas contains both human and mouse marker sets).
            Human programs are named e.g. 'Human Exc L2-3 LINC00507 FREM3 up'.
            Mouse programs are named e.g. 'Mouse 20 Ndnf HPF down'.
            Only human programs are biologically relevant for AD analysis
            of human post-mortem tissue.
        """
        db = self._load_db()

        if human_only:
            db = {k: v for k, v in db.items()
                  if not k.startswith("Mouse ")}
            print(f"  Allen Brain Atlas: {len(db)} human programs "
                  "(mouse programs excluded)")
        else:
            print(f"  Allen Brain Atlas: {len(db)} programs (human + mouse)")

        gene_index = pd.Index(adata.var_names.str.upper())
        prog_df = pd.DataFrame(0, index=gene_index, columns=list(db.keys()))
        for prog, genes in db.items():
            prog_df[prog] = gene_index.isin(
                [g.upper() for g in genes]
            ).astype(int)

        valid   = prog_df.columns[prog_df.sum() >= self.min_genes]
        prog_df = prog_df[valid]
        print(f"  Retained {len(valid)} programs (>= {self.min_genes} genes)")
        return torch.tensor(prog_df.values, dtype=torch.float32), valid.tolist()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_cells(
    model, adata: ad.AnnData, valid_cats: List[str],
    target_datasets: List[str] = None,
    confidence_threshold: float = 0.7,
    batch_size: int = 512, device=None
) -> ad.AnnData:
    """Predict cell types for non-reference cells."""
    if device is None:
        device = next(model.parameters()).device
    if target_datasets is None:
        target_datasets = ["disease", "control", "query"]
    model.eval()
    mask = adata.obs["dataset"].isin(target_datasets)
    if mask.sum() == 0:
        return adata

    X = (adata[mask].X.toarray() if scipy.sparse.issparse(adata[mask].X)
         else adata[mask].X.copy())
    X = np.nan_to_num(X).astype(np.float32)
    Xt = torch.tensor(X, dtype=torch.float32)

    probs_list = []
    for s in range(0, len(Xt), batch_size):
        logits, _ = model(Xt[s:s+batch_size].to(device))
        probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.vstack(probs_list)
    preds, confs = np.argmax(probs, 1), np.max(probs, 1)

    tidx = adata.obs_names[mask]
    adata.obs.loc[tidx, "predicted"] = [
        valid_cats[p] if c >= confidence_threshold else "Unknown"
        for p, c in zip(preds, confs)
    ]
    adata.obs.loc[tidx, "confidence"] = confs

    ridx = adata.obs_names[adata.obs["dataset"] == "reference"]
    adata.obs.loc[ridx, "predicted"] = adata.obs.loc[ridx, "cell_type"].astype(str)

    adata.obs["combined_cell_type"] = pd.Categorical(
        np.where(
            adata.obs["dataset"] == "reference",
            adata.obs["cell_type"].astype(str),
            adata.obs.get("predicted",
                pd.Series("Unknown", index=adata.obs_names)).astype(str),
        ),
        categories=valid_cats + ["Unknown"],
    )
    n = int(mask.sum())
    n_unk = int((adata.obs.loc[tidx, "predicted"] == "Unknown").sum())
    print(f"  Predicted {n} cells: {n - n_unk} assigned, {n_unk} Unknown")
    return adata


@torch.no_grad()
def generate_embeddings(
    model, adata: ad.AnnData, device=None, batch_size: int = 256
) -> ad.AnnData:
    """Store 640-dim embeddings in adata.obsm[X_ORBIT]."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    X = (adata.X.toarray() if scipy.sparse.issparse(adata.X)
         else adata.X.copy())
    X = np.nan_to_num(X).astype(np.float32)
    Xt = torch.tensor(X, dtype=torch.float32)
    embs = [model.get_embeddings(Xt[s:s+batch_size].to(device))
            for s in range(0, len(Xt), batch_size)]
    adata.obsm["X_ORBIT"] = np.vstack(embs)
    print(f"  Embeddings stored: {adata.obsm['X_ORBIT'].shape}")
    return adata
