# ORBIT: Learning Gene Program Co-Activation Structure for Cell-Type-Stratified Pathway Rewiring Analysis in Single-Cell Transcriptomics

This repository contains the official implementation of ORBIT, accompanying the NeurIPS 2026 submission of the same name. The code reproduces all experiments reported in the paper.

## Overview

ORBIT is a self-supervised transformer that learns asymmetric, directed dependencies among gene programs from observational single-cell RNA-sequencing data alone. Stage 1 trains the pathway attention transformer with a masked-reconstruction objective augmented by an intervention-consistent influence loss, which forces attention weights to predict score-space shifts under single-program input ablation. Stage 2 fine-tunes a thin classification head (3.7% of Stage 1 parameters) on cell-type labels to enable condition-stratified analysis. The learned P × P attention matrix serves as the primary interpretability object for downstream analyses of gene program co-activation structure. Architectural and methodological details are given in Section 2 and Appendices A–D of the paper.

## Installation

```bash
git clone <anonymous-repo-url>
cd ORBIT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All experiments in the paper were trained on a single NVIDIA A100 (40 GB). Approximate Stage 1 wall-clock times: ABA ≈ 1.2 h, KEGG ≈ 1.2 h, Reactome ≈ 1.5 h. Stage 2 adds ≈ 0.3–0.5 h per vocabulary.

## Data

The real-data experiments use publicly available human prefrontal cortex snRNA-seq datasets:

- Reference atlas: GSE174367 (Morabito et al., 2021), 191,890 nuclei across 7 cell types
- Held-out disease cohort: Lau et al. (2020)
- Held-out neuron-enriched control: Otero-Garcia et al. (2022)

Expected data layout under the path set by `BASE` in `scripts/main.py`:

```
data/
├── reference/
│   ├── GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5
│   └── GSE174367_snRNA-seq_cell_meta.csv.gz
├── disease/
│   └── [AD donor files]
└── control/
    └── [control donor .h5 files]
```

Pathway vocabularies are loaded automatically via `src/orbit/pathway_kegg.py` and `src/orbit/pathway_reactome.py`. The Allen Brain Atlas vocabulary is constructed from the reference atlas itself.

## Reproducing the experiments

The complete pipeline runs from a single entry point. Edit the `BASE` variable at the top of `scripts/main.py` to point to your data directory, then:

```bash
python scripts/main.py
```

This executes the full pipeline end-to-end: dataset loading, pathway-mask construction, Stage 1 self-supervised training, Stage 2 cell-type classification fine-tuning, evaluation, prediction on disease and control cohorts, mean attention computation, dysregulation analysis (AD vs. control), and figure generation. All hyperparameters are set in the `CONFIG` dictionary at the top of `main.py` and match the values in Appendix Tables 1–3 of the paper.

The synthetic identifiability benchmark (Section 3.2, Table 1 of the paper) is run separately:

```bash
python -m orbit.synthetic
```

YAML configurations are provided in configs/ for each vocabulary and the synthetic benchmark, serving both as reference and as an alternative file-based execution interface. The CONFIG dictionary in main.py defines the default runtime configuration, while the YAML files mirror these settings to ensure reproducibility and enable flexible execution across different experimental setups.

## Repository structure

```
src/orbit/
  model.py                    Pathway encoder, transformer, classification head
  training.py                 Stage 1 + Stage 2 training and intervention-consistent loss
  data_pipeline.py            snRNA-seq preprocessing and AnnData assembly
  pathway_kegg.py             KEGG vocabulary loader (cached GMT)
  pathway_reactome.py         Reactome vocabulary loader (cached GMT)
  benchmarking.py             Cross-method evaluation routines
  classification_benchmark.py CellTypist, scANVI, and MLP baselines
  biological_validation.py    AUCell agreement, cross-dataset replication
  dysregulation.py            AD-vs-control attention rewiring analysis
  dysregulation_viz.py        Rewiring figure generation
  synthetic.py                Synthetic generator and triplet evaluation
  visualisation.py            Attention heatmaps and gene-influence plots
  figure_data.py              Figure-ready table assembly
scripts/
  main.py                     End-to-end pipeline
configs/                      YAML configurations per vocabulary
tests/                        Unit and integration tests
```

## Hyperparameters

Stage 1 (Appendix Table 1): AdamW (β₁ = 0.9, β₂ = 0.999, weight decay 0.01), peak learning rate 1e-4 with cosine schedule and 10% linear warmup, gradient clipping at norm 1.0, batch size 128, 50 epochs, masking ratio 0.40, n_t = 4 intervention targets per step, loss weights 1.00 / 0.02 / 0.05 for the reconstruction, entropy, and intervention-consistent terms.

Stage 2 (Appendix Table 2): Phase 1 linear probe up to 50 epochs with early stopping (patience 5); Phase 2 end-to-end fine-tuning for 20 epochs at learning rate 5e-5 with cosine warmup; inverse-frequency class weighting; vascular cells (<1% of nuclei) oversampled and excluded from macro-F1.

Architecture (Appendix Table 3): pathway encoder 96 → 128 with 32-dim learnable program-identity embedding; two pre-norm transformer blocks with 8-head self-attention and FFN expansion factor 4; dropout 0.30 (encoder) / 0.10 (attention and FFN). Stage 1 ≈ 415,104 parameters; classification head ≈ 16,000 parameters (3.9% of Stage 1).

## Reproducibility

The default random seed is 42 throughout (dataset splitting, weight initialization, masking, intervention sampling). Hyperparameters were selected on a 10% validation split of the reference atlas and fixed before evaluation on the held-out 20% test set. Rewiring results reported as primary findings survive three independent criteria: |ΔĀ| ≥ 0.004, permutation FDR q < 0.05 (1,000 permutations with Benjamini–Hochberg correction), and Jaccard stability ≥ 0.70 across 10 random 80/20 subsamples.

## License

This code is released under the MIT License. See `LICENSE`.
