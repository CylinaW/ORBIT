# ORBIT: Learning Directed Gene Program Co-Activation from Single-Cell Data via Intervention-Consistent Attention

This repository contains the official implementation of ORBIT. The code reproduces all experiments reported in the paper.

## Overview

ORBIT is a self-supervised transformer that learns a directed, asymmetric P × P co-activation matrix among gene programs from observational single-cell RNA-sequencing data alone. The model is trained in two stages: masked pathway reconstruction with an intervention-consistent auxiliary loss, followed by cell-type classification on a reference atlas. The learned attention matrix is the principal output and is the object of all downstream analyses. Architectural and methodological details, including all hyperparameters, are given in Sections 3 and 4 and Appendix B of the paper.

## Installation

```bash
git clone <anonymous-repo-url>
cd ORBIT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-enabled GPU is recommended. All experiments in the paper were run on a single NVIDIA A100 (40 GB).

## Data

The real-data experiments use publicly available human cortical snRNA-seq datasets. GEO accession identifiers and preprocessing parameters are listed in Appendix A of the paper:

- Reference atlas: GSE174367 (Morabito et al., 2021)
- Held-out cohorts: Lau et al. (2020), Otero-Garcia et al. (2022)

The expected data layout is:

```
data/
├── reference.h5ad
├── query_ad.h5ad
└── query_control.h5ad
```

Pathway vocabularies (KEGG, Reactome, Allen Brain Atlas) are loaded via `src/orbit/pathway_kegg.py` and `src/orbit/pathway_reactome.py`.

## Reproducing the experiments

All experiments are launched through `scripts/main.py` with a configuration file:

```bash
python scripts/main.py --config configs/pretrain_kegg.yaml --data-dir data/
```

The experiment to run is specified inside the YAML configuration. Available configurations are:

- `configs/pretrain_kegg.yaml` — Stage 1 pretraining with the KEGG vocabulary
- `configs/pretrain_reactome.yaml` — Stage 1 pretraining with the Reactome vocabulary
- `configs/pretrain_aba.yaml` — Stage 1 pretraining with the Allen Brain Atlas vocabulary
- `configs/classify.yaml` — Stage 2 cell-type classification fine-tuning
- `configs/benchmark.yaml` — Real-data evaluation (Tables 1–2, Figures 2–4)
- `configs/synthetic.yaml` — Synthetic identifiability benchmark (Table 3, Figure 5)
- `configs/biological.yaml` — Biological validation analyses
- `configs/dysregulation.yaml` — Disease-versus-control dysregulation analyses

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
  dysregulation.py            AD-vs-control attention rewiring analysis
  dysregulation_viz.py        Rewiring figure generation
  synthetic.py                Synthetic generator and triplet evaluation
  visualisation.py            Attention heatmaps and gene-influence plots
scripts/
  main.py                     Single entry point dispatched by config
configs/                      YAML files specifying all paper hyperparameters
```

## Hyperparameters

All hyperparameters used to produce the reported results are stored in the YAML files under `configs/`. The defaults match Table A1 of the paper: encoder dimensions 96→128, two transformer blocks with 8 heads, dropout 0.30 and 0.10, AdamW with peak learning rate 1e-4, cosine schedule with 10% linear warmup, gradient clipping at norm 1.0, batch size 128, 50 epochs for Stage 1; learning rate 5e-5 with linear-probe-then-end-to-end schedule for Stage 2; masking ratio 0.40; intervention sampling n_t = 4; loss weights 1.0 / 0.02 / 0.05 for the reconstruction, entropy, and intervention terms.

## Reproducibility

The default random seed is 42 throughout. Single-seed runs reproduce the point estimates reported in the paper; multi-seed variance is reported in Appendix C.

## License

This code is released under the MIT License. See `LICENSE`.