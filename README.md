# ORBIT: Learning Gene Program Co-Activation Structure for Cell-Type-Stratified Pathway Rewiring Analysis in Single-Cell Transcriptomics

ORBIT is a self-supervised transformer that learns asymmetric, directed dependencies among gene programs from observational single-cell RNA-sequencing data alone. Stage 1 trains the pathway attention transformer with a masked-reconstruction objective augmented by an intervention-consistent influence loss, which forces the attention matrix to encode \emph{directional} dependency. Stage 2 fine-tunes a thin classification head (3.9% of Stage 1 parameters) on cell-type labels to enable condition-stratified analysis. The learned pathway attention matrix is the principal scientific output.

**Interpretation.** ORBIT's attention matrix is trained, not post-hoc inspected: the intervention-consistent objective gives every weight a defined predictive role in score space. However, the attention matrix should be read as an interpretable proxy for directed program co-activation structure, not as an identification result. ORBIT recovers candidate directed dependencies that survive permutation FDR, subsample stability, and cross-vocabulary replication; it does not certify causal mechanism. Throughout, "directed dependency" means asymmetric predictive influence under input ablation, and "co-activation" refers to the underlying expression-covariance structure ORBIT learns from.

## Method overview

Given a nucleus's library-size-normalized, log1p-transformed expression vector and a curated vocabulary of *P* gene programs, ORBIT computes a per-program scalar activation score

```
s_{i,p} = (1 / sqrt(|π_p|)) * Σ_{g ∈ π_p} g_{i,g}
```

and produces (i) a *P* × *P* attention matrix encoding the directed dependency from each program to every other program and (ii) a cell-type label. The Stage 1 objective combines three terms with the loss weights given in Appendix A:

- **Masked reconstruction.** 40% of program scores per nucleus are randomly zeroed; a linear decoder reconstructs the masked scores. This provides the basic co-activation signal.
- **Entropy regularization.** Penalizes attention rows whose normalized entropy deviates from 0.5, preventing collapse to a single program or dispersion to uniform.
- **Intervention-consistent influence loss.** For each of *n*<sub>*t*</sub> = 4 randomly sampled target programs *p* per step, a full forward pass and an intervened pass with *s*<sub>*p*</sub> set to zero produce two reconstructed score vectors. The observed shift Δ*s*<sub>*q*</sub><sup>(*p*)</sup> is taken as a stop-gradient target; the predicted shift is *A*[*p*, *q*] · *s*<sub>*p*</sub>, with no learnable parameters between the attention weight and the prediction. The loss is a variance-normalized MSE between observed and predicted shifts. This converts attention from a post-hoc summary into a trained quantity with a defined predictive role and is the mechanism that produces asymmetric *A*[*p*, *q*] ≠ *A*[*q*, *p*].

Stage 2 proceeds in two phases: a frozen-backbone linear probe (up to 50 epochs, early stopping on validation macro-F1, patience 5), followed by 20 epochs of end-to-end fine-tuning at a reduced learning rate. Rewiring analysis is purely post-hoc: per cell type *ct*, the signed delta Δ*Ā*<sub>*ct*</sub> = *Ā*<sup>AD</sup><sub>*ct*</sub> − *Ā*<sup>ctrl</sup><sub>*ct*</sub> identifies gained and lost co-activation, with significance assessed by 1,000-permutation FDR (Benjamini–Hochberg, per directed pair). Gene-level projection through the binary membership matrix *M* yields the gene-pathway influence map *I*<sub>*g*, *p*<sub>2</sub></sub> = Σ<sub>*p*<sub>1</sub></sub> *M*<sub>*g*, *p*<sub>1</sub></sub> · *Ā*<sub>*p*<sub>1</sub>, *p*<sub>2</sub></sub>.

## Architecture

The model has three components, totaling 415,104 parameters at Stage 1 (ABA vocabulary, *P* = 220):

- **Pathway encoder.** Each scalar score *s*<sub>*p*</sub> is projected to a 96-dimensional vector via a linear layer with weights shared across programs, followed by LayerNorm and GELU. A 32-dimensional learnable program-identity embedding is concatenated to give each program a unique identity, then the result is projected to *D* = 128 with LayerNorm, GELU, and dropout 0.30.
- **Pathway attention transformer.** Two stacked pre-norm transformer blocks. Each block applies 8-head self-attention (attention dropout 0.10) followed by a feed-forward network of expansion factor 4 with GELU and dropout 0.10. The attention tensor of the **final** block is retained for the intervention-consistent loss and the reported attention matrix.
- **Classification head.** A single linear layer mapping the 128-dimensional mean-pooled cell embedding to *n*<sub>cls</sub> = 7 logits. Its small capacity (16,000 parameters, 3.9% of Stage 1) prevents label supervision from distorting the Stage 1 co-activation structure.

Three quantities derived from the attention tensor are used throughout the paper, and the distinction matters when reproducing or extending results. The **per-nucleus, per-head map** *A*<sup>(*i*,*h*)</sup> ∈ ℝ<sup>*P*×*P*</sup> is the raw post-softmax attention from the final transformer block; the **head-averaged map** *Ã*<sup>(*i*)</sup> = (1/*H*) Σ<sub>*h*</sub> *A*<sup>(*i*,*h*)</sup> is used inside the intervention-consistent loss with gradient flow to the final block's query and key parameters; and the **reported attention matrix** *Ā*<sub>*p*,*q*</sub> = (1/(*N*·*H*)) Σ<sub>*i*,*h*</sub> *A*<sup>(*i*,*h*)</sup><sub>*p*,*q*</sub> is the dataset-level mean, computed post-softmax, **not symmetrized**, taken from the final transformer block. *Ā* is the matrix shown in every heatmap and used in every downstream analysis.

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

The real-data experiments use three publicly available human prefrontal cortex snRNA-seq datasets, all retrievable from GEO:

| Dataset | GEO accession | Role | n nuclei |
|---|---|---|---|
| Morabito et al. (2021) | GSE174367 | Reference atlas (training, primary results) | 191,890 |
| Lau et al. (2020) | GSE157827 | Held-out AD cohort (cross-dataset replication) | ≈ 92,000 |
| Otero-Garcia et al. (2022) | GSE189819 | Held-out neuron-enriched control | ≈ 20,000 |

Raw downloads from GEO are converted to AnnData via the loaders in `src/orbit/data_pipeline.py`, which apply the preprocessing reported in the paper: library-size normalization, log1p transformation, removal of nuclei with fewer than 200 detected genes or more than 10% mitochondrial reads, retention of all protein-coding genes intersected against the union of the three pathway vocabularies, and consolidation of vascular subtypes into a single `Vasc` class. No highly-variable-gene selection is applied; the pathway membership matrix *M* implicitly determines which genes contribute to each program score. The same preprocessing is applied identically to all three datasets to ensure that gene identifiers and pathway memberships align across cohorts.

Place the downloaded files into:

```
data/
├── reference/
│   ├── GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5
│   └── GSE174367_snRNA-seq_cell_meta.csv.gz
├── disease/
│   └── [GSE157827 per-donor files]
└── control/
    └── [GSE189819 per-donor .h5 files]
```

Pathway vocabularies are constructed automatically: KEGG and Reactome are loaded from cached GMT files via `pathway_kegg.py` and `pathway_reactome.py`; the Allen Brain Atlas vocabulary is built from the reference atlas's curated cell-type marker lists. All three vocabularies are filtered to programs with at least 5 member genes intersecting the dataset, yielding *P* = 220 (ABA), 318 (KEGG), and 170 (Reactome).

## Reproducing the experiments

The end-to-end pipeline runs from a single entry point. Edit the `BASE` and `CONFIG` block at the top of `scripts/main.py` to point to your data directory and select the vocabulary, then:

```bash
python scripts/main.py
```

This executes, in order: data loading and preprocessing, pathway-mask construction, Stage 1 self-supervised training, Stage 2 cell-type classification fine-tuning, evaluation on the held-out 20% test split, prediction on the disease and control cohorts, mean attention computation, AD-versus-control dysregulation analysis with permutation FDR, and figure generation.

The synthetic identifiability benchmark (Section 3.2 and Table 1 of the paper) runs independently:

```bash
python -m orbit.synthetic
```

YAML configurations matching each vocabulary and the synthetic benchmark are provided in `configs/` and document every hyperparameter; the values in `main.py`'s `CONFIG` dictionary mirror them.

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
scripts/
  main.py                     End-to-end pipeline
configs/                      YAML configurations per vocabulary
```


## Benchmarking

All baseline comparisons in Tables 1 and 2 of the paper are run from `src/orbit/classification_benchmark.py` (real-data classification) and `src/orbit/synthetic.py` (synthetic identifiability), with versions, training protocols, and evaluation splits fixed to ensure parity:

- **CellTypist** (Domínguez Conde et al., 2022). Version 1.6.x. Trained from scratch on the same 80/20 stratified split (seed 42) of the Morabito reference atlas, using the package's default logistic-regression backbone with SGD and the recommended `mini-batch` mode for ≥ 100k cells. We do not use any of the published CellTypist pretrained models, since they were trained on overlapping atlases.
- **scANVI** (Xu et al., 2021). scvi-tools 1.0.x. Trained on the same split using default hyperparameters with `n_latent=30`, `n_layers=2`, and 400 epochs of semi-supervised training following the official tutorial. Cell-type predictions are taken from the posterior mode.
- **MLP baseline.** A 3-layer feedforward network (hidden dims 512 → 256, GELU, dropout 0.10) operating on the same *P* = 220 KEGG pathway scores ORBIT receives. Trained with AdamW (lr 1e-3, weight decay 0.01) for 50 epochs with early stopping on the validation split. Provides an architecture-controlled comparison that isolates the contribution of attention.
- **Synthetic-benchmark baselines** (Table 1). Pearson |ρ|, mutual information, distance correlation, HSIC, Random Forest, and an MLP, all evaluated on the same *N* = 8,000 cell, *P* = 40 program synthetic dataset described in Section 3.2 and Appendix D. Implementations and seeds are fixed in `synthetic.py`.

All real-data baselines use the identical 80/20 stratified split (seed 42), identical preprocessing, identical class-imbalance handling (inverse-frequency weighting, vascular oversampling, vascular excluded from macro-F1), and identical evaluation script. To re-run the full classification benchmark from the command line:

```bash
python -m orbit.classification_benchmark
```

To re-run the synthetic benchmark:

```bash
python -m orbit.synthetic
```

## Hyperparameters

Every hyperparameter used in the paper is specified in `configs/{aba,kegg,reactome,synthetic}.yaml` and mirrored in `scripts/main.py`. The optimizer-level settings are unified across the three vocabulary models.

**Stage 1** (Appendix Table 1): AdamW with β₁ = 0.9, β₂ = 0.999, weight decay 0.01; peak learning rate 1e-4 with cosine schedule and 10% linear warmup; gradient clipping at norm 1.0; batch size 128; 50 epochs; masking ratio 0.40; *n*<sub>*t*</sub> = 4 intervention targets sampled uniformly per step; loss weights 1.00 (reconstruction), 0.02 (entropy regularization), 0.05 (intervention-consistent influence). Model selection uses the lowest combined Stage 1 validation loss on a 10% held-out split of the reference atlas.

**Stage 2** (Appendix Table 2): two-phase schedule. Phase 1 freezes the Stage 1 weights and trains the linear classification head only, for up to 50 epochs with early stopping on validation macro-F1 (patience 5). Phase 2 unfreezes all weights and fine-tunes for 20 additional epochs at learning rate 5e-5 with cosine warmup. Class imbalance is handled by inverse-frequency loss weighting; vascular cells (< 1% of nuclei) are oversampled during training and excluded from the macro-F1 average reported in Tables 2 and 5.

**Architecture** (Appendix Table 3): see the Architecture section above. All optimizer betas, weight decay, gradient clipping, and the model-selection criterion are recorded in the YAML configs and are not duplicated here.

## Reproducibility

- **Seed.** The default random seed is 42. It controls dataset splitting, weight initialization, masking-mask sampling, and intervention-target sampling. Single-seed runs reproduce the point estimates reported in the paper.
- **Validation discipline.** Hyperparameters were selected on a 10% validation split of the Morabito reference atlas and frozen before any evaluation on the held-out 20% test split. No test-set information was used during model or hyperparameter selection.
- **Significance protocol.** Rewiring results reported as primary findings (Section 3.5) survive three independent criteria simultaneously: (i) |Δ*Ā*| ≥ 0.004; (ii) permutation FDR *q* < 0.05 with 1,000 condition-label permutations and Benjamini–Hochberg correction per directed pair; (iii) Jaccard stability ≥ 0.70 of the top-*k* edge set across 10 random 80/20 subsamples.
- **Cross-vocabulary replication.** Findings highlighted in the main text replicate across all three pathway vocabularies (ABA, KEGG, Reactome). Vocabulary-specific findings are flagged as such in the captions.
- **Determinism.** PyTorch's CUDA kernels are not bitwise-deterministic across GPU architectures even with `torch.use_deterministic_algorithms(True)`; results within ± 0.001 macro-F1 are expected across A100 / H100 / RTX 6000 hardware. The CPU synthetic benchmark is fully deterministic.

## License

This code is released under the MIT License. See `LICENSE`.
