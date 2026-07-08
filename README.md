# On the Transferability and Scalability of Generative Homology-Aware Protein Language Models

**Master's thesis — Antonia Theres Ebner, BSc** · Johannes Kepler University Linz

📄 **[Thesis (PDF)](ADD_PDF_LINK_HERE)** · 📦 **[Preprocessed data (Zenodo, DOI 10.5281/zenodo.21235210)](https://zenodo.org/records/21235210)** · 💻 **[GitHub](https://github.com/antoniaebner/Generative-PLM-Transfer)**

---

This repository investigates whether protein language models trained as generative models can serve as effective *feature extractors*. We evaluate three generative, homology-aware PLMs — **PoET**, **ProtMamba**, and **Prot-xLSTM** — as **frozen encoders**, and study how well, and why, their embeddings transfer to a diverse set of downstream prediction tasks.

We analyze transferability — asking whether any improvement over a one-hot baseline stems from overparameterization, inductive bias, or weight statistics — and characterize how it **scales** along four axes:

| Axis | What varies | Controlled by |
| --- | --- | --- |
| 🐘 **Model size** | Small vs. large checkpoint (`protxlstm_26M_30B` → `protxlstm_102M_60B`) | `--encoder_name` |
| 📚 **Depth** | Which layer's embedding is extracted | `--manual_layer_min` / `--manual_layer_max` |
| ⏳ **Pretraining progress** | Partially-trained checkpoints from along the pretraining run | `--checkpoint` |
| 🧬 **Context size** | Amount of homologous MSA context at inference | `--max_context_sequences` |

## Table of contents

- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Downloading data, checkpoints & model weights](#downloading-data-checkpoints--model-weights)
- [Setting up the MSA search database & preprocessing data](#setting-up-the-msa-search-database--preprocessing-data)
- [Embedding proteins](#embedding-proteins)
- [Evaluating protein embeddings](#evaluating-protein-embeddings)
- [Acknowledgments, citation & license](#acknowledgments-citation--license)

## Quickstart

```bash
# 1. Install the environments (see Installation for why there are three)
conda env create -f requirements_clm.yml      # -> gen_prottl
conda env create -f requirements_mlm.yml      # -> mlm_prottl
conda env create -f requirements_msagen.yml   # -> msagen
pip install -e .[gen_prottl, mlm_prottl] --no-deps

# 2. Grab the data + checkpoints, then build the search databases
#    (see the sections below for the Zenodo links)
./scripts/setup_databases.sh        # UniRef30 + PDB -> ./msas/database

# 3. Retrieve homologs and pack MSAs
./scripts/execute_preprocessing.sh

# 4. Cache embeddings from a frozen model
./scripts/execute_gen_emb_protxlstm.sh

# 5. Evaluate with a lightweight downstream head
./scripts/execute_run_sklearn_protxlstm.sh
```

Each `scripts/execute_*.sh` file is a **queue** — a list of Python invocations, one per dataset/split/config. Edit it to pick what you actually want to run.

## How it works

The benchmark follows a two-stage protocol that decouples representation extraction from downstream evaluation:

1. **Representation extraction.** For each query sequence, homologs are retrieved with MMseqs2 and supplied as multiple-sequence-alignment (MSA) context to a *frozen* pretrained model. The resulting embeddings are computed once and cached under `embeddings/`.
2. **Downstream probing.** A lightweight prediction head — ridge/logistic regression (scikit-learn) or a single linear PyTorch layer — is fitted on the cached representations to assess their transferability to each task.

Decoupling generation from evaluation lets us benchmark freely across models, layers, checkpoints, and hyperparameters without recomputing anything.

The Python entry points (`run_*.py`) do one job each; the `scripts/*.sh` queues call them and log stdout to `logs/<script>/`. Shared constants (model registries, layer/dim tables, dataset paths, seeds) live in `src/params/`.

## Installation

The project relies on **three separate conda environments**, as the generative models, the MLM baselines, and the MMseqs2 tooling impose mutually incompatible dependency constraints that cannot be satisfied within a single environment:

| Env | File | Used for |
| --- | --- | --- |
| `gen_prottl` | `requirements_clm.yml` | **Main experiments** — Prot-xLSTM, ProtMamba, PoET (MSA preprocessing, embedding, evaluation) |
| `mlm_prottl` | `requirements_mlm.yml` | **Baselines** — CARP & ESM (MLM) |
| `msagen` | `requirements_msagen.yml` | MMseqs2 homology search (MSA building) |

```bash
conda env create -f requirements_clm.yml      # -> gen_prottl
conda env create -f requirements_mlm.yml      # -> mlm_prottl
conda env create -f requirements_msagen.yml   # -> msagen
pip install -e .[gen_prottl, mlm_prottl] --no-deps
```

Most stages run in `gen_prottl` (use `mlm_prottl` for the CARP/ESM baselines). `execute_preprocessing.sh` switches between `msagen` and `gen_prottl` on its own; for the other scripts, **activate the right environment yourself first**.

## Downloading data, checkpoints & model weights

**Preprocessed datasets** are on Zenodo: [10.5281/zenodo.21235210](https://zenodo.org/records/21235210). Download and unzip into place (`data/`)


**Model weights** come from each model's original source:

| Model | Family | Checkpoints | Weights |
| --- | --- | --- | --- |
| **Prot-xLSTM** | xLSTM (recurrent) | `protxlstm_26M_30B`, `protxlstm_102M_60B` | [ml.jku.at](https://ml.jku.at/research/Bio-xLSTM/downloads/Prot-xLSTM/checkpoints/) |
| **ProtMamba** | Mamba (state-space) | (`protmamba_28M_30B`, not publicly available) `protmamba_107M_195B` | [GitHub release](https://github.com/Bitbol-Lab/ProtMamba-ssm/releases/tag/v1.0) |
| **PoET** | Transformer | `poet_201M` | [Zenodo](https://zenodo.org/records/10061322) |
| **CARP / ESM** | MLM baselines | various | fetched automatically ([CARP](https://github.com/microsoft/protein-sequence-models), [ESM](https://github.com/facebookresearch/esm); CARP intermediate checkpoints [here](https://zenodo.org/records/10631963)) |

> CARP and ESM weights download themselves on first use — nothing to place by hand.

Intermediate checkpoints should be saved under `pretrain_checkpoint/<architecture>`, and final model weights should be saved in `models/`.

## Setting up the MSA search database & preprocessing data

The homology-aware models need MSA context, so we first build a search database, then retrieve homologs and pack them into the memmap format the encoders read.

```bash
# 1. Download UniRef30 + PDB and build the mmseqs database into ./msas/database
./scripts/setup_databases.sh

# 2. Retrieve homologs for every dataset and pack them
./scripts/execute_preprocessing.sh
```

Under the hood, step 2 runs two things back-to-back:

1. **Homology search** (`msagen`): `src/mmseqs/search.py` → per-sequence `.a3m` files under `msas/individual_msas/…`.
2. **MSA preprocessing** (`gen_prottl`): `run_msa_preprocessing.py` → packed memmap files (`.dat`, `_similarity.dat`, …).

The datasets span protein engineering, annotation, structure, and evolution:

| Task type | Dataset | Splits |
| --- | --- | --- |
| Protein engineering (`proeng`) | GB1 (`gb1`, `gb1_trunc`) | `low_vs_high`, `two_vs_rest`, `one_vs_rest` |
| Protein engineering (`proeng`) | AAV (`aav`) | `low_vs_high`, `two_vs_many`, `one_vs_many` |
| Protein engineering (`proeng`) | Thermostability (`thermo`) | `mixed_split` |
| Annotation | Subcellular localization (`scl`) | `balanced` |
| Structure | Secondary structure (`tape_ss3_processed`) | `casp12`, `cb513`, `ts115` |
| Evolution | Remote homology (`tape_rh_processed`) | `fold_holdout`, `superfamily_holdout`, `family_holdout` |

## Embedding proteins

Each model encodes every split once and caches the result under `embeddings/`. Pick the script for the model you want:

```bash
./scripts/execute_gen_emb_protxlstm.sh   # Prot-xLSTM
./scripts/execute_gen_emb_protmamba.sh   # ProtMamba
./scripts/execute_gen_emb_poet.sh        # PoET
```

Each line calls `run_pregen_emb.py`, e.g.:

```bash
python run_pregen_emb.py \
    --dataset_path="data/proeng/gb1/sampled.csv" \
    --msa_path="msas/individual_msas/proeng/gb1_trunc/gb1_trunc_memmap" \
    --encoder_name="protxlstm_102M_60B" \
    --checkpoint_folder="pretrain_checkpoints/protxlstm" \
    --checkpoint=1 \
    --embed_folder="embeddings" \
    --max_context_sequences=200
```

The four scaling axes are swept right here by varying `--encoder_name` (size), `--checkpoint` (pretraining progress), `--manual_layer_*` (depth), and `--max_context_sequences` (context size).
Similarly, the ablations are conducted here by passing `--reset_param=True` (rand-init) or `--resample_param=True` (stat-init).

## Evaluating protein embeddings

Train a lightweight head on the cached embeddings. Two interchangeable back-ends — pick **one per model**:

| | sklearn | pytorch |
| --- | --- | --- |
| Script | `execute_run_sklearn_*.sh` → `run_protran_sklearn.py` | `execute_run_pytorch_*.sh` → `run_protran_pytorch.py` |
| Head | Ridge / logistic regression (sweeps `--alphas`) | Single linear layer, early stopping |
| Best for | Fast probing, closed-form regression | Larger datasets, GPU training |

```bash
./scripts/execute_run_sklearn_protxlstm.sh   # or execute_run_pytorch_protxlstm.sh
```

A representative invocation:

```bash
python run_protran_sklearn.py \
    --dataset_path="data/proeng/gb1/two_vs_rest.csv" \
    --encoder_name="protxlstm_102M_60B" \
    --embed_folder="embeddings" \
    --all_embed_layers=True \
    --all_result_folder="results/sklearn-protxlstm"
```

> **One-hot baselines** skip embedding entirely — `execute_run_{sklearn,pytorch}_onehot.sh` encode on the fly.
> For **mutational datasets**, `run_permute_emb.py` reuses a reference split's embeddings for a related split instead of re-encoding near-duplicate sequences.

**Aggregate & plot.** Once results land in `results/` (Spearman & NDCG for regression, accuracy & ROC-AUC for classification), collapse them into summary CSVs and figures:

```bash
python run_results_analysis.py
```

For the interactive figures, see the notebooks in `notebooks/`.

## Acknowledgments, citation & license

This project stands on the shoulders of some excellent open-source work, and extends the **[microsoft/protein-transfer](https://github.com/microsoft/protein-transfer)** benchmark methodology to homology-aware generative models and the context-scaling axis:

- **Prot-xLSTM** / **xLSTM** — [ML-JKU/Prot-xLSTM](https://github.com/ml-jku/Prot-xLSTM), [NX-AI/xlstm](https://github.com/NX-AI/xlstm)
- **ProtMamba** / **mamba_ssm** — [Bitbol-Lab/ProtMamba-ssm](https://github.com/Bitbol-Lab/ProtMamba-ssm), [state-spaces/mamba](https://github.com/state-spaces/mamba)
- **PoET** — [OpenProteinAI/PoET](https://github.com/OpenProteinAI/PoET)
- **CARP / ESM** baselines — [microsoft/protein-sequence-models](https://github.com/microsoft/protein-sequence-models), [facebookresearch/esm](https://github.com/facebookresearch/esm)
- **MMseqs2** homology search — [soedinglab/MMseqs2](https://github.com/soedinglab/MMseqs2)

Vendored source files (`src/models/protxlstm/`, `src/models/mamba/`, `src/models/poet/`) keep their upstream license headers and note any local modifications.

**Citation:**

```bibtex
@mastersthesis{ebner2026generative,
  author  = {Ebner, Antonia},
  title   = {On the Transferability and Scalability of Generative Homology-Aware Protein Language Models},
  school  = {Johannes Kepler University Linz},
  year    = {2026},
  address = {Austria},
  type    = {Master's thesis},
}
```

**License:** Apache License 2.0. Vendored code under `src/models/` is distributed under its respective upstream license (Apache-2.0 and others; see file headers).
