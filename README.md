# SentiCouple

**Multimodal sentiment prediction with contiguous visual gaps.**

SentiCouple starts with a transcript-based forecast. It shares one signed update between the positive-to-negative log odds and the intensity estimate, while a separate contrast adjusts the neutral logit. This gives graded sentiment labels a direct way to guide the class forecast when the visual stream is incomplete.

Qisong Zhou, Shu Zhang, Yi Liu, and Yuting Liu. Southwest Petroleum University.

## Results

Relative to a matched independent intensity head, the shared update lowered ranked probability score by 0.0059 on CH-SIMS v2 and class-balanced ranked probability score by 0.0034 on RAVDESS. On CH-SIMS v2, a 32-clip forward pass using cached inputs took 1.97 ms on an RTX 5070.

## Start here

| Task | Entry point |
| --- | --- |
| Train the text anchor and shared or independent heads | [`train.py`](train.py) |
| Inspect the architecture and shared score update | [`networks.py`](networks.py) |
| Match the published fusion cores | [`benchmark.py`](benchmark.py), [`published_baselines.py`](published_baselines.py) |
| Recompute the paired estimates | [`primary_comparisons.py`](primary_comparisons.py), [`conditional_intensity_effects.py`](conditional_intensity_effects.py) |
| Regenerate the manuscript figures from prediction packets | [`paper_figures.py`](paper_figures.py), [`REPRODUCE.md`](REPRODUCE.md) |

## Setup

Use Python 3.12. The reported training runs used PyTorch 2.11 with CUDA 12.8.

```sh
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python train.py --help
```

Obtain the feature files from the original providers listed in [REPRODUCE.md](REPRODUCE.md). The example below trains the fixed-pooling model and its matched controls on CH-SIMS v2 at seed 97.

```sh
python train.py --data data/ch_sims_v2_unaligned.pkl --group-reference data/ch_sims_unaligned_39.pkl --split new-videos --selection rps --seed 97 --backbone temporal --visual-encoding unscaled --inputs visual --calibrate-score-gain --skip-reference-analysis --arms intensity intensity_pointwise fusion independent_intensity --output runs/sims_v2_compact_pooled_seed97
```

The peer-review companion archive contains the paired predictions, fitted example checkpoints and source data for the figures. The public repository contains the scripts and dependency specifications. Source datasets are obtained from their publishers under their respective access terms.
