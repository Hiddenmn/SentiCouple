# SentiCouple

**Multimodal sentiment prediction with contiguous visual gaps.**

SentiCouple starts with a transcript-based forecast. It shares one signed update between the positive-to-negative log odds and the intensity estimate, while a separate contrast adjusts the neutral logit. This gives graded sentiment labels a direct way to guide the class forecast when the visual stream is incomplete.

Qisong Zhou, Shu Zhang, Yi Liu, and Yuting Liu. Southwest Petroleum University.

## Results

Sharing the signed update reduced ranked probability score by 0.0081 with text, audio and vision on CH-SIMS v2 and by 0.0059 with text and vision. It also reduced class-balanced ranked probability score by 0.0034 on held-out RAVDESS actors. In the matched three-stream comparison on CH-SIMS v2 and CMU-MOSEI, SentiCouple used 85–88% fewer downstream parameters than UCMR-Net and achieved 3.5–4.5 percentage points higher mean macro-F1.

## Start here

| Task | Entry point |
| --- | --- |
| Train the text anchor and shared or independent heads | [`train.py`](train.py) |
| Inspect the architecture and shared score update | [`networks.py`](networks.py) |
| Match the published fusion cores | [`benchmark.py`](benchmark.py), [`published_baselines.py`](published_baselines.py) |
| Compare six models under matched visual gaps | [`benchmark_mosei.py`](benchmark_mosei.py), [`mosei_results.py`](mosei_results.py), [`frozen_benchmark.py`](frozen_benchmark.py) |
| Replay the three-stream comparison with UCMR-Net | [`three_stream_benchmark.py`](three_stream_benchmark.py) |
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
