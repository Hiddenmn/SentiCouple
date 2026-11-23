# Reproduce SentiCouple results

## Data sources

- [CH-SIMS, CMU-MOSI and CMU-MOSEI](https://github.com/thuiar/MMSA#2-datasets) and [CH-SIMS v2](https://thuiar.github.io/sims.github.io/chsims) provide feature packets. Store the CH-SIMS and CH-SIMS v2 packets under `data/` with the filenames shown in the training command.
- [RAVDESS strength annotations](https://doi.org/10.1371/journal.pone.0196391), [RAVDESS facial tracks](https://doi.org/10.5281/zenodo.3255102), and [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D/tree/1658cd342dff90010aa843eaeebd53610a08b1dc) provide the repeated-statement corpora.
- [OMG-Emotion](https://github.com/knowledgetechnologyuhh/OMGEmotionChallenge), [MSP-Face](https://github.com/Lab-MSP/MSP_Face/tree/18fc6d717aab5a5731ad6bd59495ead3f1d80916), [MOUD](https://web.eecs.umich.edu/~mihalcea/downloads.html#MOUD), and [MELD](https://affective-meld.github.io/) provide media sources and annotations.

The `prepare_*.py` and `encode_*.py` entry points describe their inputs with `--help`. For media extraction, install `opencv-python`, `av` and `transformers` in addition to the pinned training dependencies. The media preparation commands also use FFmpeg.

## Matched training and estimation

- The paired comparisons use seeds `97`, `17`, `521`, `31`, `59`, `113`, `251`, `359`, `701`, and `997`. Pass the same split and mask settings to each model. RPS and intensity errors are computed on the same held-out clips and two gap widths.
- The benchmark adapters use the released implementations of [TFR-Net](https://github.com/thuiar/TFR-Net) at `00f06a68a44a6b6fc0043f01954336f30ec3b090`, [LNLN](https://github.com/Haoyu-ha/LNLN) at `80a052c62b7b3b7a6cb6956846e9d9c121f56eda`, and [RM-Net](https://github.com/rudykon/EmoMosaic) at `04efd21bd4999e15892af8072ee90ba6fe0e2e8d`. Clone those repositories separately. Their adapters additionally require `transformers` and `einops`.

Extract the companion archive and point the scripts to its `research/av_evidence/runs/` directory or a directory with regenerated packets. Figure generation requires pdfLaTeX and Poppler (`pdftoppm`).

```sh
python primary_comparisons.py --runs runs
python conditional_intensity_effects.py --evidence runs/conditional_intensity_evidence.npz --output conditional_intensity_effects.json
python msp_face_effects.py --evidence runs/msp_face_refit_evidence.npz --frozen-evidence runs/msp_face_transfer_matched_evidence.npz --output msp_face_refit_paired.json
python paper_figures.py --runs runs --output figures
python paper_figures.py --runs runs --output figures --supplement
```

## Matched CMU-MOSEI forecasts

With the companion archive or a complete set of refits, regenerate the matched comparison with:

```sh
python mosei_results.py --output reproduced_mosei.json
```

To refit, use the MMSA aligned packet at `data/mosei_aligned_50.pkl` and the archived video list. Each run uses 24 bins and the recorded 35% and 65% visual gaps. The reference adapters retain the released fusion and training components on the common text and vision features. The LNLN, API, and UCMR-Net regression checkpoints use a proportional-odds calibration fitted on the tuning split. TFR-Net and RM-Net use a joint classification and intensity readout.

| Reference | Source and pinned revision |
| --- | --- |
| LNLN | [Author code](https://github.com/Haoyu-ha/LNLN/tree/80a052c62b7b3b7a6cb6956846e9d9c121f56eda) |
| TFR-Net | [Author code](https://github.com/thuiar/TFR-Net/tree/00f06a68a44a6b6fc0043f01954336f30ec3b090) |
| RM-Net | [Author code](https://github.com/rudykon/EmoMosaic/tree/04efd21bd4999e15892af8072ee90ba6fe0e2e8d) |
| API | [Author code](https://github.com/KX-yolo/API/tree/696110460b064599f5fdb4ae0329bbc2cf356246) |
| UCMR-Net | [Author code](https://github.com/keliang-debiao/UCMR-Net-Code/tree/1b12c4232995b5661d1b5f760cb5749c6db279d9) |

Check out each revision in a separate local directory. Keep `UCMR-Net_code.zip` in its author checkout. Install `transformers` and `einops` for the upstream adapters. Set `LNLN_DIR`, `TFR_DIR`, `RM_DIR`, `API_DIR`, and `UCMR_DIR` to the corresponding checkout paths. The refit command creates a separate output directory.

```sh
python benchmark_mosei.py --models api ucmr-net tfr-net rm-net \
  --upstream "$API_DIR" "$UCMR_DIR" "$TFR_DIR" "$RM_DIR" --output refit_mosei --device cuda
python mosei_results.py --matched-runs refit_mosei --output refit_mosei_results.json
python mosei_results.py --profile-upstream "$LNLN_DIR" "$API_DIR" "$UCMR_DIR" "$TFR_DIR" "$RM_DIR" \
  --output replayed_mosei_with_timing.json
```

The first command uses the bundled same-seed SentiCouple forecasts to match clip IDs, ratings and masks. To refit those forecasts, use `train.py` with `--split official --eval-groups splits/mosei_eval_videos.txt --backbone attention --inputs visual --skip-reference-analysis --stream-text --selection rps --arms fusion intensity independent_intensity` for seeds 97, 17 and 521. Save each run as `mosei_fresh_visual_seed<seed>` under a new run directory.

Refit LNLN using `benchmark.py --model lnln --upstream <LNLN_DIR> --corpus mosei --head regression --split official --inputs visual --data data/mosei_aligned_50.pkl --eval-groups splits/mosei_eval_videos.txt --epochs 32 --patience 8 --batch 32 --lr 0.001 --weight-decay 0.01 --device cuda`. Pass the seed and its matching SentiCouple run through `--seed` and `--match-run`. Save each run as `mosei_lnln_regression_seed<seed>` alongside the SentiCouple fits. Point `--reference-runs` to that directory to use the refits in `benchmark_mosei.py` and `mosei_results.py`. The timing replay uses the bundled seed 97 weights and checks its forecasts against the saved probabilities and scores before timing.

## Reproduce OMG-Emotion GPU latency

Use the OMG-Emotion feature packet and bundled seed 97 weights. Run the four methods in the order `pointwise`, `tfr_joint`, `lnln_joint`, `rmnet_core`, then repeat in reverse order. Each invocation checks the forecast and times the same single-clip and 32-clip batches. Use the pinned checkout for each reference through `--upstream`.

```sh
python published_inference.py --cohort omg --method pointwise --runs runs \
  --data data/omg_unaligned.pkl --output omg_pointwise_timing.json
python published_inference.py --cohort omg --method tfr_joint --runs runs \
  --data data/omg_unaligned.pkl --upstream "$TFR_DIR" --output omg_tfr_timing.json
```

Table 4 reports first-order medians. `runs/omg_inference_profile.json` retains both timing orders and the recorded quartiles.

## Media examples

The CREMA-D excerpts in the paper retain their [database contents license](https://opendatacommons.org/licenses/dbcl/1-0/). To reconstruct the figure packets from the original clips, run `figure_example.py`, `gap_examples.py`, or `statement_examples.py` with the CREMA-D checkout and the saved fit inputs. Each script documents the required paths.
The modality figure compares the same utterance and gap masks using the stored three-stream and single-stream fits. Regenerate the packet with `audiovisual_example.py` using the seed-97 CH-SIMS v2 forecasts, the paired stream ablations, and the original feature packet. The same packet supplies the acoustic channels in the architecture figure.
