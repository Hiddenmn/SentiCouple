# Reproduce SentiCouple results

The companion archive contains the inputs used to draw the manuscript figures, the run configurations and the aggregate results. Use this repository for the training and analysis scripts.

## Recreate the figures

Work from `research/av_evidence` after extracting the archive. Use Python 3.12 and the packages in `requirements.txt`. The experiments used PyTorch 2.11.0 with CUDA 12.8 on an NVIDIA GeForce RTX 5070. Install the matching CUDA build with the following commands.

```sh
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The commands use the plotted values and result summaries in `runs/`. Figure generation also uses pdfLaTeX and Poppler's `pdftoppm` to assemble the supplementary panels. The separate LaTeX source archive contains both documents and their vector figures.

```sh
python paper_figures.py --runs runs --output reproduced_figures
python paper_figures.py --runs runs --output reproduced_figures --supplement
```

The benchmark JSON files contain the forecast metrics and inference times used in Tables 4 and 5. Recompute the crossed fit-and-media intervals for the CH-SIMS v2 text-and-vision and three-stream comparisons and for RAVDESS from the grouped loss sums.

```sh
python primary_comparisons.py --grouped runs/primary_grouped_rps.json
```

The grouped record contains clip counts and paired loss sums, with the class weights already applied for RAVDESS. The following command reproduces the ordering gains and intervals in Table 3 from pair counts and concordance sums by source video and fit.

```sh
python conditional_intensity_effects.py --grouped-order runs/within_video_order_groups.json --output reproduced_ordering.json
python conditional_intensity_effects.py --grouped-order runs/within_video_order_groups.json \
  --equal-video-weight --output reproduced_ordering_equal_video.json
```

`runs/conditional_intensity_effects.json` summarizes the ordering and score-calibration comparisons. `runs/within_polarity_examples.npz` gives the 16 clips plotted in the main paper, with updates averaged across the 35% and 65% visual gaps. The MSP-Face transition counts in `runs/msp_face_refit_paired.json` summarize which forecasts improved after visual fusion before and after refitting.

## Recreate inputs and train models

The input and target definitions are in Sections 3 and 4 of the main text. Obtain annotations and features from the original providers below. `prepare_*.py` reads the source releases and `encode_*.py` extracts features where needed. Use `train.py` for the anchor and the coupled or independent heads, and `benchmark.py` for the matched reference cores. Each command documents its inputs with `--help`. The `run.json` files under `runs/` give the archived fit settings. Write new checkpoints and predictions under `refits/`, using the recorded run names and local input paths. The refit commands create new run directories. `splits/mosei_eval_videos.txt` identifies the CMU-MOSEI evaluation videos.

- [CH-SIMS, CMU-MOSI and CMU-MOSEI features](https://github.com/thuiar/MMSA#2-datasets) and [CH-SIMS v2](https://thuiar.github.io/sims.github.io/chsims).
- [OMG-Emotion annotations](https://github.com/knowledgetechnologyuhh/OMGEmotionChallenge) and [MSP-Face annotations and source links](https://github.com/Lab-MSP/MSP_Face/tree/18fc6d717aab5a5731ad6bd59495ead3f1d80916).
- [RAVDESS strength ratings](https://doi.org/10.1371/journal.pone.0196391), [RAVDESS facial tracks](https://zenodo.org/records/3255102), and [CREMA-D media and audience ratings](https://github.com/CheyneyComputerScience/CREMA-D/tree/1658cd342dff90010aa843eaeebd53610a08b1dc).
- [MOUD video reviews](https://web.eecs.umich.edu/~mihalcea/downloads.html#MOUD) and [MELD dialogue](https://affective-meld.github.io/).

Use the published access conditions for each corpus. The OMG-Emotion release supplies the utterance annotations used by `prepare_omg.py`. The MSP-Face release supplies the source links, segments, labels and official partitions used by `prepare_msp_face.py`. CREMA-D is published under the [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/), and the frame excerpts retain the [database contents license](https://opendatacommons.org/licenses/dbcl/1-0/).

## Prepare and fit the fixed-statement RAVDESS evaluation

Install `openpyxl` to read the RAVDESS ratings workbook. The OMG-Emotion and MSP-Face media tools use `yt-dlp`, Node.js, `imageio-ffmpeg` and OpenCV. The CREMA-D video preparation also uses OpenCV.

Set `RAVDESS_TRACKS` and `RAVDESS_RATINGS` to the released OpenFace track ZIP and the speech ratings workbook. The preparation script writes four actor folds. The following commands fit the pointwise encoder and four heads on each fold, with calibration on its tuning actors.

```sh
python prepare_ravdess.py --tracks "$RAVDESS_TRACKS" --ratings "$RAVDESS_RATINGS" --output data/ravdess

for fold in 0 1 2 3; do
  for seed in 97 17 521 31 59 113 251 359 701 997; do
    python train.py --data "data/ravdess/ravdess_fold${fold}.pkl" --split official \
      --seed "$seed" --backbone temporal --visual-encoding pointwise --inputs visual \
      --selection rps --skip-reference-analysis --calibrate-score-gain \
      --arms fusion intensity independent_intensity intensity_shuffled \
      --output "refits/ravdess_fold${fold}_seed${seed}" --device cuda
  done
done
python ravdess_effects.py --runs refits --output reproduced_ravdess.json
```

## Fit the conditional ordering comparison

Use the prepared feature packets below and the ten seeds above. The conditional heads use the same feature encoders and masks as the paired Huber heads on each corpus.

```sh
for seed in 97 17 521 31 59 113 251 359 701 997; do
  python train.py --data data/ch_sims_v2_unaligned.pkl --split new-videos \
    --group-reference data/ch_sims_unaligned_39.pkl --visual-encoding unscaled --seed "$seed" \
    --backbone temporal --inputs visual --selection rps --skip-reference-analysis \
    --calibrate-score-gain --arms conditional_shared conditional_independent conditional_shuffled \
    --output "refits/sims_v2_conditional_intensity_seed${seed}" --device cuda

  python train.py --data data/omg_unaligned.pkl --split official \
    --neutral-margin 0.10 --visual-encoding pointwise --seed "$seed" \
    --backbone temporal --inputs visual --selection rps --skip-reference-analysis \
    --calibrate-score-gain --arms conditional_shared conditional_independent conditional_shuffled \
    --output "refits/omg_conditional_intensity_seed${seed}" --device cuda
done
```

To reproduce the full comparison, also fit the Huber heads and the class-mean controls with the corresponding archived configurations and names under `refits/`.

```sh
cp runs/sims_v2_published_seeds.json runs/omg_published_seeds.json refits/
python conditional_intensity_effects.py --runs refits --output reproduced_conditional_intensity.json
```

## Matched text-and-vision forecasts

After refitting the models, run the four within-corpus comparisons in Table 4 from the resulting predictions and checkpoints.

```sh
python mosei_results.py --cohort sims_v2 --reference-runs refits --output reproduced_sims_v2.json
python mosei_results.py --cohort omg --reference-runs refits --output reproduced_omg.json
python mosei_results.py --cohort mosei --reference-runs refits --matched-runs refits/mosei_matched --output reproduced_mosei.json
python mosei_results.py --cohort mosi --reference-runs refits --output reproduced_mosi.json
```

Each run uses 24 bins and the recorded 35% and 65% visual gaps. API and UCMR-Net retain the published regression objectives and fit proportional-odds calibration on the tuning split. CH-SIMS v2 and OMG-Emotion ratings are rescaled to the seven-bin MOSEI range during training and converted back to the original units for evaluation. LNLN uses the regression readout on CMU-MOSI and CMU-MOSEI and the joint readout on CH-SIMS v2 and OMG-Emotion. TFR-Net and RM-Net use the joint readout throughout.

| Reference | Source and pinned revision |
| --- | --- |
| LNLN | [Author code](https://github.com/Haoyu-ha/LNLN/tree/80a052c62b7b3b7a6cb6956846e9d9c121f56eda) |
| TFR-Net | [Author code](https://github.com/thuiar/TFR-Net/tree/00f06a68a44a6b6fc0043f01954336f30ec3b090) |
| RM-Net | [Author code](https://github.com/rudykon/EmoMosaic/tree/04efd21bd4999e15892af8072ee90ba6fe0e2e8d) |
| API | [Author code](https://github.com/KX-yolo/API/tree/696110460b064599f5fdb4ae0329bbc2cf356246) |
| UCMR-Net | [Author code](https://github.com/keliang-debiao/UCMR-Net-Code/tree/1b12c4232995b5661d1b5f760cb5749c6db279d9) |

Check out each revision in a separate local directory. Keep `UCMR-Net_code.zip` in its author checkout. Install `transformers` and `einops` for the upstream adapters. Set `LNLN_DIR`, `TFR_DIR`, `RM_DIR`, `API_DIR`, and `UCMR_DIR` to the corresponding checkout paths. Fit the paired SentiCouple and LNLN models below before running the refit and summary commands.

```sh
python benchmark_mosei.py --models api ucmr-net tfr-net rm-net \
  --upstream "$API_DIR" "$UCMR_DIR" "$TFR_DIR" "$RM_DIR" --reference-runs refits --output refits/mosei_matched --device cuda
python mosei_results.py --reference-runs refits --matched-runs refits/mosei_matched \
  --output refit_mosei_results.json
python mosei_results.py --reference-runs refits --matched-runs refits/mosei_matched \
  --profile-upstream "$LNLN_DIR" "$API_DIR" "$UCMR_DIR" "$TFR_DIR" "$RM_DIR" \
  --output replayed_mosei_with_timing.json
```

The first command uses the same-seed SentiCouple forecasts to match clip IDs, ratings and masks. Fit these first with `train.py --data data/mosei_aligned_50.pkl --split official --eval-groups splits/mosei_eval_videos.txt --backbone attention --inputs visual --skip-reference-analysis --stream-text --selection rps --arms fusion intensity independent_intensity` for seeds 97, 17 and 521. Save each run with `--output refits/mosei_fresh_visual_seed<seed>`, replacing `<seed>` with the chosen seed.

Refit LNLN using `benchmark.py --model lnln --upstream <LNLN_DIR> --corpus mosei --head regression --split official --inputs visual --data data/mosei_aligned_50.pkl --eval-groups splits/mosei_eval_videos.txt --epochs 32 --patience 8 --batch 32 --lr 0.001 --weight-decay 0.01 --device cuda`. Pass the seed and its matching SentiCouple run through `--seed` and `--match-run`. Save each run as `refits/mosei_lnln_regression_seed<seed>` alongside the SentiCouple fits. The timing replay uses the locally fitted seed 97 weights.

## Refit on CMU-MOSI

Use `data/mosi_unaligned_50.pkl` and the distributed split. Fit SentiCouple with the CMU-MOSEI options above, omitting `--eval-groups`. Name the three fits `mosi_matched_shared_seed<seed>` for seeds 97, 17 and 521.

For each paired reference, run `benchmark.py` with the corresponding upstream checkout, `--corpus mosi --data data/mosi_unaligned_50.pkl --split official --inputs visual --device cuda`, the same seed and its `--match-run`. Use the options below. Name the outputs `mosi_matched_<model>_seed<seed>`, replacing hyphens in model names with underscores. Place all eighteen fits under `refits/` and summarize the cohort with the command above.

| Model | Head | Epochs | Patience | Batch | Learning rate | Weight decay |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TFR-Net | joint | 32 | 8 | 32 | 0.001 | 0.01 |
| LNLN | regression | 32 | 8 | 32 | 0.001 | 0.01 |
| RM-Net | joint | 48 | 10 | 64 | 0.0001 | 0.001 |
| API | regression | 24 | 6 | 64 | 0.0001 | 0 |
| UCMR-Net | regression | 24 | 6 | 64 | 0.001 | 0.0001 |

## Compare frozen forecasts on MSP-Face and MOUD

The frozen transfer rows in Table 4 use the six OMG-Emotion models with their source preprocessing, weights and calibration. Forecasts pair the ten source seeds across both gap widths. MSP-Face uses the refits' test clips and visual masks. MOUD uses polarity annotations and the translated transcripts.

Use the ten locally fitted sets of OMG-Emotion models under `refits/`. Set `OMG_FEATURES` and `MOUD_FEATURES` to the prepared source and target feature packets.

```sh
python frozen_benchmark.py --cohort moud --runs refits --source-data "$OMG_FEATURES" \
  --target "$MOUD_FEATURES" --upstream "$TFR_DIR" "$LNLN_DIR" "$RM_DIR" "$API_DIR" "$UCMR_DIR" \
  --save-evidence refits/moud_predictions.npz --output reproduced_moud.json
```

For MSP-Face, select `--cohort msp_face` and pass the prepared MSP-Face features as the target. Supply `--msp-manifest` with the generated cohort file and `--msp-match` with the target-refit prediction packet. Choose MSP-Face names for the outputs. To regenerate the transition counts, run `msp_face_effects.py --evidence` on the refitted packet and supply the same-mask frozen packet via `--frozen-evidence`.

## Refit API and UCMR-Net on CH-SIMS v2 and OMG-Emotion

Fit SentiCouple under `refits/` using the recorded configurations `sims_v2_compact_pooled_seed<seed>` and `omg_pointwise_heads_seed<seed>`. Refit TFR-Net, LNLN and RM-Net with the archived `tfr_joint`, `lnln_joint` and `rmnet_core` configurations for the same splits and masks. Use their recorded run names, including `sims_v2_tfr_joint_seed97_cuda`. The following API command shows how to pass the locally refitted predictions for seed 97.

```sh
python benchmark.py --model api --upstream "$API_DIR" --head regression --corpus sims \
  --data data/ch_sims_v2_unaligned.pkl --split new-videos --group-reference data/ch_sims_unaligned_39.pkl \
  --match-run refits/sims_v2_compact_pooled_seed97 --output refits/sims_v2_api_seed97 \
  --seed 97 --epochs 24 --patience 6 --batch 64 --lr 0.0001 --weight-decay 0 --device cuda
```

For UCMR-Net, set `--model ucmr-net --lr 0.001 --weight-decay 0.0001`, use its pinned checkout, and substitute `ucmr_net` in the output name. For OMG-Emotion, use `--corpus omg --data data/omg_unaligned.pkl --split official --neutral-margin 0.10`, omit `--group-reference`, and match `refits/omg_pointwise_heads_seed<seed>`. Name the output `omg_api_seed<seed>` or `omg_ucmr_net_seed<seed>`. Run the corresponding summary command above once all six models have been refitted.

## Reproduce GPU latency

Use the corresponding feature packet and the locally fitted seed 97 checkpoints. Each invocation times the same single-clip and 32-clip batches for every model. For the other within-corpus comparisons, select the corresponding cohort. With CMU-MOSEI also pass `--matched-runs refits/mosei_matched`.

```sh
python mosei_results.py --cohort sims_v2 --reference-runs refits \
  --models tfr-net lnln rm-net api ucmr-net \
  --profile-upstream "$TFR_DIR" "$LNLN_DIR" "$RM_DIR" "$API_DIR" "$UCMR_DIR" \
  --output replayed_sims_v2_with_timing.json
```

To time the frozen models, first run the frozen inference command above to save the prediction packet. The following replay uses the same target features and seed 97 checkpoints. A MOUD batch set cycles through its 498 clips.

```sh
python frozen_benchmark.py --cohort moud --runs refits --source-data "$OMG_FEATURES" \
  --target "$MOUD_FEATURES" --upstream "$TFR_DIR" "$LNLN_DIR" "$RM_DIR" "$API_DIR" "$UCMR_DIR" \
  --evidence refits/moud_predictions.npz --profile --output reproduced_moud_with_timing.json
```

For MSP-Face use the same manifest and masks as for frozen inference, and pass its locally saved prediction packet.

Tables 4 and 5 report the median forward time at each batch size. Supplementary Table S3 reports the corresponding interquartile intervals.

## Three-stream comparison on CMU-MOSEI

The audio replication uses the same three seeds and held-out videos as the text-and-vision comparisons. Fit with `train.py` as above, replacing `--inputs visual` with `--inputs audiovisual` and naming each run `mosei_three_stream_seed<seed>`. The summary checks that the visual gaps, held-out ratings and paired seeds match the text-and-vision forecasts.

```sh
python mosei_stream_results.py --runs refits --output reproduced_mosei_streams.json
```

## Repeat the CH-SIMS v2 three-stream head comparison

Table 2 includes ten paired fits on the original 85 test videos. The modality ablations and the UCMR-Net comparison use their three common seeds, 97, 17 and 521. Use the script below to fit the matched class-only, shared and independent heads from the original feature packets.

```sh
for seed in 97 17 521 31 59 113 251 359 701 997; do
  python train.py --data data/ch_sims_v2_unaligned.pkl --split new-videos \
    --group-reference data/ch_sims_unaligned_39.pkl --backbone temporal \
    --selection rps --device cuda --seed "$seed" \
    --arms fusion intensity independent_intensity --output "refits/sims_v2_shared_score_seed${seed}"
done
```

The grouped RPS command above also recovers the ten-fit mean, the gain across the seven additional fits, the native-view gain, and the change in gain from native to gapped views.

## Three-stream comparison with UCMR-Net

Table 5 pairs the three full `SentiCouple` fits with UCMR-Net fitted with its acoustic encoder enabled. Both models use the same text, audio and vision features, partitions and gap masks.

```sh
python three_stream_benchmark.py --cohort sims_v2 --runs refits --output reproduced_sims_v2_three_stream.json
python three_stream_benchmark.py --cohort mosei --runs refits --output reproduced_mosei_three_stream.json
```

Add `--profile-upstream "$UCMR_DIR"` to either command to measure the seed 97 models on CUDA. It replays both models with the recorded audio and visual masks before timing the same batch sizes as Table 4. Add `--categorical-check` to replay the argmax and tuning neutral-offset comparisons from the same fitted runs. RPS and MAE use the probability and intensity forecasts before the neutral decision offset.

To refit UCMR-Net, use the pinned checkout and `benchmark.py --model ucmr-net --head regression --inputs audiovisual --epochs 24 --patience 6 --batch 64 --lr 0.001 --weight-decay 0.0001 --device cuda`. Pass the data and split flags for the corresponding corpus as above. On CH-SIMS v2, match `refits/sims_v2_shared_score_seed<seed>`. On CMU-MOSEI, match `refits/mosei_three_stream_seed<seed>` and use `--eval-groups splits/mosei_eval_videos.txt`. Name the outputs `sims_v2_three_stream_ucmr_net_seed<seed>` or `mosei_three_stream_ucmr_net_seed<seed>` with seeds 97, 17 and 521.

## Recreate the media examples

After fitting `sims_v2_shared_score_seed97` and `sims_v2_streams_seed97` as specified in their `run.json` files, recreate the CH-SIMS v2 example from the paired forecasts and original feature packet.

```sh
python audiovisual_example.py --run refits/sims_v2_shared_score_seed97 --streams-run refits/sims_v2_streams_seed97 --features CH_SIMS_V2_FEATURES --output refits/audiovisual_example.npz --english-transcript "How much longer do we have to stay in this lousy place?"
```

The example supplies the same acoustic and visual strips in the architecture and modality figures. The aggregate modality table uses the matched forecasts at both gap widths from seeds 97, 17 and 521.

To rebuild the video examples, use the CREMA-D checkout linked above, the 24-bin feature packet and the locally fitted CREMA-D checkpoints. Pass the dataset locations in place of `RELEASE_DIR` and `CREMAD_FEATURES`.

```sh
python figure_example.py --run refits/cremad_heads_seed97 --features CREMAD_FEATURES --release RELEASE_DIR --output paper_example.npz
python gap_examples.py --runs refits --features CREMAD_FEATURES --release RELEASE_DIR --output gap_examples.npz
python statement_examples.py --runs refits --release RELEASE_DIR --output statement_examples.npz
```
