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

The CREMA-D excerpts in the paper retain their [database contents license](https://opendatacommons.org/licenses/dbcl/1-0/). To reconstruct the figure packets from the original clips, run `figure_example.py`, `gap_examples.py`, or `statement_examples.py` with the CREMA-D checkout and the saved fit inputs. Each script documents the required paths.
The modality figure compares the same utterance and gap masks using the stored three-stream and single-stream fits. Regenerate the packet with audiovisual_example.py using the seed-97 CH-SIMS v2 forecasts, the paired stream ablations, and the original feature packet.
