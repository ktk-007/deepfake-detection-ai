# Unmask: Deepfake Video Detection

Unmask checks whether the face in a video has been swapped or manipulated. Upload a video and it returns
**Likely Real**, **Likely Fake** or **Inconclusive**, with a confidence level, per-frame scores, Grad-CAM heatmaps
showing where the model looked, and a downloadable HTML report.

The main model is **CLIP ViT-L/14 with LayerNorm tuning** (the GenD recipe), trained on the DFDC dataset and tested
on DFDC plus two outside datasets it never saw during training.

**Authors:** Tharun Kumar Kumaravel, Deny Pupalla

---

## Contents

- [Results](#results)
- [How it works](#how-it-works)
- [Data and splits](#data-and-splits)
- [Training](#training)
- [Key findings](#key-findings)
- [Try it online (GitHub Codespaces)](#try-it-online-github-codespaces)
- [Run it locally](#run-it-locally)
- [Reproduce the experiments](#reproduce-the-experiments)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Results

Video-level ROC AUC. Brackets are 95% bootstrap intervals. Each test set was scored **once** per model, after every
choice (checkpoint, calibration, thresholds, ensemble) had been fixed on the validation part.

| Model | Validation (part 08, robust) | DFDC part 09 (test) | Celeb-DF v2 (outside) | WildDeepfake (outside) |
| --- | --- | --- | --- | --- |
| **CLIP ViT-L/14 (main)** | **0.9805** | **0.9827** (0.976-0.988) | **0.9135** (0.887-0.936) | **0.9279** (0.883-0.961) |
| CLIP ViT-B/16 | 0.9690 | 0.9632 (0.951-0.973) | 0.8210 (0.784-0.854) | 0.8914 (0.838-0.933) |
| EfficientNet-B4 (DFDC-winner style baseline) | 0.9753 | 0.9474 (0.933-0.961) | 0.7831 (0.740-0.823) | 0.8200 (0.752-0.878) |
| Ensemble (ViT-L/14 + EfficientNet-B4) | 0.9862 | 0.9744 (0.964-0.983) | 0.8897 (0.861-0.915) | 0.9207 (0.879-0.954) |
| CLIP ViT-B/16 linear probe (no LayerNorm tuning) | 0.7035 | 0.6845 (0.660-0.711) | 0.5919 (0.534-0.644) | 0.7227 (0.641-0.810) |
| Published GenD checkpoint (trained on FF++), for reference | 0.8901 | 0.9650 (0.952-0.976) | 0.9485 (0.931-0.964) | 0.8897 (0.828-0.934) |

"Validation (robust)" is the mean AUC over three views of DFDC part 08: clean, tight face crops, and degraded
(compression and resizing). Full numbers: [`results/final_table.md`](results/final_table.md) and
[`results/phase4/metrics.json`](results/phase4/metrics.json). ROC curves:
[DFDC part 09](results/plots/roc_test09.svg), [Celeb-DF v2](results/plots/roc_celebdf.svg),
[WildDeepfake](results/plots/roc_wild.svg).

App check on held-out demo videos (RTX 4050 laptop GPU): about 4.5-6 s per video; ViT-L/14 got 6 of 7 right, and the
seventh (a Celeb-DF fake) was marked Inconclusive rather than wrong.

---

## How it works

```
video -> 32 evenly spaced frames -> MTCNN face detection -> one face track -> square face crops (224 x 224)
      -> CLIP ViT-L/14 (frozen, LayerNorms tuned) -> L2-normalised features -> linear head -> fake probability per frame
      -> mean over frames -> temperature calibration -> verdict band -> Likely Real / Inconclusive / Likely Fake
```

1. **Frames.** 32 frames spread evenly across the video.
2. **Face track.** MTCNN finds faces; the most confident face in the first frame with one is followed through the
   video (nearest box each frame), so the crop does not jump to another person.
3. **Model.** The CLIP ViT-L/14 image encoder (OpenAI weights) stays frozen except its LayerNorm layers. A linear
   head on the L2-normalised CLS feature gives a fake probability per frame. Fewer than 0.1% of the parameters are
   trained, which helps the model generalise to unseen datasets.
4. **Video score.** The mean of the frame probabilities.
5. **Verdict.** A temperature and two thresholds, all fitted on the validation part only
   ([`results/calibration.json`](results/calibration.json)). Scores between the thresholds are reported as
   **Inconclusive**; the thresholds allow at most 5% of each class on the wrong side.
6. **Explanation.** Grad-CAM over the later transformer blocks (SmoothGrad, masked to the face) on the most
   suspicious frames.

---

## Data and splits

| Role | Data | Size |
| --- | --- | --- |
| Training | DFDC parts 00-07 and 10-19 | 5,810 real videos and their fakes |
| Validation (every decision) | DFDC part 08 | 277 real / 859 fake |
| Test (sealed until the end) | DFDC part 09 | 288 real / 920 fake |
| Outside test | Celeb-DF v2 test list | 518 videos (178 real / 340 fake) |
| Outside test | WildDeepfake test split | 806 clips (396 real / 410 fake) |

Rules followed throughout:
- Split by original video: a real video and all its fakes are always on the same side.
- Test sets were never used for training, tuning or picking thresholds, epochs or settings.
- Augmentation is identical for real and fake, training only.
- FaceForensics++ was not used for training.

No dataset videos or face images are included in this repository. Download the datasets from their official
sources and accept their licenses.

---

## Training

| Model | Trainable | Loss | Augmentation | LR | Batch | Epochs (best) | Speed (RTX 4050) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-L/14 | LayerNorms + head | Cross-entropy | light | 2e-4 | 32 (grad. checkpointing) | 4 (3) | about 28 img/s |
| CLIP ViT-B/16 | LayerNorms + head | Cross-entropy | light | 3e-4 | 64 | 12 (11) | about 95-150 img/s |
| EfficientNet-B4 | all layers | Cross-entropy (smoothing 0.01) | strong | 1e-4 | 32 | 6 (4) | about 130 img/s |

Trained on one laptop: RTX 4050 (6 GB), i7, 16 GB RAM, Windows, PyTorch with bf16 autocast. Every run is listed in
[`results/results.csv`](results/results.csv), with configs in [`configs/`](configs) and logs in [`logs/`](logs).

---

## Key findings

- **LayerNorm tuning matters.** On the same development set (DFDC parts 00-05), tuning the LayerNorms of ViT-B/16
  lifted robust validation AUC from 0.70 (linear probe, head only) to 0.87.
- **Plain cross-entropy beat the GenD loss at our batch size.** With batch 64 (GenD uses 512), cross-entropy plus
  alignment and uniformity terms peaked after 1-2 epochs (0.73), while cross-entropy alone kept improving (0.84 at
  lr 1e-4, 0.87 at lr 3e-4 on the development set).
- **Light augmentation won.** Heavier augmentation (robust, strong) scored lower on every validation view.
- **Bigger backbone, better transfer.** ViT-L/14 beat ViT-B/16 by 9 AUC points on Celeb-DF v2 and the CNN baseline
  by 13 points, although all three were close on DFDC itself.
- The ensemble was best on validation but not on the test sets, so ViT-L/14 alone is the default.

---

## Try it online (GitHub Codespaces)

No GPU or install needed; runs on the CPU, so each video takes noticeably longer than on a GPU.

1. On this repository, click **Code -> Codespaces -> Create codespace on main**.
2. Wait for the first setup to finish (about 10-15 minutes: CPU PyTorch, packages, ViT-L/14 download).
   The app then starts by itself on port 8000.
3. Open the **Ports** tab and open port 8000. To share it, right-click it and choose
   **Port visibility -> Public**.

If the app is not running, start it from the codespace terminal with `bash .devcontainer/start.sh`
(log: `/tmp/unmask.log`). EfficientNet-B4 and the ensemble appear only when `weights/effb4.pt` is present (not
included in the repository because of its size).

---

## Run it locally

Tested on Windows with Python 3.12 and an NVIDIA GPU (CPU works, slower).

```powershell
git clone https://github.com/ktk-007/deepfake-detection-ai.git
cd deepfake-detection-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
pip install facenet-pytorch==2.6.0 --no-deps
python scripts\run_app.py               # GPU if available
python scripts\run_app.py --device cpu  # CPU only
```

Then open http://127.0.0.1:8000. The first start downloads the CLIP backbone (about 1.7 GB for ViT-L/14).
Install `facenet-pytorch` with `--no-deps`, otherwise pip replaces your torch version. Uploads: MP4, MOV or AVI, up
to 100 MB. Uploaded videos are deleted as soon as their frames are read.

---

## Reproduce the experiments

Run from the project folder with the virtual environment active, one GPU job at a time.

```powershell
python scripts\process_batch.py                                   # extract face crops from DFDC zips in "zip data\"
python scripts\train.py --name e7_l14 --config configs\e7_l14.yaml --epochs 4 --lr 2e-4
python scripts\eval_val.py --run e7_l14                           # clean / tight / degraded views of part 08
python scripts\run_queue.py configs\queue_finals.yaml             # or run the final queue in order
python scripts\phase4_val.py                                      # lock every choice on part 08
python scripts\phase4_test.py                                     # score each test set once, write the final table
python scripts\app_check.py                                       # time the app on held-out demo videos
```

`scripts\prepare_celebdf.py` and `scripts\prepare_wild.py` prepare the two outside test sets. Tests:
`python -m pytest`.

---

## Repository layout

```
unmask/
  app/        web app: FastAPI server, analysis pipeline, report, static page
  data/       DFDC zip streaming, frame extraction, face detection and cropping, quality gates, datasets
  models/     CLIP + LayerNorm tuning, linear-norm head, EfficientNet-B4, losses
  train/      training loop and checkpoints
  eval/       metrics, validation views, Phase 4 evaluation
  explain/    Grad-CAM
scripts/      command-line entry points (training, evaluation, data preparation, app)
configs/      one YAML per experiment, plus the final queue
results/      results table, final table, calibration, ensemble choice, Phase 4 scores, ROC plots
weights/      final trained checkpoints (CLIP runs store only the trained LayerNorms and head)
logs/         logs of the runs behind the results
tests/        pytest suite
.devcontainer/ GitHub Codespaces setup
```

---

## Limitations

- Trained on DFDC only. Accuracy drops on other datasets (0.91-0.93 AUC outside vs 0.98 on DFDC), and newer
  generation methods may score lower still.
- Built for face swaps in videos with a visible face. It does not check audio, fully synthetic scenes or images.
- Follows one face per video; videos with several people are judged on the main face only.
- Strong compression and low resolution reduce accuracy (see the degraded validation view).
- A research and learning project. Do not use its verdict as the only evidence about a real person or event.

---

## Acknowledgements

- [GenD](https://github.com/yermandy/GenD): the LayerNorm-tuning recipe this project follows, and the published
  checkpoint used as a reference row.
- [OpenAI CLIP](https://github.com/openai/CLIP) weights through [open_clip](https://github.com/mlfoundations/open_clip),
  [timm](https://github.com/huggingface/pytorch-image-models) and
  [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (MTCNN).
- Datasets: DFDC (Deepfake Detection Challenge), Celeb-DF v2 and WildDeepfake, each under its own license.

---

## License

Code: [MIT](LICENSE). The datasets and pretrained backbones keep their own licenses.
