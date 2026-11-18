# PathSnap: Distributed Map-Matching & Route Reconstruction
**Team 15** | PES University — Machine Learning Hackathon (UE24CS352A)

PathSnap matches noisy GPS observations to a road network and reconstructs the route a vehicle actually drove.

## Repository layout

```
data/processed/                 cleaned data (output of preprocessing)
  train/<trip>_{track,nodes,arcs,route}.csv          80 trips
  student_test/<trip>_{track,nodes,arcs,route}.csv   10 trips
  preprocessing_summary.csv
preprocessing/
  inspect_dataset.py            raw file counts / sample contents
  preprocess.py                 raw -> cleaned CSVs (both partitions)
  validate_data.py              checks every cleaned CSV
eda/
  eda.ipynb                     dataset exploration, anomaly register, plots, Folium map
  outputs/                      EDA tables, plots, interactive map
feature_engineering/            per-trip GPS features, road graph, candidate generation, maps
model_comparison/
  model_comparison.ipynb        labelling, candidate features, 5-model comparison and selection
  outputs/                      comparison tables, fold logs, configs, plots, Folium map
pathsnap/                       shared code used by every script below (and later by the API)
  data.py                       loading cleaned trips, local metric projection
  candidates.py                 reference labelling, candidate generation, candidate features
  hmm.py                        HMM emissions/transitions + Viterbi decoding
  metrics.py, dataset.py        Top-1 / continuity / route metrics, cached partition builder
hpo/
  bayes_opt_lightgbm.py         Bayesian (Optuna TPE) tuning of the selected LightGBM
  outputs/                      all trials, best params, tuned-vs-untuned table, plots
hmm_decoding/
  run_hmm_experiments.py        decoder tuning on training OOF, classifier vs classifier+HMM, Folium maps
  outputs/                      comparison tables, per-trip results, hmm_config.json, maps/
experiments/
  README.md                     accuracy log: every change we tried and why it was kept or rejected
  error_analysis.py             error categories on training out-of-fold predictions
  accuracy_ladder.py            re-runs every experiment step on the same partitions
  outputs/
models/
  selected_untuned_model.joblib selected candidate classifier (before Bayesian tuning)
  tuned_lightgbm.joblib         Bayesian-tuned LightGBM used by the final pipeline
report/
  report.html                   technical report source (figures are read from the output folders)
  PathSnap_Technical_Report_Team15.pdf
```

## Setup

```bash
pip install -r requirements.txt
```

The raw dataset is **not** committed. Place it so that it contains `train/` and `student_test/`. Any of these locations is found automatically: `data/raw/`, `data/raw/Student folder/`, `Student folder/` in the repo, or `../Student folder/` next to the repo. You can also point to it with `export PATHSNAP_DATA=/path/to/folder`. The evaluator-testing partition is never read by any script or notebook.

## Running the pipeline

Run these from the repository root:

```bash
# 1. Preprocessing: raw -> data/processed/
python preprocessing/inspect_dataset.py
python preprocessing/preprocess.py
python preprocessing/validate_data.py

# 2. Exploratory data analysis (reads raw data)
jupyter nbconvert --to notebook --execute --inplace eda/eda.ipynb

# 3. Model comparison (reads data/processed/ only)
jupyter nbconvert --to notebook --execute --inplace model_comparison/model_comparison.ipynb
```

You can also open the notebooks in Jupyter and run them top to bottom. The model comparison writes a local feature cache to `cache/` (git-ignored) and takes about 6 minutes on a laptop.

```bash
# 4. Bayesian hyperparameter optimisation (~8 min, writes models/tuned_lightgbm.joblib)
python hpo/bayes_opt_lightgbm.py --trials 40

# 5. HMM / Viterbi route reconstruction (~4 min, writes hmm_config.json and Folium maps)
python hmm_decoding/run_hmm_experiments.py --model tuned

# 6. Documentation of every accuracy step (~10 min)
python experiments/error_analysis.py
python experiments/accuracy_ladder.py

# 7. Report PDF (headless Chrome)
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer \
  --print-to-pdf=report/PathSnap_Technical_Report_Team15.pdf report/report.html
```

## Model comparison

Each GPS observation (sampled every 5 s) gets candidate road arcs within 50 m. Each (observation, candidate) pair is labelled 1 if it is the reference arc. The labels come from a monotone alignment of the GPS track to `.route`.

Five classifiers from different families are compared with 5-fold `GroupKFold` cross-validation (grouped by trip) on the training trips, then evaluated on the student-test trips:

- Logistic Regression
- distance-weighted KNN
- Random Forest
- LightGBM
- MLP

The primary metric is observation-level Top-1 accuracy. Macro F1, log loss, route continuity and timing are also reported. Training trips whose reference route does not match the GPS track (median distance > 50 m) are excluded from fitting; see `model_comparison/outputs/excluded_trips.json`.

## Results so far (student-testing trips, 3,836 observations)

| Pipeline | Top-1 | Route continuity | Worst trip |
|---|---|---|---|
| Nearest arc (no learning) | 0.5360 | 0.9077 | – |
| Untuned LightGBM, classifier only | 0.8879 | 0.9587 | 0.7356 |
| Bayesian-tuned LightGBM, classifier only | 0.8853 | 0.9587 | 0.7254 |
| **Tuned LightGBM + HMM / Viterbi (final)** | **0.9583** | **0.9775** | **0.8896** |

All decisions (features, model, hyperparameters, decoder parameters) were made on out-of-fold predictions over the
training trips. The step-by-step log, including the ideas that did not work, is in `experiments/README.md`.
Student-test numbers are development-stage estimates on 10 trips, so expect roughly ±1 point on unseen trips.

## Progress against the hackathon work process

| # | Stage | Status |
|---|---|---|
| 1–2 | Understand the problem, explore the raw dataset | done (`eda/`) |
| 3 | Data processing and feature engineering | done (`preprocessing/`, `feature_engineering/`, `pathsnap/candidates.py`) |
| 4–5 | Train and compare 5 models, select one | done (`model_comparison/`), LightGBM selected |
| 6 | Bayesian hyperparameter optimisation | done (`hpo/`) |
| 7 | HMM and Viterbi route reconstruction | done (`pathsnap/hmm.py`, `hmm_decoding/`) |
| 8 | Export model and artifacts | done: `models/tuned_lightgbm.joblib`, `hmm_decoding/outputs/hmm_config.json` |
| 9 | FastAPI backend (Laptop 1) | **to do** |
| 10 | Evaluator UI (Laptop 2) | **to do** |
| 11 | Integration test on two laptops | **to do** |
| 12 | Technical report | draft in `report/`; section 7 (API/UI screenshots) pending |

## Academic integrity and AI use

* **Data partitions.** Only `train` (80 trips) and `student_test` (10 trips) are read by any code in this repository.
  The evaluator-testing partition is never loaded. Its ground truth must stay server-side and will only be used inside
  the API to compute accuracy after prediction, never as model input.
* **No selection on test data.** Features, models, hyperparameters and HMM parameters were all chosen using
  group-aware cross-validation / out-of-fold predictions on the training trips. Student-test results are reported
  but did not decide anything; `experiments/README.md` lists one case (E3b) where a change scored higher on
  student-test and was still rejected for that reason.
* **No hardcoding.** No predictions, routes or trip-specific rules are hardcoded anywhere. Two training trips whose
  reference route does not match their GPS track are excluded from fitting by a rule computed on training data
  (`model_comparison/outputs/excluded_trips.json`).
* **AI assistance.** The team used an AI coding assistant (Claude) during development to help write and refactor
  code (the shared `pathsnap/` package, the HPO, HMM and experiment scripts), to analyse errors and to draft report
  text. Every experiment was actually run, every number in the report comes from the output files in this
  repository, and the team has reviewed the code and results and is able to explain them.
