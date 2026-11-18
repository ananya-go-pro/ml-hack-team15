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
models/
  selected_untuned_model.joblib selected candidate classifier (before Bayesian tuning)
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

## Model comparison

Each GPS observation (sampled every 5 s) gets candidate road arcs within 50 m. Each (observation, candidate) pair is labelled 1 if it is the reference arc. The labels come from a monotone alignment of the GPS track to `.route`.

Five classifiers from different families are compared with 5-fold `GroupKFold` cross-validation (grouped by trip) on the training trips, then evaluated on the student-test trips:

- Logistic Regression
- distance-weighted KNN
- Random Forest
- LightGBM
- MLP

The primary metric is observation-level Top-1 accuracy. Macro F1, log loss, route continuity and timing are also reported. Training trips whose reference route does not match the GPS track (median distance > 50 m) are excluded from fitting; see `model_comparison/outputs/excluded_trips.json`.
