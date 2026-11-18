# PathSnap: Distributed Map-Matching & Route Reconstruction

## Installation

Requires Python 3 and pip.

```bash
git clone https://github.com/ananya-go-pro/ml-hack-team15.git
cd ml-hack-team15
pip install -r requirements.txt
```

## Execution

Run from the repository root, in this order:

```bash
# 1. Preprocessing: raw -> data/processed/
python preprocessing/inspect_dataset.py
python preprocessing/preprocess.py
python preprocessing/validate_data.py

# 2. Exploratory data analysis
jupyter nbconvert --to notebook --execute --inplace eda/eda.ipynb

# 3. Model comparison (about 6 min)
jupyter nbconvert --to notebook --execute --inplace model_comparison/model_comparison.ipynb

# 4. Bayesian hyperparameter optimisation (about 8 min, writes models/tuned_lightgbm.joblib)
python hpo/bayes_opt_lightgbm.py --trials 40

# 5. HMM / Viterbi route reconstruction (about 4 min, writes hmm_decoding/outputs/hmm_config.json and maps)
python hmm_decoding/run_hmm_experiments.py --model tuned

# 6. Error analysis and accuracy ladder (about 10 min)
python experiments/error_analysis.py
python experiments/accuracy_ladder.py
```

The notebooks can also be opened in Jupyter and run top to bottom.

## Deployment

Both laptops must be on the same network, with the repository cloned on each.

**Laptop 1 (API).** Install the requirements, put the evaluation trips in one folder (one sub-folder per trip with
`.track`, `.nodes`, `.arcs`, `.route`) and start the API:

```bash
python api/main.py --trips-dir /path/to/evaluator_test --host 0.0.0.0 --port 8000
```

Find the laptop's IP address (`ipconfig getifaddr en0` on macOS, `ipconfig` on Windows, `hostname -I` on Linux) and
allow incoming connections on port 8000 if the firewall asks. Requests are logged in this terminal.

**Laptop 2 (UI).** Requires only Python 3 and a browser with internet access (for map tiles):

```bash
python ui/serve.py --api http://<laptop-1-ip>:8000
```

This opens `http://localhost:8501`. Select a trip to run it. The API address can also be changed on the page or
passed as `http://localhost:8501/?api=http://<laptop-1-ip>:8000`.

**Single-laptop check** using the student-test trips:

```bash
python api/main.py --trips-dir "Student folder/student_test" --port 8000
python ui/serve.py --api http://localhost:8000
```
