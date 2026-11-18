"""Bayesian hyperparameter optimisation of the selected classifier (LightGBM).

Sampler   : Optuna TPE (Tree-structured Parzen Estimator), seed 42.
Objective : mean 5-fold GroupKFold (grouped by trip) observation-level Top-1 accuracy
            on the training trips, the same primary metric used to select the model.
Data      : training partition only (minus the two excluded trips). The student-test
            partition is used only after tuning, to compare tuned vs untuned.
            The evaluator partition is never read.

Run from the repo root:  python hpo/bayes_opt_lightgbm.py [--trials 40]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.candidates import DEFAULT_CFG, FEATURES_V1  # noqa: E402
from pathsnap.dataset import build_split, excluded_train_trips, stack  # noqa: E402
from pathsnap.metrics import classifier_metrics, sequence_metrics, top1_from_proba  # noqa: E402

SEED = 42
OUT = Path(__file__).resolve().parent / "outputs"
MODELS = Path(__file__).resolve().parents[1] / "models"
FEATURES = FEATURES_V1

UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8, min_child_samples=20, reg_lambda=0.0, reg_alpha=0.0)

SEARCH_SPACE = {
    "n_estimators":      ("int_log", 100, 1500),
    "learning_rate":     ("float_log", 0.01, 0.2),
    "num_leaves":        ("int_log", 15, 255),
    "min_child_samples": ("int_log", 5, 300),
    "subsample":         ("float", 0.5, 1.0),
    "colsample_bytree":  ("float", 0.4, 1.0),
    "reg_lambda":        ("float_log", 1e-3, 10.0),
    "reg_alpha":         ("float_log", 1e-3, 10.0),
}


def make_model(params):
    return lgb.LGBMClassifier(**dict(params, subsample_freq=1), random_state=SEED, n_jobs=-1, verbose=-1)


def cv_scores(params, c, meta, folds):
    top1, mf1, ll = [], [], []
    X, y = c[FEATURES].values, c["y"].values
    oof = np.zeros(len(c))
    for tr_i, va_i in folds:
        m = make_model(params).fit(X[tr_i], y[tr_i])
        p = m.predict_proba(X[va_i])[:, 1]
        oof[va_i] = p
        va = c.iloc[va_i]
        vmeta = meta[meta.trip.isin(va.trip.unique())]
        top1.append(sequence_metrics(top1_from_proba(va, p), vmeta, "train", continuity=False)[0]["top1"])
        cm = classifier_metrics(va, p)
        mf1.append(cm["macro_f1"]); ll.append(cm["log_loss"])
    return dict(cv_top1=np.mean(top1), cv_top1_std=np.std(top1, ddof=1), cv_macro_f1=np.mean(mf1),
                cv_macro_f1_std=np.std(mf1, ddof=1), cv_log_loss=np.mean(ll)), oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    trips = build_split("train")
    bad = excluded_train_trips()
    c, meta = stack(trips, [t for t in trips if t not in bad])
    folds = list(GroupKFold(n_splits=5).split(c, c["y"], c["trip"]))

    def objective(trial):
        params = {}
        for k, (kind, lo, hi) in SEARCH_SPACE.items():
            if kind == "int_log":
                params[k] = trial.suggest_int(k, lo, hi, log=True)
            elif kind == "float_log":
                params[k] = trial.suggest_float(k, lo, hi, log=True)
            else:
                params[k] = trial.suggest_float(k, lo, hi)
        t0 = time.perf_counter()
        s, _ = cv_scores(params, c, meta, folds)
        for k, v in s.items():
            trial.set_user_attr(k, float(v))
        trial.set_user_attr("cv_seconds", time.perf_counter() - t0)
        print(f"trial {trial.number:3d}  CV Top-1 {s['cv_top1']:.4f}  macroF1 {s['cv_macro_f1']:.4f}  "
              f"logloss {s['cv_log_loss']:.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        return s["cv_top1"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="lightgbm_cv_top1")
    t_start = time.perf_counter()
    study.optimize(objective, n_trials=args.trials)
    hpo_seconds = time.perf_counter() - t_start

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))
    trials.to_csv(OUT / "optuna_trials.csv", index=False)
    best = dict(study.best_params)
    json.dump(dict(sampler="TPESampler(seed=42)", objective="mean 5-fold GroupKFold CV observation-level Top-1",
                   n_trials=args.trials, total_seconds=hpo_seconds, search_space=SEARCH_SPACE,
                   untuned=UNTUNED, best_params=best, best_cv_top1=study.best_value),
              open(OUT / "best_params.json", "w"), indent=2)

    # ---- tuned vs untuned on identical folds and the same student-test partition ----
    te = build_split("student_test")
    tc, tm = stack(te)
    rows, fitted = [], {}
    for name, params in [("untuned", {k: UNTUNED[k] for k in SEARCH_SPACE}), ("tuned", best)]:
        cvs, _ = cv_scores(params, c, meta, folds)
        t0 = time.perf_counter(); m = make_model(params).fit(c[FEATURES].values, c["y"].values)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict_proba(tc[FEATURES].values)[:, 1]
        inf_s = time.perf_counter() - t0
        st, _ = sequence_metrics(top1_from_proba(tc, p), tm, "student_test")
        cm = classifier_metrics(tc, p)
        rows.append(dict(model=f"LightGBM ({name})", **cvs, test_top1=st["top1"], test_macro_f1=cm["macro_f1"],
                         test_log_loss=cm["log_loss"], test_route_continuity=st["route_continuity"],
                         fit_s=fit_s, inference_s=inf_s))
        fitted[name] = m
        print(rows[-1], flush=True)
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "tuned_vs_untuned.csv", index=False)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(dict(model=fitted["tuned"], features=FEATURES, config=DEFAULT_CFG, name="LightGBM (tuned)",
                     params=best), MODELS / "tuned_lightgbm.joblib")

    # ---- plots ----
    vals = trials["value"].values
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(vals, "o", ms=4, alpha=.6, label="trial CV Top-1")
    ax[0].plot(np.maximum.accumulate(vals), "r-", label="best so far")
    ax[0].axhline(comp.loc[0, "cv_top1"], ls="--", c="grey", label="untuned (same folds)")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV Top-1"); ax[0].set_title("Optuna TPE optimisation history")
    ax[0].legend()
    x = np.arange(len(comp)); w = .2
    for i, (col, lab) in enumerate([("cv_macro_f1", "CV Macro F1"), ("cv_top1", "CV Top-1"),
                                    ("test_top1", "Student-test Top-1"), ("test_macro_f1", "Student-test Macro F1")]):
        ax[1].bar(x + (i - 1.5) * w, comp[col], w, label=lab)
    ax[1].set_xticks(x, comp["model"]); ax[1].set_ylim(.8, 1); ax[1].legend(fontsize=8)
    ax[1].set_title("Untuned vs tuned (same folds, same student-test trips)")
    plt.tight_layout(); plt.savefig(OUT / "hpo_history_and_comparison.png", dpi=120)
    try:
        imp = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(7, 4)); plt.barh(list(imp)[::-1], list(imp.values())[::-1])
        plt.title("Hyperparameter importance (fANOVA)"); plt.tight_layout()
        plt.savefig(OUT / "hpo_param_importance.png", dpi=120)
    except Exception as e:  # importance needs sklearn; not critical
        print("param importance skipped:", e)
    print("done in", round(hpo_seconds), "s. best", best)


if __name__ == "__main__":
    main()
