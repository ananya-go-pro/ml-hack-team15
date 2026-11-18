# Accuracy log: from 88.8% to 95.8% Top-1

Every step below was **decided on training data only** (Top-1 on out-of-fold predictions from
5-fold `GroupKFold` over the 78 usable training trips). Student-test numbers are reported for each
step but were never used to accept or reject a change. The evaluator partition is never read.

Reproduce:

```bash
python experiments/error_analysis.py      # where the errors come from (training OOF)
python hpo/bayes_opt_lightgbm.py          # Bayesian HPO, writes models/tuned_lightgbm.joblib
python hmm_decoding/run_hmm_experiments.py --model tuned
python experiments/accuracy_ladder.py     # re-runs every step below -> outputs/accuracy_ladder.csv
```

## The ladder

| Step | Change | Train OOF Top-1 (decides) | Student-test Top-1 (reported) | Worst student trip | Route continuity | Decision |
|---|---|---|---|---|---|---|
| E0 | Baseline: untuned LightGBM, 17 features, classifier-only | 0.8763 | 0.8879 | 0.7356 | 0.9587 | baseline |
| E1 | + HMM / Viterbi decoding, default β = 10 m | 0.9267 | 0.9450 | 0.8519 | 0.9773 | **keep** (+5.0 OOF) |
| E1b | Signed same-arc distance in transitions | 0.9255 | 0.9390 | 0.8301 | 0.9773 | reject (−0.1 OOF) |
| E2 | + β chosen on training OOF (β = 7) | 0.9313 | 0.9447 | 0.8493 | 0.9773 | keep (+0.5 OOF) |
| E3a | 23 features (wider heading/context window, twin flag), classifier-only | 0.8988 | 0.8989 | 0.7650 | 0.9574 | helps the classifier (+2.3 OOF)… |
| E3b | 23 features + HMM (β chosen on OOF) | 0.9299 | 0.9494 | 0.8748 | 0.9778 | …but not after HMM (−0.1 vs E2): reject |
| E4a | Bayesian-tuned LightGBM (Optuna TPE, 40 trials), classifier-only | 0.8795 | 0.8853 | 0.7254 | 0.9587 | better on every CV metric |
| **E4b** | **Bayesian-tuned LightGBM + HMM (β = 2 chosen on OOF)** | **0.9310** | **0.9583** | **0.8896** | **0.9775** | **final pipeline** |

Notes, so nobody over-reads the table:

* **E2 and E4b are tied on the decision metric** (0.9313 vs 0.9310). We use the tuned model because
  it is better on every classifier-level CV metric (Top-1 0.8744→0.8775, Macro F1 0.9276→0.9294,
  log loss 0.0849→0.0838) and the brief makes the Bayesian-tuned model the final one. The 1.4-point
  student-test gap between E2 and E4b should **not** be credited to tuning: 3,836 student-test
  observations give a binomial 95% interval of about ±0.6 points, and observations inside a trip are
  correlated, so the real uncertainty is larger.
* **E3b scored higher than E2 on student-test but was rejected**, because the decision is made on
  training OOF. Choosing it would have been choosing on student-test.
* The single change that matters is the HMM (E1). Everything else moves the result by tenths of a point.

## Why the HMM helps so much: error analysis (training OOF)

`outputs/error_categories_train_oof.csv`, share of all observations:

| Error type | Classifier only | + HMM |
|---|---|---|
| reverse twin (right road, wrong direction) | 5.23% | 2.12% |
| other (parallel road, 2+ hops away) | 2.21% | 1.06% |
| adjacent arc (just before/after a junction) | 1.87% | 1.76% |
| shares a node (other arc at the same junction) | 1.51% | 0.56% |
| no candidate within 50 m | 0.91% | 0.91% |
| reference arc > 30 m from the GPS fix | 0.64% | 0.57% |

* Reverse-twin errors are almost all at low speed (`outputs/twin_errors_by_speed.csv`: 23–31% error rate
  below 2 m/s, 0.1% above 8 m/s). These are stops and crawling traffic inside car trips. At that speed a ±2 s heading window moves less than the GPS noise,
  so the heading feature cannot see the direction. The HMM fixes most of them because driving forward
  along A→B is cheap in network distance while the twin B→A needs a U-turn.
* For the reverse-twin errors that survive the HMM *while the vehicle is clearly moving*, the GPS moves
  **against the reference arc's direction in 67.6% of cases** (vs 0.9% for all other observations,
  `outputs/reference_direction_check.csv`). In those cases the model agrees with the physical motion and
  the reference label is the inconsistent part. We treat this as label noise and do not try to learn it.

## What is left, and what we would try next (in order)

The remaining errors are mostly at junctions (adjacent / shares-node, ~2.3%) and slow-speed direction
ambiguity (~2.1%), part of which is label noise.

1. **Train on decoded sequences, not single rows.** Tune β and the emission weight jointly with the
   classifier (or use the HMM posterior as a feature in a second-stage model) — still on training OOF.
2. **Junction handling.** Add the projection position on the *next* arc in the route graph as a feature,
   and a small transition bonus for continuing straight. Targets the 1.8% adjacent-arc errors.
3. **Denser decoding.** Decode on every 2nd fix instead of every 5th and report on the same 5 s
   observations. More steps give the HMM more evidence at low speed.
4. **Report a label-noise-aware score** alongside Top-1 (e.g. count a junction-adjacent prediction within
   10 m of the node as correct) so the evaluator can see how much of the remaining error is ambiguity.

None of these use student-test or evaluator data to make decisions.
