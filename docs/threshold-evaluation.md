# Validation-based threshold evaluation

All experiments use synthetic data. These thresholds are not a production fraud
policy and do not establish real financial performance.

## Ranking versus operational decisions

ROC-AUC and Average Precision (AP, reported as PR-AUC) measure how well a model
ranks observations using probabilities. They do not change when a classification
threshold changes. Precision, recall, F1, confusion-matrix cells, and alert
volume do change with the threshold, so they express an operational policy.

The canonical dataset is generated in memory with
`generate_transactions(rows=10_000, seed=42)`. The frozen Stage 7 models use
the same chronological 70%/15%/15% split: 7,000 train rows (111 frauds), 1,500
validation rows (12 frauds), and 1,500 test rows (25 frauds). Models fit only
on train.

For each model, validation probabilities produce in-memory precision-recall and
ROC curve arrays. No curve output, prediction file, or model artifact is saved.

## Policy frozen before test

The predefined experimental policy is: among validation thresholds with recall
at least 0.60, choose the highest precision. Ties within numerical tolerance
prefer higher recall and then the higher threshold. If no candidate qualifies,
the explicit fallback chooses highest recall, then precision, then threshold,
and records `constraint_satisfied = false`.

`0.60` is an experimental recall constraint, not a real banking requirement.
In production it would depend on fraud loss, investigation capacity, false
positive burden, regulation, and product risk.

Thresholds were selected using validation data only. Test labels were not used
in threshold selection. Each selected threshold was frozen before one final
test classification evaluation.

## Ranking metrics

| Model | Validation ROC-AUC | Validation AP | Test ROC-AUC | Test AP |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.5882 | 0.0136 | 0.7159 | 0.0432 |
| Random Forest | 0.5289 | 0.0304 | 0.7243 | 0.0339 |
| HistGradientBoosting | 0.5186 | 0.0086 | 0.6844 | 0.0328 |

These match the Stage 7 ranking results. Validation AP and ROC-AUC remain mixed
across models, and validation has only 12 positives.

## Logistic Regression

At validation threshold 0.50, precision/recall/F1 are 0.0126/0.5000/0.0245,
with 478 alerts (31.87%) and `(TN, FP, FN, TP) = (1016, 472, 6, 6)`.

The validation-selected threshold is `0.292472`; it satisfies the constraint.
Validation precision/recall/F1 are 0.0106/0.9167/0.0209 with 1,041 alerts
(69.40%) and `(458, 1030, 1, 11)`.

Frozen test results at that threshold are precision 0.0212, recall 0.8800, F1
0.0414, 1,038 alerts (69.20%), and `(459, 1016, 3, 22)`.

## Random Forest

At validation threshold 0.50, precision/recall/F1 are 0.0187/0.1667/0.0336,
with 107 alerts (7.13%) and `(1383, 105, 10, 2)`.

The validation-selected threshold is `0.092929`; it satisfies the constraint.
Validation precision/recall/F1 are 0.0085/1.0000/0.0168 with 1,417 alerts
(94.47%) and `(83, 1405, 0, 12)`.

Frozen test results are precision 0.0177, recall 1.0000, F1 0.0348, 1,412
alerts (94.13%), and `(88, 1387, 0, 25)`.

## HistGradientBoosting

At validation threshold 0.50, precision/recall/F1 are 0.0000/0.0000/0.0000,
with 53 alerts (3.53%) and `(1435, 53, 12, 0)`.

The validation-selected threshold is `0.024855`; it satisfies the constraint.
Validation precision/recall/F1 are 0.0103/0.8333/0.0203 with 975 alerts
(65.00%) and `(523, 965, 2, 10)`.

Frozen test results are precision 0.0245, recall 0.9600, F1 0.0478, 980 alerts
(65.33%), and `(519, 956, 1, 24)`.

## Operational comparison on frozen test rules

| Model | Threshold | Precision | Recall | F1 | Alerts | Alert rate | Alerts/TP | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.292472 | 0.0212 | 0.8800 | 0.0414 | 1,038 | 69.20% | 47.18 | 1,016 | 3 | 22 |
| Random Forest | 0.092929 | 0.0177 | 1.0000 | 0.0348 | 1,412 | 94.13% | 56.48 | 1,387 | 0 | 25 |
| HistGradientBoosting | 0.024855 | 0.0245 | 0.9600 | 0.0478 | 980 | 65.33% | 40.83 | 956 | 1 | 24 |

For comparison, all selected thresholds are lower than 0.50. Recall rises, but
precision falls or remains very low and alert burden rises sharply. F1 is still
reported, but it weights precision and recall symmetrically and does not encode
real fraud-review costs.

## Relative-cost sensitivity

These illustrative scores do not select thresholds and are not monetary values.

| Model | FP=1, FN=10 | FP=1, FN=25 |
|---|---:|---:|
| Logistic Regression | 1,046 | 1,091 |
| Random Forest | 1,387 | 1,387 |
| HistGradientBoosting | 966 | 981 |

## Generalization and status

The frozen rules preserve high recall on test, but do so by alerting 65% to 94%
of transactions. With only 12 validation frauds and mixed validation ranking,
this is not stable enough to promote a challenger. Promotion status remains
**inconclusive**.

No retraining, estimator tuning, feature changes, resampling, calibration, or
test-guided threshold adjustment occurred. Future work may define a business
cost model and evaluate threshold policies on larger, representative temporal
data. Those tasks are outside this stage.
