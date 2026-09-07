# Tree-based model comparison

All experiments use synthetic data. These results do not establish production
fraud-detection performance.

## Experimental question and protocol

This experiment asks whether fixed, non-linear tree models improve ranking and
fraud detection over the Stage 6 Logistic Regression baseline. The canonical
dataset is generated in memory with `generate_transactions(rows=10_000,
seed=42)`. No CSV, label, distribution, or feature was changed.

All models use a non-shuffled chronological 70%/15%/15% split by `occurred_at`:

| Split | Rows | Fraud | Fraud rate | UTC period |
|---|---:|---:|---:|---|
| Train | 7,000 | 111 | 1.59% | 2025-01-01 00:54:48 to 2025-05-07 16:46:24 |
| Validation | 1,500 | 12 | 0.80% | 2025-05-07 17:05:32 to 2025-06-03 16:24:09 |
| Test | 1,500 | 25 | 1.67% | 2025-06-03 16:34:07 to 2025-06-29 23:55:15 |

The models use the same numeric features: `amount`, `account_age_days`,
`transactions_last_24h`, `avg_amount_last_30d`, `device_age_days`,
`distance_from_home_km`, and `hour`. Categorical features are `currency`,
`merchant_category`, `country`, `transaction_type`, `channel`, and
`is_international`.

`external_id`, `customer_id`, `device_id`, `occurred_at`, and `is_fraud` are
excluded from the feature matrix. The target is `is_fraud`. Each pipeline fits
only on train; validation and test are transformed and evaluated without refit.
The test partition was not used to select a model, threshold, features, or
parameters.

## Fixed configurations

| Model | Preprocessing | Estimator |
|---|---|---|
| Logistic Regression | `StandardScaler` numeric; one-hot categorical | `class_weight="balanced"`, `solver="lbfgs"`, `max_iter=1000`, `random_state=42` |
| Random Forest | numeric passthrough; dense one-hot categorical | `n_estimators=300`, `max_depth=8`, `min_samples_leaf=5`, `class_weight="balanced"`, `random_state=42`, `n_jobs=-1` |
| HistGradientBoosting | numeric passthrough; dense one-hot categorical | `learning_rate=0.05`, `max_iter=200`, `max_leaf_nodes=15`, `min_samples_leaf=20`, `class_weight="balanced"`, `random_state=42` |

The installed scikit-learn API supports `class_weight` for
`HistGradientBoostingClassifier`, so weighting is learned from train labels by
the estimator. No sampling, tuning, feature engineering, calibration, or model
persistence occurs.

All metrics below use a fixed 0.50 threshold for classes. It is not optimized.
ROC-AUC and PR-AUC / Average Precision (AP) are probability-ranking metrics;
precision, recall, F1, and confusion matrices depend on that threshold. The
positive prevalence in each split is the approximate AP reference for a random
ranking.

## Results

| Split / model | Precision | Recall | F1 | ROC-AUC | PR-AUC / AP |
|---|---:|---:|---:|---:|---:|
| Train / Logistic Regression | 0.0344 | 0.6486 | 0.0654 | 0.7470 | 0.0417 |
| Train / Random Forest | 0.2126 | 0.9459 | 0.3471 | 0.9902 | 0.9099 |
| Train / HistGradientBoosting | 0.4286 | 1.0000 | 0.6000 | 1.0000 | 0.9986 |
| Validation / Logistic Regression | 0.0126 | 0.5000 | 0.0245 | 0.5882 | 0.0136 |
| Validation / Random Forest | 0.0187 | 0.1667 | 0.0336 | 0.5289 | 0.0304 |
| Validation / HistGradientBoosting | 0.0000 | 0.0000 | 0.0000 | 0.5186 | 0.0086 |
| Test / Logistic Regression | 0.0356 | 0.6800 | 0.0676 | 0.7159 | 0.0432 |
| Test / Random Forest | 0.0189 | 0.0800 | 0.0305 | 0.7243 | 0.0339 |
| Test / HistGradientBoosting | 0.0392 | 0.0800 | 0.0526 | 0.6844 | 0.0328 |

Validation deltas from Logistic Regression are -0.0593 ROC-AUC and +0.0169 AP
for Random Forest, and -0.0696 ROC-AUC and -0.0050 AP for
HistGradientBoosting. On test, Random Forest has +0.0084 ROC-AUC and -0.0093
AP; HistGradientBoosting has -0.0315 ROC-AUC and -0.0104 AP.

The validation confusion matrices `(TN, FP, FN, TP)` are Logistic Regression
`(1016, 472, 6, 6)`, Random Forest `(1383, 105, 10, 2)`, and
HistGradientBoosting `(1458, 30, 12, 0)`. Test matrices are Logistic Regression
`(1014, 461, 8, 17)`, Random Forest `(1371, 104, 23, 2)`, and
HistGradientBoosting `(1426, 49, 23, 2)`.

## Generalization and conclusion

Both tree models have near-perfect train ranking but much weaker validation
ranking, a clear possible overfitting signal. Random Forest has the highest
validation AP, but Logistic Regression has the highest validation ROC-AUC.
Their test results are also mixed: Random Forest slightly improves ROC-AUC but
not AP, while HistGradientBoosting is lower on both ranking metrics.

With only 12 frauds in validation, these comparisons are highly variable. The
promotion status is **inconclusive**: neither challenger has sufficiently
consistent validation and test evidence to displace the baseline. This is an
experimental conclusion, not a production decision.

Random Forest's top transformed impurity importances are `numeric__amount`,
`numeric__hour`, `numeric__account_age_days`,
`numeric__avg_amount_last_30d`, `numeric__distance_from_home_km`,
`numeric__device_age_days`, `categorical__is_international_False`,
`categorical__country_BR`, `categorical__currency_BRL`, and
`categorical__is_international_True`. Impurity importance is descriptive only;
it does not imply causal fraud factors or justify feature removal.

Future work can investigate validation-only threshold selection, feature
hypotheses, calibration, and additional temporal evaluations. Those activities
are intentionally outside this fixed comparison.
