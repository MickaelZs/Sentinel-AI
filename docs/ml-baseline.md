# Logistic Regression baseline

The dataset is synthetic. This baseline is not a production fraud model. It is
the first reproducible supervised reference for future Sentinel AI experiments.

## Dataset and split

The canonical source is `generate_transactions(rows=10_000, seed=42)`, not a
CSV. The target is `is_fraud` where 1 means fraud. The 10,000 rows contain 148
fraud labels (1.48%).

Rows are ordered by `occurred_at` before a non-shuffled 70%/15%/15% split:

| Split | Rows | Fraud | Fraud rate | UTC period |
|---|---:|---:|---:|---|
| Train | 7,000 | 111 | 1.59% | 2025-01-01 00:54:48 to 2025-05-07 16:46:24 |
| Validation | 1,500 | 12 | 0.80% | 2025-05-07 17:05:32 to 2025-06-03 16:24:09 |
| Test | 1,500 | 25 | 1.67% | 2025-06-03 16:34:07 to 2025-06-29 23:55:15 |

The temporal boundaries do not overlap. Train is fitted once; validation and
test are only transformed and evaluated. The test split is not used to choose
features, class weight, threshold, or hyperparameters.

## Features and preprocessing

Numeric candidates are `amount`, `account_age_days`,
`transactions_last_24h`, `avg_amount_last_30d`, `device_age_days`,
`distance_from_home_km`, and `hour`. They use `StandardScaler` fitted only on
train data.

Categorical candidates are `currency`, `merchant_category`, `country`,
`transaction_type`, `channel`, and `is_international`. They use
`OneHotEncoder(handle_unknown="ignore")`, also fitted only on train.

`external_id`, `customer_id`, `device_id`, `occurred_at`, and `is_fraud` are
excluded. IDs risk memorization; `occurred_at` is reserved for ordering and
audit; `is_fraud` is the target. The fitted transformer produces 29 features.

## Baseline configuration

The pipeline is `ColumnTransformer` followed by `LogisticRegression` with
`class_weight="balanced"`, `solver="lbfgs"`, `max_iter=1000`, and
`random_state=42`. Class weighting is an explicit choice to make the minority
class visible without sampling. There is no feature engineering, tuning,
resampling, calibration, or persisted model.

The decision threshold is fixed at 0.50. It was not optimized and is not claimed
to be appropriate for production. PR-AUC below means Average Precision, computed
from probabilities; ROC-AUC also uses probabilities.

## Results

| Split | Precision | Recall | F1 | ROC-AUC | PR-AUC / AP | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 0.0344 | 0.6486 | 0.0654 | 0.7470 | 0.0417 | 4,870 | 2,019 | 39 | 72 |
| Validation | 0.0126 | 0.5000 | 0.0245 | 0.5882 | 0.0136 | 1,016 | 472 | 6 | 6 |
| Test | 0.0356 | 0.6800 | 0.0676 | 0.7159 | 0.0432 | 1,014 | 461 | 8 | 17 |

The always-legitimate trivial baseline has 98.33% test accuracy and 0.00 fraud
recall. This illustrates why accuracy is not a selection metric for this
imbalanced task.

Train PR-AUC is higher than validation/test, while test ranking and recall are
closer to train. Validation has only 12 positives, so its metrics are naturally
volatile. These results are observations on one synthetic experiment, not claims
of real-world fraud performance.

## Sanity checks and next steps

The model converged without `ConvergenceWarning`; coefficients and probabilities
are finite; probabilities are within [0, 1]; and the pipeline produces no NaNs.

Future work may investigate validation-only threshold selection, temporal
robustness, feature hypotheses from EDA, calibration, and other model families.
Those decisions are intentionally outside this baseline.
