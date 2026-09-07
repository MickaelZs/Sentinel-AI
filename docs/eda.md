# Exploratory data analysis

This analysis describes synthetic data, not real financial behavior. It is
reproducible from `generate_transactions(rows=10_000, seed=42)` and does not use
the generated CSV as a source of truth.

## Quality and balance

The canonical dataset has 10,000 rows, 18 columns, no missing values, no
duplicate `external_id` values, no non-positive amounts, and no invalid
timestamps. Its UTC period is 2025-01-01 00:54:48 to 2025-06-29 23:55:15.

There are 148 fraud labels (1.48%) and 9,852 non-fraud labels. The imbalance
ratio is 66.57, defined as `non_fraud_count / fraud_count`. A future model that
always predicts non-fraud could appear highly accurate while detecting no fraud,
so accuracy alone would be unhelpful.

## Distributions and class differences

`amount` is right-skewed: mean 110.11, median 65.96, standard deviation 154.38,
minimum 1.34, maximum 4,382.58, p25 33.91, p75 128.11, p95 352.92, and p99
683.24. Fraud rows have a higher mean (153.31 versus 109.46), median (71.42
versus 65.84), and p95 (443.69 versus 350.03). This is an intended synthetic
signal, not causal evidence.

International transactions have a 3.04% fraud rate versus 0.86% for domestic
transactions. USD and EUR transactions also have higher observed rates than BRL;
country is therefore related to the synthetic international signal. Newer devices
have lower average age among fraud rows (232 versus 274 days). Other numeric
differences are modest because the target is probabilistic and uses interactions.

## Outliers and correlation

Outliers use the IQR rule and are reported, not removed: `amount` 8.13%,
`avg_amount_last_30d` 5.29%, `distance_from_home_km` 4.57%, `device_age_days`
3.73%, `account_age_days` 3.48%, and `transactions_last_24h` 0.73%. Extreme
values can be legitimate signals in fraud work and are not automatically errors.

No pair of candidate numeric features reaches the exploratory absolute-correlation
flag of 0.90. `amount` and `avg_amount_last_30d` have near-zero linear correlation
(-0.005), so a future defensive ratio such as `amount / avg_amount_last_30d`
remains a hypothesis rather than a redundant duplicate. No numeric feature has
an absolute correlation of 0.90 or more with the target.

## Leakage and temporal review

The audit excludes identifiers (`external_id`, `customer_id`, `device_id`), the
audit timestamp (`occurred_at`), and the target (`is_fraud`) from future feature
inputs. There are no explicit target-derived columns, no perfect numeric class
separation, and no sufficiently represented categorical group with only one
class. These are exploratory checks, not guarantees against every possible form
of leakage.

The first chronological half has a 1.76% fraud rate and the second 1.20%; median
amount is 64.95 versus 66.87, international rate 27.84% versus 28.68%, and mean
24-hour activity 2.21 versus 2.18. This does not show large drift in the
non-target behavior, but the label-rate difference reinforces using time order.
Future ML should use a chronological 70% train, 15% validation, 15% test split,
with past observations preceding future observations. No split is created here.

## Future feature recommendation

Exclude from training: `external_id`, `customer_id`, `device_id`, `occurred_at`,
and `is_fraud` (target). High-cardinality identifiers risk entity memorization;
the timestamp is primarily for ordering, split, and audit.

Candidate numeric features: `amount`, `account_age_days`,
`transactions_last_24h`, `avg_amount_last_30d`, `device_age_days`,
`distance_from_home_km`, and `hour`.

Candidate categorical features: `currency`, `merchant_category`, `country`,
`transaction_type`, `channel`, and `is_international`. Since country and
international status are related, later feature work should reassess redundancy.

Future hypotheses only: amount-to-customer-average ratio, night-transaction
indicator, device-recency buckets, and distance buckets. No feature engineering,
model, or training pipeline is implemented in this stage.
