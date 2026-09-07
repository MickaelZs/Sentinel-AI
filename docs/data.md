# Synthetic transaction dataset

The local dataset is entirely synthetic and exists for later exploratory
analysis and ML experiments. It contains no real people, accounts, cards, or
financial records.

`generate_transactions` creates a reproducible dataset from a local NumPy RNG.
Transaction amounts are lognormal, counts are Poisson, account and device ages
are positive-skewed, distances have an exponential tail, and timestamps span a
180-day UTC window with non-uniform hours.

`is_fraud` is research-only ground truth. It is a minority, probabilistic label
influenced by pre-decision behavior such as international activity, new devices,
high recent activity, unusual hours, large amounts, and distance. The generator
does not export a risk score, a fraud probability, a reason, or any label-derived
feature, preventing explicit label leakage.

Generated CSV files belong under `/data/`, which Git ignores. The operational
PostgreSQL `transactions` table does not contain `is_fraud`.
