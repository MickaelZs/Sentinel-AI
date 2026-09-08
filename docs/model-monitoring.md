# Offline model monitoring

This stage provides an experimental, offline monitoring foundation for the
trusted persisted Logistic Regression artifact. The canonical reference is the
chronological train split of `generate_transactions(10_000, 42)`; validation is
the current dataset. Test is excluded from reference distributions, bins, and
monitoring decisions.

The report checks current-data quality, score distribution and alert rate, PSI
for seven numeric features and scores, TVD for six categorical features, and
supervised performance only when `is_fraud` is supplied. PSI uses reference-only
quantile bins with `-inf/+inf` bounds and epsilon `1e-6`; PSI/TVD levels are
heuristics (`<0.10` low, `<0.25` moderate, otherwise high), not universal
banking thresholds or proof of model degradation.

The optional `--simulate-drift` scenario changes only a copy of validation:
amount is multiplied, distance increases, and channel becomes a new synthetic
category. It is a detector demo, not real customer drift.

Without labels, performance is `null`. With delayed real-world labels,
performance would be calculated only after settlement or investigation. Drift
is a statistical signal, not causal proof, and a high alert rate does not equal
fraud prevalence; this baseline threshold was selected for recall.

No data, scores, reports, metrics, or explanations are persisted by default;
no retraining, calibration, scheduler, alerting system, database, dashboard, or
external monitoring service is included. Joblib artifacts remain trusted-source
only and all project data in this stage are synthetic.
