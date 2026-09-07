# Model artifact management

## Purpose and policy

This stage persists a complete, local, auditable Logistic Regression pipeline:
preprocessing, estimator, validation-selected threshold, metadata, and a
SHA-256 integrity digest. Logistic Regression is the sole persisted model
because it remains the official baseline; Random Forest and
HistGradientBoosting remain experimental with inconclusive promotion status.

The artifact is generated from `generate_transactions(rows=10_000, seed=42)`.
It fits only the chronological 70% train partition. The threshold is selected
from validation using the fixed policy `recall >= 0.60`, then maximum precision.
Test labels do not participate in the artifact build.

## Local structure

```text
artifacts/models/sentinel-logistic-baseline-v1/
├── model.joblib
└── metadata.json
```

`/artifacts/` is ignored by Git. The artifact stores neither dataset rows,
predictions, credentials, nor test labels.

## Metadata and integrity

`metadata.json` is UTF-8, indented, deterministic, and rejects NaN/Infinity.
It records the model artifact version separately from the service version,
runtime versions, dataset and split summary, canonical feature contract,
transformed feature count, threshold policy, relevant estimator parameters, and
the SHA-256 of `model.joblib`.

The loader validates metadata and recalculates the SHA-256 before deserializing.
A mismatch fails explicitly. It emits an informative warning when stored and
current scikit-learn major/minor versions differ.

## Safe local operation

The builder writes temporary files in the target directory and atomically
replaces the model and metadata where possible. It cleans up temporary files on
failure. Overwrite is disabled by default; an existing artifact requires the
explicit `--overwrite` option.

Run the canonical build with:

```powershell
.venv\Scripts\python.exe scripts\build_model_artifact.py --rows 10000 --seed 42
```

After saving, the script reloads the artifact and compares deterministic
validation sample probabilities and threshold classifications. This is a
reproducibility check, not an inference API.

## Security

**Joblib/pickle-based model files must only be loaded from trusted sources
because deserialization can execute arbitrary code.** Hash verification detects
unexpected byte changes but does not make an untrusted artifact safe to load.

## Limitations

The artifact is local, based on synthetic data, and not a production fraud
model. There is no remote registry, calibration, retraining/release policy, or
production inference endpoint.
