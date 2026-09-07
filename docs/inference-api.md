# Model inference API

`POST /api/risk/score` scores one transaction with the complete persisted
Logistic Regression pipeline. The service lazy-loads `MODEL_ARTIFACT_PATH`
(default `artifacts/models/sentinel-logistic-baseline-v1`) under a lock, and
validates metadata plus SHA-256 before loading.

The request contains the 13 canonical model features. It validates finite,
positive/non-negative numeric values, hour from 0 to 23, non-empty strings,
three-character currency and two-character country. Unknown categories remain
valid because the pipeline's `OneHotEncoder(handle_unknown="ignore")` handles
them safely.

Responses contain `risk_probability`, `risk_prediction`, `threshold`,
`model_name`, and `artifact_version`. The rule is
`risk_probability >= metadata.threshold`; the API never trains, selects a
threshold, or uses test data.

Missing or corrupted artifacts return `503` with `model artifact unavailable`.
`GET /health` remains a service liveness check and does not require an artifact.

```bash
curl -X POST http://localhost:8000/api/risk/score -H "Content-Type: application/json" -d '{"amount":215.5,"currency":"BRL","merchant_category":"electronics","country":"BR","transaction_type":"purchase","channel":"online","is_international":false,"account_age_days":730,"transactions_last_24h":3,"avg_amount_last_30d":95.2,"device_age_days":180,"distance_from_home_km":12.4,"hour":14}'
```

The endpoint is unauthenticated and has no rate limit in this development
stage; it must not be exposed publicly as-is. Requests and responses are not
persisted. Joblib artifacts must come from trusted sources.
