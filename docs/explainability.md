# Deterministic risk explainability

`POST /api/risk/explain` provides a local explanation for the persisted
Logistic Regression baseline. It uses the artifact's fitted `ColumnTransformer`
and `LogisticRegression`; it never fits data, rebuilds preprocessing, changes a
threshold, or uses test data.

For one transaction, the model is decomposed in log-odds space:

```text
logit = intercept + sum(transformed_feature_value * coefficient)
probability = sigmoid(logit)
```

The service verifies that the reconstructed logit matches `decision_function()`
and that its sigmoid matches the positive-class probability. Contributions above
or below `1e-12` are respectively `increases_risk` or `decreases_risk`; zero
and neutral contributions are omitted.

Transformed contributions are grouped by their original feature before the top
five increasing and decreasing local reason codes are selected. The supported
codes are `AMOUNT_SIGNAL`, `ACCOUNT_AGE_SIGNAL`, `RECENT_ACTIVITY_SIGNAL`,
`CUSTOMER_AVERAGE_AMOUNT_SIGNAL`, `DEVICE_AGE_SIGNAL`,
`DISTANCE_FROM_HOME_SIGNAL`, `HOUR_SIGNAL`, `CURRENCY_SIGNAL`,
`MERCHANT_CATEGORY_SIGNAL`, `COUNTRY_SIGNAL`, `TRANSACTION_TYPE_SIGNAL`,
`CHANNEL_SIGNAL`, and `INTERNATIONAL_SIGNAL`.

Reason codes report mathematical local evidence that moved this model score;
they are not causal claims, proof of fraud, or an investigation conclusion.
Contributions are log-odds values, not direct percentage changes in fraud
probability. Unknown categories are ignored by the fitted one-hot encoder and
produce no invented categorical reason.

The endpoint is development-only, unauthenticated, stateless, and must not be
exposed publicly as-is. It loads only trusted joblib artifacts because joblib
deserialization is unsafe for untrusted sources. In the future, an LLM may turn
the score and these controlled reason codes into narrative text, but it must not
invent new evidence.
