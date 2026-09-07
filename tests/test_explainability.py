from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from sentinel_ai.app import app
from sentinel_ai.data.generator import generate_transactions
from sentinel_ai.inference.explainability import (
    REASON_CODES,
    ExplainabilityError,
    explain_logistic_regression,
)
from sentinel_ai.inference.schemas import TransactionRiskRequest
from sentinel_ai.inference.service import ModelInferenceService
from sentinel_ai.ml.artifact_builder import build_baseline_artifact


def _payload() -> dict[str, object]:
    return {
        "amount": 215.5,
        "currency": "BRL",
        "merchant_category": "electronics",
        "country": "BR",
        "transaction_type": "purchase",
        "channel": "online",
        "is_international": False,
        "account_age_days": 730,
        "transactions_last_24h": 3,
        "avg_amount_last_30d": 95.2,
        "device_age_days": 180,
        "distance_from_home_km": 12.4,
        "hour": 14,
    }


@pytest.fixture
def service(tmp_path: Path) -> ModelInferenceService:
    artifact_path = tmp_path / "artifact"
    build_baseline_artifact(
        generate_transactions(rows=2_000, seed=42), artifact_path, seed=42
    )
    return ModelInferenceService(artifact_path)


def test_reconstruction_and_aggregation_match_the_fitted_pipeline(
    service: ModelInferenceService,
) -> None:
    request = TransactionRiskRequest(**_payload())
    score = service.score(request)
    artifact = service._load()
    explanation = explain_logistic_regression(
        artifact,
        service._feature_frame(request),
        cast(dict[str, str | int | float | bool], request.model_dump()),
        score.probability,
        top_n=13,
    )
    classifier = artifact.pipeline.named_steps["classifier"]

    assert np.isclose(explanation.reconstructed_logit, explanation.model_logit)
    assert np.isclose(explanation.reconstructed_probability, score.probability)
    assert np.isclose(
        float(classifier.intercept_[0])
        + sum(item.contribution for item in explanation.contributions),
        explanation.model_logit,
    )
    assert sum(
        item.contribution for item in explanation.contributions
    ) == pytest.approx(
        sum(item.contribution for item in explanation.increasing_reasons)
        + sum(item.contribution for item in explanation.decreasing_reasons)
    )


def test_reason_codes_cover_the_canonical_feature_contract() -> None:
    assert set(REASON_CODES) == {
        "amount",
        "account_age_days",
        "transactions_last_24h",
        "avg_amount_last_30d",
        "device_age_days",
        "distance_from_home_km",
        "hour",
        "currency",
        "merchant_category",
        "country",
        "transaction_type",
        "channel",
        "is_international",
    }


def test_reasons_are_directional_sorted_and_limited(
    service: ModelInferenceService,
) -> None:
    _, explanation = service.explain(TransactionRiskRequest(**_payload()))

    assert len(explanation.increasing_reasons) <= 5
    assert len(explanation.decreasing_reasons) <= 5
    assert all(reason.contribution > 0 for reason in explanation.increasing_reasons)
    assert all(reason.contribution < 0 for reason in explanation.decreasing_reasons)
    assert [reason.contribution for reason in explanation.increasing_reasons] == sorted(
        (reason.contribution for reason in explanation.increasing_reasons), reverse=True
    )
    assert [reason.contribution for reason in explanation.decreasing_reasons] == sorted(
        reason.contribution for reason in explanation.decreasing_reasons
    )


def test_unknown_categories_do_not_create_a_false_reason(
    service: ModelInferenceService,
) -> None:
    request = TransactionRiskRequest(
        **{**_payload(), "currency": "GBP", "merchant_category": "new-category"}
    )
    score, explanation = service.explain(request)

    assert score.probability == pytest.approx(explanation.model_probability)
    explained_features = {
        reason.feature
        for reason in (*explanation.increasing_reasons, *explanation.decreasing_reasons)
    }
    assert "currency" not in explained_features
    assert "merchant_category" not in explained_features


def test_explain_endpoint_matches_score_and_preserves_raw_values(
    service: ModelInferenceService,
) -> None:
    previous = app.state.inference_service
    app.state.inference_service = service
    try:
        with TestClient(app) as client:
            score = client.post("/api/risk/score", json=_payload())
            explanation = client.post("/api/risk/explain", json=_payload())
            assert (
                client.post(
                    "/api/risk/explain", json={**_payload(), "hour": 24}
                ).status_code
                == 422
            )
            assert "/api/risk/explain" in client.get("/openapi.json").json()["paths"]
        assert score.status_code == explanation.status_code == 200
        score_body = score.json()
        explanation_body = explanation.json()
        for field in (
            "risk_probability",
            "risk_prediction",
            "threshold",
            "model_name",
            "artifact_version",
        ):
            assert explanation_body[field] == score_body[field]
        reasons = (
            explanation_body["risk_increasing_factors"]
            + explanation_body["risk_decreasing_factors"]
        )
        assert any(
            reason["feature"] == "amount" and reason["value"] == _payload()["amount"]
            for reason in reasons
        )
        assert any(
            reason["feature"] == "merchant_category"
            and reason["value"] == _payload()["merchant_category"]
            for reason in reasons
        )
        assert any(
            reason["feature"] == "is_international"
            and reason["value"] is _payload()["is_international"]
            for reason in reasons
        )
    finally:
        app.state.inference_service = previous


def test_missing_corrupted_and_unsupported_artifacts_fail_safely(
    service: ModelInferenceService, tmp_path: Path
) -> None:
    previous = app.state.inference_service
    artifact = service._load()
    try:
        app.state.inference_service = ModelInferenceService(tmp_path / "missing")
        with TestClient(app) as client:
            response = client.post("/api/risk/explain", json=_payload())
        assert response.status_code == 503
        assert response.json() == {"detail": "model artifact unavailable"}

        (service._artifact_path / "model.joblib").write_bytes(b"corruption")
        app.state.inference_service = ModelInferenceService(service._artifact_path)
        with TestClient(app) as client:
            response = client.post("/api/risk/explain", json=_payload())
        assert response.status_code == 503
        assert response.json() == {"detail": "model artifact unavailable"}

        incompatible = replace(artifact, pipeline=cast(Pipeline, Pipeline([])))
        with pytest.raises(ExplainabilityError):
            explain_logistic_regression(
                incompatible,
                service._feature_frame(TransactionRiskRequest(**_payload())),
                cast(dict[str, str | int | float | bool], _payload()),
                0.5,
            )

        class IncompatibleClassifier:
            classes_ = np.array([0, 1])

        class IncompatiblePipeline:
            def __init__(self) -> None:
                self.named_steps = {"classifier": IncompatibleClassifier()}

            def predict_proba(self, _features: object) -> np.ndarray:
                return np.array([[0.5, 0.5]])

        unsupported_service = ModelInferenceService(tmp_path / "unsupported")
        unsupported_service._loaded = replace(
            artifact, pipeline=cast(Pipeline, IncompatiblePipeline())
        )
        app.state.inference_service = unsupported_service
        with TestClient(app) as client:
            response = client.post("/api/risk/explain", json=_payload())
        assert response.status_code == 503
        assert response.json() == {"detail": "model artifact unavailable"}
    finally:
        app.state.inference_service = previous
