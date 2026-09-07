import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sentinel_ai.app import app
from sentinel_ai.data.generator import generate_transactions
from sentinel_ai.inference.schemas import TransactionRiskRequest
from sentinel_ai.inference.service import ModelInferenceService
from sentinel_ai.ml.artifact_builder import build_baseline_artifact
from sentinel_ai.ml.artifacts import LoadedModelArtifact, load_model_artifact
from sentinel_ai.ml.metadata import ModelArtifactMetadata


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
def artifact_dir(tmp_path: Path) -> Path:
    path = tmp_path / "artifact"
    build_baseline_artifact(generate_transactions(rows=2_000, seed=42), path, seed=42)
    return path


@pytest.fixture
def client_with_artifact(artifact_dir: Path):
    previous = app.state.inference_service
    app.state.inference_service = ModelInferenceService(artifact_dir)
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.state.inference_service = previous


def test_score_returns_artifact_metadata_and_is_order_independent(
    client_with_artifact: TestClient,
) -> None:
    payload = _payload()
    first = client_with_artifact.post("/api/risk/score", json=payload)
    second = client_with_artifact.post(
        "/api/risk/score", json=dict(reversed(payload.items()))
    )
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    body = first.json()
    assert 0 <= body["risk_probability"] <= 1
    assert body["risk_prediction"] == (body["risk_probability"] >= body["threshold"])
    assert body["model_name"] == "Logistic Regression"
    assert body["artifact_version"] == "sentinel-logistic-baseline-v1"
    assert not {"is_fraud", "customer_id", "device_id", "sha256"}.intersection(body)


def test_unknown_categories_are_scored_and_invalid_input_is_rejected(
    client_with_artifact: TestClient,
) -> None:
    unknown = {**_payload(), "currency": "GBP", "merchant_category": "new-category"}
    assert client_with_artifact.post("/api/risk/score", json=unknown).status_code == 200
    assert (
        client_with_artifact.post(
            "/api/risk/score", json={**_payload(), "amount": 0, "hour": 24}
        ).status_code
        == 422
    )


def test_missing_or_corrupted_artifact_returns_safe_503(
    tmp_path: Path, artifact_dir: Path
) -> None:
    previous = app.state.inference_service
    try:
        app.state.inference_service = ModelInferenceService(tmp_path / "missing")
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            response = client.post("/api/risk/score", json=_payload())
            assert response.status_code == 503
            assert response.json() == {"detail": "model artifact unavailable"}
        (artifact_dir / "model.joblib").write_bytes(b"corruption")
        app.state.inference_service = ModelInferenceService(artifact_dir)
        with TestClient(app) as client:
            assert client.post("/api/risk/score", json=_payload()).status_code == 503
    finally:
        app.state.inference_service = previous


def test_threshold_comes_from_artifact_metadata(
    client_with_artifact: TestClient, artifact_dir: Path
) -> None:
    metadata_path = artifact_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["threshold"] = 0.42
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    app.state.inference_service = ModelInferenceService(artifact_dir)
    response = client_with_artifact.post("/api/risk/score", json=_payload())
    assert response.status_code == 200
    assert response.json()["threshold"] == pytest.approx(0.42)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("amount", -1),
        ("account_age_days", -1),
        ("transactions_last_24h", -1),
        ("device_age_days", -1),
        ("distance_from_home_km", -1),
        ("currency", "US"),
        ("country", "BRA"),
        ("merchant_category", " "),
    ],
)
def test_request_contract_rejects_invalid_values(
    client_with_artifact: TestClient, field: str, value: object
) -> None:
    payload = _payload()
    payload[field] = value

    response = client_with_artifact.post("/api/risk/score", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [("amount", float("nan")), ("distance_from_home_km", float("inf"))],
)
def test_request_schema_rejects_non_finite_values(field: str, value: float) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        TransactionRiskRequest(**payload)


def test_openapi_exposes_risk_score_contract() -> None:
    response = TestClient(app).get("/openapi.json")

    assert response.status_code == 200
    assert "/api/risk/score" in response.json()["paths"]


def test_service_loads_an_artifact_once(
    artifact_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sentinel_ai.inference.service as service_module

    original_loader = service_module.load_model_artifact
    calls = 0

    def counted_loader(path: Path) -> LoadedModelArtifact:
        nonlocal calls
        calls += 1
        return original_loader(path)

    monkeypatch.setattr(service_module, "load_model_artifact", counted_loader)
    service = ModelInferenceService(artifact_dir)
    request = TransactionRiskRequest(**_payload())

    service.score(request)
    service.score(request)

    assert calls == 1


def test_probability_equal_to_threshold_is_positive(artifact_dir: Path) -> None:
    loaded = load_model_artifact(artifact_dir)
    metadata = replace(loaded.metadata, threshold=0.42)

    class FixedClassifier:
        classes_ = np.array([0, 1])

    class FixedPipeline:
        def __init__(self) -> None:
            self.named_steps = {"classifier": FixedClassifier()}

        def predict_proba(self, _features: object) -> np.ndarray:
            return np.array([[0.58, 0.42]])

    service = ModelInferenceService(artifact_dir)
    service._loaded = cast(
        LoadedModelArtifact,
        SimpleNamespace(
            pipeline=FixedPipeline(),
            metadata=cast(ModelArtifactMetadata, metadata),
        ),
    )

    score = service.score(TransactionRiskRequest(**_payload()))

    assert score.probability == pytest.approx(0.42)
    assert score.prediction is True
