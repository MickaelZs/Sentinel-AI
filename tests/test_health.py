from fastapi.testclient import TestClient

from sentinel_ai import __version__
from sentinel_ai.app import app

client = TestClient(app)


def test_health_returns_liveness_contract() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "sentinel-ai",
        "version": __version__,
    }
