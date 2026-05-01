"""Integration test for the /eval stub endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_eval_endpoint_returns_501(client: TestClient) -> None:
    r = client.post("/eval", json={})
    assert r.status_code == 501
    assert "not yet wired" in r.json()["detail"].lower()


def test_eval_endpoint_appears_in_openapi(client: TestClient) -> None:
    """The route must be present in the OpenAPI schema even though it is
    a stub — clients depend on a stable schema across releases."""
    schema = client.get("/openapi.json").json()
    assert "/eval" in schema["paths"]
    assert "post" in schema["paths"]["/eval"]
