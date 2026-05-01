"""Integration tests for the /eval endpoint."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


def test_eval_endpoint_appears_in_openapi(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/eval" in schema["paths"]
    assert "post" in schema["paths"]["/eval"]


def test_eval_endpoint_runs_dataset(client: TestClient, tmp_path: Path) -> None:
    """A small in-line dataset should round-trip through ingestion +
    retrieval + the harness. We never hit OpenAI: FakeOpenAI in the
    integration conftest is wired in via dependency overrides."""
    client.post(
        "/ingest",
        json={
            "text": (
                "Decorators wrap functions to add behavior. "
                "They use @ syntax above the function definition."
            ),
            "source": "decorators.txt",
        },
    )

    dataset = {
        "version": "v1",
        "items": [
            {
                "id": "q1",
                "question": "what do decorators do?",
                "expected_substrings_in_answer": ["[1]"],  # FakeChat returns "stub answer [1]"
                "expected_substrings_in_context": ["decorator"],
                "must_be_answerable": True,
            },
        ],
    }
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(dataset), encoding="utf-8")

    r = client.post("/eval", json={"dataset_path": str(p)})
    assert r.status_code == 200
    body = r.json()
    assert body["num_questions"] == 1
    assert "context_precision" in body["aggregate"]
    assert "answer_correctness" in body["aggregate"]
    assert "answer_relevancy" in body["aggregate"]
    assert len(body["rows"]) == 1


def test_eval_endpoint_404_on_missing_dataset(client: TestClient) -> None:
    r = client.post("/eval", json={"dataset_path": "/does/not/exist.json"})
    assert r.status_code == 404


def test_eval_endpoint_400_on_unknown_prompt_version(
    client: TestClient, tmp_path: Path
) -> None:
    p = tmp_path / "ds.json"
    p.write_text(json.dumps({"version": "v1", "items": []}), encoding="utf-8")
    r = client.post(
        "/eval",
        json={"dataset_path": str(p), "prompt_version": "v99"},
    )
    assert r.status_code == 400
