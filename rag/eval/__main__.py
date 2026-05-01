"""CLI entry point: ``python -m rag.eval``.

Usage::

    python -m rag.eval
    python -m rag.eval --top-k 6 --output evals/results/run.json

The script:

1. Loads config (OpenAI key, models, chunk sizes) from the environment.
2. Builds a real :class:`ChromaVectorStore` and ingests the corpus on
   the first run; subsequent runs reuse the persisted store because
   chunk ids are content-deterministic.
3. Runs the bundled golden set against the configured model.
4. Prints a per-question table and writes a JSON artifact.

Running this requires a real ``OPENAI_API_KEY`` — the unit tests in
``tests/unit/test_harness.py`` cover the harness wiring with fakes,
so CI does not need a live API key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openai import OpenAI

from rag.config import get_settings
from rag.eval.harness import EvalHarness, EvalReport, load_golden_dataset
from rag.generation.generator import Generator
from rag.generation.llm_client import OpenAIChatClient, OpenAIEmbedder
from rag.ingestion.chunker import ChunkingConfig
from rag.ingestion.pipeline import IngestionPipeline
from rag.logging import configure_logging, get_logger
from rag.retrieval.reranker import KeywordOverlapReranker
from rag.retrieval.retriever import Retriever
from rag.vectorstore.base import VectorStore
from rag.vectorstore.factory import build_vector_store

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "evals" / "corpus"
GOLDEN_DATASET = REPO_ROOT / "evals" / "golden_dataset.json"
DEFAULT_OUTPUT = REPO_ROOT / "evals" / "results" / "latest.json"


def _ingest_corpus_if_empty(pipeline: IngestionPipeline, vector_store: VectorStore) -> int:
    """Ingest each ``.txt`` file in ``evals/corpus`` if the store is empty.

    Idempotent across runs: chunk ids are derived from content hashes,
    so re-running upserts the same ids without growing the store.
    """
    existing = int(vector_store.count())
    if existing > 0:
        return existing
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        pipeline.ingest_file(path)
    return int(vector_store.count())


def _print_table(report: EvalReport) -> None:
    print()
    print(f"{'id':<35} {'cprec':>6} {'acorr':>6} {'arelv':>6}")
    print("-" * 60)
    for row in report.rows:
        print(
            f"{row.id:<35} "
            f"{row.context_precision:>6.2f} "
            f"{row.answer_correctness:>6.2f} "
            f"{row.answer_relevancy:>6.2f}"
        )
    print("-" * 60)
    agg = report.aggregate
    print(
        f"{'AGGREGATE':<35} "
        f"{agg['context_precision']:>6.2f} "
        f"{agg['answer_correctness']:>6.2f} "
        f"{agg['answer_relevancy']:>6.2f}"
    )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAG eval harness.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the JSON eval report.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=GOLDEN_DATASET,
        help="Path to the golden dataset JSON.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(level=settings.log_level)
    logger = get_logger(__name__)

    if settings.openai_api_key in {"", "sk-test", "sk-replace-me"}:
        logger.error("eval.missing_api_key")
        print(
            "ERROR: OPENAI_API_KEY is not set to a real key. "
            "Edit your .env or export the variable.",
            file=sys.stderr,
        )
        return 2

    openai_client = OpenAI(api_key=settings.openai_api_key)
    embedder = OpenAIEmbedder(client=openai_client, model=settings.openai_embedding_model)
    chat_client = OpenAIChatClient(
        client=openai_client,
        model=settings.openai_llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    vector_store = build_vector_store(settings)
    chunking = ChunkingConfig(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    ingestion = IngestionPipeline(
        embedder=embedder,
        vector_store=vector_store,
        chunking_config=chunking,
    )

    count = _ingest_corpus_if_empty(ingestion, vector_store)
    logger.info("eval.corpus_ready", chunk_count=count)

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        reranker=KeywordOverlapReranker(alpha=0.7),
    )
    generator = Generator(
        retriever=retriever,
        llm_client=chat_client,
        model_name=settings.openai_llm_model,
        prompt_version=settings.prompt_template_version,
    )

    harness = EvalHarness(
        generator=generator,
        retrieve_fn=lambda q, k: retriever.retrieve(q, top_k=k),
        top_k=args.top_k,
    )
    dataset = load_golden_dataset(args.dataset)
    report = harness.run(dataset)
    report = EvalReport(
        rows=report.rows,
        aggregate=report.aggregate,
        metadata={
            "model": settings.openai_llm_model,
            "embedding_model": settings.openai_embedding_model,
            "prompt_version": settings.prompt_template_version,
            "top_k": str(args.top_k),
            "dataset": str(args.dataset),
        },
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(args.output)
    _print_table(report)
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
