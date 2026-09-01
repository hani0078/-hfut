from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from .io import iter_jsonl, write_json
from .text import normalize_text


FINAL_TABLE_FIELDS = (
    "dataset",
    "method",
    "calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "calls_per_article",
)


def read_cost_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    return tuple(iter_jsonl(path))


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a non-negative number")
    return result


def summarize_cost(
    records: Iterable[Mapping[str, Any]],
    test_entity_ids: Sequence[str],
    method: str,
    dataset: str,
) -> dict[str, str | int | float]:
    """Return exact average-per-test-topic cost from per-article call records."""

    entity_ids = tuple(normalize_text(value) for value in test_entity_ids)
    if not entity_ids or any(not value for value in entity_ids):
        raise ValueError("test_entity_ids must be a non-empty sequence")
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("test_entity_ids must not contain duplicates")
    method = normalize_text(method)
    dataset = normalize_text(dataset)
    if not method or not dataset:
        raise ValueError("method and dataset must be non-empty strings")

    allowed = set(entity_ids)
    calls = 0.0
    input_tokens = 0.0
    output_tokens = 0.0
    articles: set[tuple[str, str]] = set()
    included_rows = 0
    for row_number, row in enumerate(records, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"cost record {row_number} must be an object")
        entity_id = normalize_text(row.get("entity_id", row.get("topic")))
        if entity_id not in allowed:
            continue
        input_value = _number(row.get("input_tokens"), "input_tokens")
        output_value = _number(row.get("output_tokens"), "output_tokens")
        call_value = _number(
            row.get("call_count", row.get("calls", 1)), "call_count"
        )
        article_id = normalize_text(row.get("article_id")) or f"row_{row_number}"
        articles.add((entity_id, article_id))
        calls += call_value
        input_tokens += input_value
        output_tokens += output_value
        included_rows += 1
    if included_rows == 0:
        raise ValueError("no cost records belong to the requested test entities")

    topic_count = len(entity_ids)
    article_count = len(articles)
    return {
        "dataset": dataset,
        "method": method,
        "calls": round(calls / topic_count, 2),
        "input_tokens": round(input_tokens / topic_count),
        "output_tokens": round(output_tokens / topic_count),
        "total_tokens": round((input_tokens + output_tokens) / topic_count),
        "calls_per_article": round(calls / article_count, 4),
    }


def final_table(
    rows: Iterable[Mapping[str, str | int | float]],
) -> list[dict[str, str | int | float]]:
    output: list[dict[str, str | int | float]] = []
    for index, row in enumerate(rows, 1):
        missing = [field for field in FINAL_TABLE_FIELDS if field not in row]
        if missing:
            raise ValueError(f"final-table row {index} is missing: {missing}")
        output.append({field: row[field] for field in FINAL_TABLE_FIELDS})
    if not output:
        raise ValueError("final table must contain at least one row")
    return output


def write_final_table(
    path: str | Path,
    rows: Iterable[Mapping[str, str | int | float]],
) -> list[dict[str, str | int | float]]:
    table = final_table(rows)
    write_json(path, table)
    return table


__all__ = [
    "FINAL_TABLE_FIELDS",
    "final_table",
    "read_cost_records",
    "summarize_cost",
    "write_final_table",
]
