from __future__ import annotations

from pathlib import Path


class FakeReader:
    dataset = "crest"

    def __init__(self) -> None:
        from mecta.schema import Article, Constraint, ReferenceEvent

        self._entities = ("alpha", "beta")
        self._constraints = {
            entity_id: (Constraint(entity_id, "c0", f"about {entity_id}"),)
            for entity_id in self._entities
        }
        self._references = {
            ("alpha", "c0"): (
                ReferenceEvent(
                    "alpha", "c0", "r0", "2020-01-01", "alpha event"
                ),
            ),
            ("beta", "c0"): (
                ReferenceEvent(
                    "beta", "c0", "r0", "2020-02-01", "beta event"
                ),
            ),
        }
        self._articles = {
            entity_id: (
                Article(
                    entity_id,
                    f"a-{entity_id}",
                    "2020-01-01",
                    f"{entity_id} title",
                    f"{entity_id} body",
                ),
            )
            for entity_id in self._entities
        }

    def entity_ids(self, partition: str):
        assert partition in {"train", "development", "test"}
        return self._entities

    def constraints(self, entity_id: str):
        return self._constraints[entity_id]

    def references_for(self, entity_ids):
        assert tuple(entity_ids) == self._entities
        return dict(self._references)

    def articles(self, entity_id: str):
        return self._articles[entity_id]


def test_train_reference_mentions_include_deterministic_cross_entity_pool(
    tmp_path: Path,
) -> None:
    from mecta.io import iter_jsonl, read_json
    from mecta.reference_mentions import (
        ReferenceInputSettings,
        materialize_reference_mentions,
    )

    settings = ReferenceInputSettings(
        train_cross_entity_distractor_ratio=1.0,
        min_train_cross_entity_distractors=1,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    summary = materialize_reference_mentions(
        FakeReader(), "train", first, settings, seed=42
    )
    materialize_reference_mentions(
        FakeReader(), "train", second, settings, seed=42
    )

    assert summary["partition_reference_mentions"] == 2
    assert summary["cross_entity_training_distractors"] == 2
    assert summary["language_model_loaded"] is False
    alpha = tuple(iter_jsonl(first / "alpha.jsonl"))
    assert len(alpha) == 2
    assert {row["summary"] for row in alpha} == {"alpha event", "beta event"}
    provenance = tuple(iter_jsonl(first / "_meta" / "reference_provenance.jsonl"))
    assert {row["role"] for row in provenance} == {
        "partition_reference",
        "cross_entity_training_distractor",
    }
    assert all(
        row["source_entity_id"] != row["target_entity_id"]
        for row in provenance
        if row["role"] == "cross_entity_training_distractor"
    )
    assert (first / "alpha.jsonl").read_bytes() == (
        second / "alpha.jsonl"
    ).read_bytes()
    assert read_json(first / "_meta" / "summary.json") == summary

    cost_rows = tuple(iter_jsonl(first / "_meta" / "call_records.jsonl"))
    assert len(cost_rows) == 2
    assert all(row["call_count"] == 0 for row in cost_rows)
    assert all(row["input_tokens"] == row["output_tokens"] == 0 for row in cost_rows)


def test_non_train_reference_mentions_do_not_include_distractors(tmp_path: Path) -> None:
    from mecta.io import iter_jsonl
    from mecta.reference_mentions import (
        ReferenceInputSettings,
        materialize_reference_mentions,
    )

    summary = materialize_reference_mentions(
        FakeReader(),
        "test",
        tmp_path / "test",
        ReferenceInputSettings(
            train_cross_entity_distractor_ratio=100.0,
            min_train_cross_entity_distractors=100,
        ),
        seed=42,
    )
    assert summary["partition_reference_mentions"] == 2
    assert summary["cross_entity_training_distractors"] == 0
    assert len(tuple(iter_jsonl(tmp_path / "test" / "alpha.jsonl"))) == 1
