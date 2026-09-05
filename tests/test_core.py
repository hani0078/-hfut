from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_config_resolves_paths_and_schema_round_trip(tmp_path: Path) -> None:
    from pooltls.config import load_config
    from pooltls.schema import Event

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "dataset: crest\npaths:\n  data_root: ./data\n",
        encoding="utf-8",
    )
    config = load_config(config_file, require_complete=False)
    assert config["paths"]["data_root"] == (tmp_path / "data").resolve()

    event = Event("topic", "e1", "2020-01-02", "An event")
    assert Event.from_dict(event.to_dict()) == event


def test_word_f1_uses_lowercase_alphanumeric_multisets() -> None:
    from pooltls.text import word_f1

    assert word_f1("Aid aid arrives", "aid arrives") == 0.8
    assert word_f1("", "aid") == 0.0


def test_stage_slice_is_inclusive_and_ordered() -> None:
    from pooltls.pipeline import STAGE_NAMES, select_stages

    selected = select_stages("generate_train", "prepare_stage2")
    assert selected[0] == "generate_train"
    assert selected[-1] == "prepare_stage2"
    assert selected == STAGE_NAMES[2:7]


class FakeEncoder:
    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self.vectors = vectors

    def encode(self, texts):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def test_complete_link_blocks_chain_merge() -> None:
    from pooltls.consolidation import consolidate_mentions
    from pooltls.schema import Mention

    mentions = (
        Mention("u", "a", "a1", "2020-01-01", "alpha"),
        Mention("u", "b", "a2", "2020-01-01", "bravo"),
        Mention("u", "c", "a3", "2020-01-01", "charlie"),
    )
    angle = np.deg2rad
    encoder = FakeEncoder(
        {
            "alpha": (1.0, 0.0),
            "bravo": (float(np.cos(angle(40))), float(np.sin(angle(40)))),
            "charlie": (float(np.cos(angle(80))), float(np.sin(angle(80)))),
        }
    )
    candidates = consolidate_mentions(
        mentions,
        encoder,
        semantic_threshold=0.7,
        word_f1_threshold=1.0,
    )
    assert sorted(len(candidate.member_ids) for candidate in candidates) == [1, 2]


def test_global_negative_screening_preserves_other_constraint_pair() -> None:
    from pooltls.schema import Candidate, Constraint, ReferenceEvent
    from pooltls.supervision import build_supervision

    candidate = Candidate("u", "c0", "2020-01-01", "shared event")
    constraints = tuple(Constraint("u", str(index), f"constraint {index}") for index in range(5))
    references = {str(index): () for index in range(5)}
    references["1"] = (
        ReferenceEvent("u", "1", "r0", "2020-01-01", "gold event"),
    )
    encoder = FakeEncoder({"shared event": (1.0, 0.0), "gold event": (1.0, 0.0)})
    result = build_supervision(
        (candidate,),
        constraints,
        references,
        encoder,
        semantic_weight=1.0,
        positive_threshold=0.55,
        negative_threshold=0.38,
        include_reference_positives=False,
    )
    labels = {(row.candidate_id, row.constraint_id): row.label for row in result.examples}
    assert labels[("c0", "1")] == 1
    assert ("c0", "0") not in labels


def test_percentile_rank_uses_average_ties_per_constraint() -> None:
    from pooltls.ranking import percentile_ranks

    ranked = percentile_ranks(np.array([[1.0, 4.0], [1.0, 2.0], [3.0, 1.0]]))
    np.testing.assert_allclose(ranked[:, 0], [0.5, 0.5, 1.0])


def test_decode_uses_budget_and_allows_cross_timeline_sharing() -> None:
    from pooltls.schema import Candidate, Constraint
    from pooltls.timeline import build_timelines

    candidates = {"u": (Candidate("u", "c0", "2020-01-01", "best"),)}
    constraints = {
        "u": (
            Constraint("u", "0", "first"),
            Constraint("u", "1", "second"),
        )
    }
    scores = {"u": np.asarray([[1.0, 1.0]], dtype=np.float32)}
    timelines = build_timelines(
        candidates,
        constraints,
        scores,
        {("u", "0"): 1, ("u", "1"): 1},
    )
    assert timelines[("u", "0")][0].candidate_id == "c0"
    assert timelines[("u", "1")][0].candidate_id == "c0"


def test_perfect_timeline_has_unit_metrics() -> None:
    from pooltls.evaluation import evaluate_predictions
    from pooltls.schema import ReferenceEvent, TimelineEvent

    key = ("u", "0")
    references = {
        key: (
            ReferenceEvent(
                "u", "0", "r0", "2020-01-01", "the city approved emergency aid"
            ),
        )
    }
    predictions = {
        key: (
            TimelineEvent(
                "u",
                "0",
                "c0",
                "2020-01-01",
                "the city approved emergency aid",
                1.0,
            ),
        )
    }
    metrics = evaluate_predictions(predictions, references)
    assert metrics["rouge1_f1"] == 1.0
    assert metrics["rouge2_f1"] == 1.0
    assert metrics["date_f1"] == 1.0


def test_stage1_target_deduplicates_across_constraints() -> None:
    from pooltls.schema import ReferenceEvent
    from pooltls.stage1_data import deduplicate_references

    references = (
        ReferenceEvent("u", "0", "r0", "2020-01-01", "city approves aid"),
        ReferenceEvent("u", "1", "r1", "2020-01-01", "city approved aid"),
    )
    encoder = FakeEncoder(
        {"city approves aid": (1.0, 0.0), "city approved aid": (1.0, 0.0)}
    )
    groups = deduplicate_references(
        "u",
        references,
        encoder,
        semantic_threshold=0.8,
        word_f1_threshold=0.8,
        semantic_weight=0.75,
    )
    assert len(groups) == 1
    assert len(groups[0].reference_ids) == 2
    assert not hasattr(groups[0], "constraint_id")


def test_both_datasets_use_the_same_aligned_gold_stage1_builder() -> None:
    from pooltls.schema import Article, Constraint, ReferenceEvent
    from pooltls.stage1_data import prepare_stage1_records

    class Reader:
        def entity_ids(self, partition):
            assert partition == "train"
            return ("u",)

        def constraints_for(self, entity_ids):
            assert tuple(entity_ids) == ("u",)
            return {
                "u": tuple(
                    Constraint("u", str(index), f"request {index}")
                    for index in range(5)
                )
            }

        def articles_for(self, entity_ids):
            assert tuple(entity_ids) == ("u",)
            return {
                "u": (
                    Article(
                        "u",
                        "a0",
                        "2020-01-01",
                        "Event title",
                        "The article describes the gold event.",
                    ),
                )
            }

        def references_for(self, entity_ids):
            assert tuple(entity_ids) == ("u",)
            return {
                ("u", str(index)): (
                    ReferenceEvent(
                        "u",
                        str(index),
                        f"r{index}",
                        "2020-01-01",
                        "gold event",
                    ),
                )
                for index in range(5)
            }

        def wcep_article_targets(self, partition):
            raise AssertionError("WCEP must not use a dataset-specific Stage-I builder")

    encoder = FakeEncoder(
        {
            "gold event": (1.0, 0.0),
            "Event title The article describes the gold event.": (1.0, 0.0),
        }
    )
    stage1 = {
        "reference_dedup_semantic_threshold": 0.8,
        "reference_dedup_word_f1_threshold": 0.8,
        "reference_dedup_semantic_weight": 0.75,
        "retrieval_semantic_weight": 0.7,
        "top_retrieval": 1,
        "max_articles_per_event": 1,
        "empty_article_ratio": 0.0,
        "require_explicit_target_name": False,
    }
    for dataset in ("crest", "wcep_ctg"):
        artifacts = prepare_stage1_records(
            {"dataset": dataset, "seed": 42, "stage1": stage1},
            Reader(),
            encoder=encoder,
        )
        assert artifacts["summary"]["dataset"] == dataset
        assert len(artifacts["records"]) == 1
        assert artifacts["records"][0]["event_ids"]


def test_generation_prompt_contains_all_constraints_and_parser_filters_bad_item() -> None:
    from pooltls.schema import Article, Constraint
    from pooltls.stage1_data import joint_messages
    from pooltls.stage1_generation import parse_generation_response

    article = Article("u", "a0", "2020-01-01", "Title", "Full article text")
    constraints = tuple(Constraint("u", str(index), f"request {index}") for index in range(5))
    messages = joint_messages(
        article,
        constraints,
        seed=42,
        require_explicit_target_name=False,
    )
    assert all(item.text in messages[1]["content"] for item in constraints)
    positions = [
        messages[1]["content"].index(f"- request {index}")
        for index in range(5)
    ]
    assert sorted(range(5), key=positions.__getitem__) == [4, 1, 3, 2, 0]
    parsed = parse_generation_response(
        '[{"date":"2020-01-02","event_summary":"valid event"},'
        '{"date":"bad","event_summary":"invalid"}]'
    )
    assert [(item.event_date, item.summary) for item in parsed.events] == [
        ("2020-01-02", "valid event")
    ]
    assert parsed.status == "partial"


def test_balanced_cross_entropy_equals_sum_of_class_means() -> None:
    import math
    import torch

    from pooltls.cross_encoder import balanced_binary_cross_entropy_with_logits

    logits = torch.zeros(3, dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 0.0])
    loss = balanced_binary_cross_entropy_with_logits(
        logits,
        targets,
        positive_count=1,
        negative_count=2,
    )
    assert float(loss) == pytest.approx(2.0 * math.log(2.0))
