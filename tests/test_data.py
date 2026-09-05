from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONSTRAINT_IDS = tuple(str(index) for index in range(5))


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _reader_config(dataset: str, root: Path, *, constraints: Path | None = None):
    paths: dict[str, Path] = {"data_root": root}
    if constraints is not None:
        paths["constraints"] = constraints
    return {"dataset": dataset, "paths": paths}


def _build_wcep(root: Path) -> dict[str, str]:
    topics = {
        "train": "Train_Topic",
        "validation": "Nicolás_Maduro",
        "test": "Test_Topic",
    }
    for split, topic_id in topics.items():
        split_root = root / split
        _write_jsonl(
            split_root / "topics.jsonl",
            [{"topic_id": topic_id, "topic": topic_id.replace("_", " "), "split": split}],
        )
        _write_jsonl(
            split_root / "constraints.jsonl",
            [
                {
                    "topic_id": topic_id,
                    "constraint_id": index,
                    "constraint": f"request {index}",
                }
                for index in range(5)
            ],
        )
        _write_jsonl(
            split_root / "documents.jsonl",
            [
                {
                    "topic_id": topic_id,
                    "document_id": 0,
                    "event_date": "2020-01-01",
                    "title": "A title",
                    "text": "A body",
                }
            ],
        )
        _write_jsonl(
            split_root / "gold_timelines.jsonl",
            [
                {
                    "topic_id": topic_id,
                    "constraint_id": index,
                    "timeline": (
                        [
                            {
                                "date": "2020-01-02",
                                "events": ["first event", "second event"],
                            }
                        ]
                        if index == 0
                        else []
                    ),
                }
                for index in range(5)
            ],
        )
    return topics


def _build_crest(root: Path) -> tuple[dict[str, str], Path]:
    topics = {
        "train": "Train_Entity",
        "validation": "Development_Entity",
        "test": "Test_Entity",
    }
    constraint_values = {
        topic_id: {constraint_id: f"request {constraint_id}" for constraint_id in CONSTRAINT_IDS}
        for topic_id in topics.values()
    }
    constraint_path = root / "constraint_dict.json"
    constraint_path.parent.mkdir(parents=True, exist_ok=True)
    constraint_path.write_text(
        json.dumps(constraint_values, ensure_ascii=False), encoding="utf-8"
    )
    for split, topic_id in topics.items():
        entity_root = root / split / topic_id
        entity_root.mkdir(parents=True, exist_ok=True)
        with gzip.open(
            entity_root / "articles.preprocessed.jsonl.gz", "wt", encoding="utf-8"
        ) as handle:
            handle.write(
                json.dumps(
                    {
                        "id": 0,
                        "time": "2020-01-01T12:00:00Z",
                        "title": "Title-only article",
                        "text": "",
                        "sentences": [],
                    }
                )
                + "\n"
            )
        for constraint_id in CONSTRAINT_IDS:
            timeline = (
                [["2020-01-02 00:00:00", ["first event", "second event"]]]
                if constraint_id == "0"
                else []
            )
            _write_jsonl(
                entity_root / constraint_id / "timelines.jsonl", [timeline]
            )
    return topics, constraint_path


def test_wcep_split_jsonl_layout(tmp_path: Path) -> None:
    from pooltls.data import DatasetReader

    topics = _build_wcep(tmp_path)
    reader = DatasetReader(_reader_config("wcep_ctg", tmp_path))

    assert reader.entity_ids("development") == (topics["validation"],)
    assert [item.constraint_id for item in reader.constraints(topics["validation"])] == list(
        CONSTRAINT_IDS
    )
    article = reader.articles(topics["validation"])[0]
    assert (article.article_id, article.published_at) == ("0", "2020-01-01")
    references = reader.references(topics["validation"])["0"]
    assert [item.summary for item in references] == ["first event", "second event"]
    assert len({item.event_id for item in references}) == 2


def test_crest_partition_directory_layout(tmp_path: Path) -> None:
    from pooltls.data import DatasetReader

    topics, constraint_path = _build_crest(tmp_path)
    reader = DatasetReader(
        _reader_config("crest", tmp_path, constraints=constraint_path)
    )

    assert reader.entity_ids("development") == (topics["validation"],)
    article = reader.articles(topics["validation"])[0]
    assert (article.article_id, article.published_at, article.text) == (
        "0",
        "2020-01-01",
        "",
    )
    assert [item.summary for item in reader.references(topics["validation"])["0"]] == [
        "first event",
        "second event",
    ]


@pytest.mark.parametrize(
    ("config_name", "expected"),
    [
        (
            "crest.yaml",
            {
                "train": (28, 2141, 519),
                "development": (9, 739, 182),
                "test": (10, 784, 183),
            },
        ),
        (
            "wcep_ctg.yaml",
            {
                "train": (24, 16954, 2956),
                "development": (8, 3890, 743),
                "test": (8, 2694, 589),
            },
        ),
    ],
)
def test_repository_dataset_configs_without_loading_models(
    config_name: str, expected: dict[str, tuple[int, int, int]]
) -> None:
    from pooltls.config import load_config
    from pooltls.data import DatasetReader

    config = load_config(PROJECT_ROOT / "configs" / config_name)
    reader = DatasetReader(config)
    for partition, counts in expected.items():
        entity_ids = reader.entity_ids(partition)
        actual = (
            len(entity_ids),
            sum(len(reader.articles(entity_id)) for entity_id in entity_ids),
            sum(
                len(events)
                for entity_id in entity_ids
                for events in reader.references(entity_id).values()
            ),
        )
        assert actual == counts


def test_preflight_only_checks_model_paths(tmp_path: Path) -> None:
    from pooltls.pipeline import preflight

    data_root = tmp_path / "data"
    _build_wcep(data_root)
    model_root = tmp_path / "models"
    for name in ("base", "gte", "cross"):
        (model_root / name).mkdir(parents=True)
    config = _reader_config("wcep_ctg", data_root)
    config["paths"].update(
        {
            "base_model": model_root / "base",
            "gte_model": model_root / "gte",
            "cross_encoder_model": model_root / "cross",
        }
    )

    result = preflight(config)

    assert result["model_paths_checked"] is True
    assert result["model_weights_loaded"] is False
    assert result["partitions"]["development"] == {
        "entities": 1,
        "articles": 1,
        "reference_events": 2,
    }
