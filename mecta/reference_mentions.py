from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import ceil, isfinite
from pathlib import Path
import random
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from .io import write_json, write_jsonl
from .schema import Mention, ReferenceEvent
from .text import stable_id


if TYPE_CHECKING:
    from .data import DatasetReader


@dataclass(frozen=True, slots=True)
class ReferenceInputSettings:
    train_cross_entity_distractor_ratio: float = 2.0
    min_train_cross_entity_distractors: int = 32

    def __post_init__(self) -> None:
        ratio = float(self.train_cross_entity_distractor_ratio)
        if not isfinite(ratio) or ratio < 0.0:
            raise ValueError(
                "train_cross_entity_distractor_ratio must be a finite non-negative number"
            )
        minimum = self.min_train_cross_entity_distractors
        if type(minimum) is not int or minimum < 0:
            raise ValueError(
                "min_train_cross_entity_distractors must be a non-negative integer"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "ReferenceInputSettings":
        resolved = {} if values is None else dict(values)
        return cls(
            train_cross_entity_distractor_ratio=float(
                resolved.get("train_cross_entity_distractor_ratio", 2.0)
            ),
            min_train_cross_entity_distractors=int(
                resolved.get("min_train_cross_entity_distractors", 32)
            ),
        )


def _entity_seed(seed: int, partition: str, entity_id: str) -> int:
    payload = f"{int(seed)}\0{partition}\0{entity_id}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _ordered_references(
    references: Sequence[ReferenceEvent],
) -> tuple[ReferenceEvent, ...]:
    return tuple(
        sorted(
            references,
            key=lambda item: (
                item.entity_id,
                item.event_date,
                item.summary.casefold(),
                item.summary,
                item.constraint_id,
                item.event_id,
            ),
        )
    )


def _mention(
    target_entity_id: str,
    source: ReferenceEvent,
    *,
    role: str,
) -> tuple[Mention, dict[str, Any]]:
    if role not in {"partition_reference", "cross_entity_training_distractor"}:
        raise ValueError(f"unsupported reference mention role: {role}")
    mention_id = stable_id(
        "mention_",
        "reference_event_input",
        role,
        target_entity_id,
        source.entity_id,
        source.constraint_id,
        source.event_id,
        source.event_date,
        source.summary.casefold(),
    )
    article_id = (
        f"reference::{source.constraint_id}::{source.event_id}"
        if role == "partition_reference"
        else (
            "reference-distractor::"
            f"{source.entity_id}::{source.constraint_id}::{source.event_id}"
        )
    )
    mention = Mention(
        entity_id=target_entity_id,
        mention_id=mention_id,
        article_id=article_id,
        event_date=source.event_date,
        summary=source.summary,
    )
    provenance = {
        "mention_id": mention_id,
        "target_entity_id": target_entity_id,
        "source_entity_id": source.entity_id,
        "source_constraint_id": source.constraint_id,
        "source_event_id": source.event_id,
        "event_date": source.event_date,
        "summary": source.summary,
        "role": role,
    }
    return mention, provenance


def _select_distractors(
    target_entity_id: str,
    own_references: Sequence[ReferenceEvent],
    all_references: Sequence[ReferenceEvent],
    settings: ReferenceInputSettings,
    *,
    partition: str,
    seed: int,
) -> tuple[ReferenceEvent, ...]:
    if partition != "train":
        return ()
    pool = [
        item for item in all_references if item.entity_id != target_entity_id
    ]
    requested = max(
        settings.min_train_cross_entity_distractors,
        ceil(
            len(own_references)
            * float(settings.train_cross_entity_distractor_ratio)
        ),
    )
    random.Random(_entity_seed(seed, partition, target_entity_id)).shuffle(pool)
    return tuple(pool[: min(requested, len(pool))])


def materialize_reference_mentions(
    reader: "DatasetReader",
    partition: str,
    output_dir: str | Path,
    settings: ReferenceInputSettings,
    *,
    seed: int,
) -> dict[str, Any]:
    """Write the normal Mention contract without invoking a language model.

    Partition references form the primary mention pool.  Training additionally
    receives deterministic cross-entity reference events so the unchanged
    global reliable-negative screen can retain a negative class.
    """

    if partition not in {"train", "development", "test"}:
        raise ValueError("partition must be train, development, or test")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")

    entity_ids = tuple(sorted(reader.entity_ids(partition)))
    references = reader.references_for(entity_ids)
    references_by_entity: dict[str, tuple[ReferenceEvent, ...]] = {}
    for entity_id in entity_ids:
        values = tuple(
            event
            for constraint in reader.constraints(entity_id)
            for event in references.get(
                (entity_id, constraint.constraint_id), ()
            )
        )
        references_by_entity[entity_id] = _ordered_references(values)
    all_references = _ordered_references(
        tuple(
            event
            for entity_id in entity_ids
            for event in references_by_entity[entity_id]
        )
    )

    output.mkdir(parents=True)
    metadata = output / "_meta"
    metadata.mkdir()
    provenance_rows: list[dict[str, Any]] = []
    per_entity: dict[str, dict[str, int]] = {}
    total_primary = 0
    total_distractors = 0
    for entity_id in entity_ids:
        own = references_by_entity[entity_id]
        distractors = _select_distractors(
            entity_id,
            own,
            all_references,
            settings,
            partition=partition,
            seed=seed,
        )
        resolved = [
            _mention(entity_id, item, role="partition_reference") for item in own
        ]
        resolved.extend(
            _mention(
                entity_id,
                item,
                role="cross_entity_training_distractor",
            )
            for item in distractors
        )
        mentions = tuple(
            sorted(
                (item[0] for item in resolved),
                key=lambda item: (
                    item.event_date,
                    item.summary.casefold(),
                    item.summary,
                    item.mention_id,
                ),
            )
        )
        identifiers = tuple(item.mention_id for item in mentions)
        if len(identifiers) != len(set(identifiers)):
            raise RuntimeError(f"duplicate mention IDs for {entity_id}")
        write_jsonl(
            output / f"{entity_id}.jsonl",
            (item.to_dict() for item in mentions),
        )
        provenance_rows.extend(item[1] for item in resolved)
        total_primary += len(own)
        total_distractors += len(distractors)
        per_entity[entity_id] = {
            "partition_reference_mentions": len(own),
            "cross_entity_training_distractors": len(distractors),
            "mentions": len(mentions),
        }

    raw_article_count = sum(
        len(reader.articles(entity_id)) for entity_id in entity_ids
    )

    provenance_rows.sort(
        key=lambda row: (
            str(row["target_entity_id"]),
            str(row["event_date"]),
            str(row["summary"]).casefold(),
            str(row["mention_id"]),
        )
    )
    write_jsonl(metadata / "reference_provenance.jsonl", provenance_rows)
    write_jsonl(metadata / "parse_failures.jsonl", ())
    summary = {
        "dataset": reader.dataset,
        "partition": partition,
        "entities": list(entity_ids),
        "raw_articles": raw_article_count,
        "articles": raw_article_count,
        "mentions": total_primary + total_distractors,
        "partition_reference_mentions": total_primary,
        "cross_entity_training_distractors": total_distractors,
        "parse_failures": 0,
        "parse_statuses": {"not_run_reference_event_input": raw_article_count},
        "source": "partition_reference_events",
        "uses_partition_references": True,
        "cross_entity_distractors_train_only": True,
        "language_model_loaded": False,
        "per_entity": per_entity,
        "settings": {
            "train_cross_entity_distractor_ratio": (
                settings.train_cross_entity_distractor_ratio
            ),
            "min_train_cross_entity_distractors": (
                settings.min_train_cross_entity_distractors
            ),
        },
    }
    write_json(metadata / "summary.json", summary)
    return summary


__all__ = [
    "ReferenceInputSettings",
    "materialize_reference_mentions",
]
