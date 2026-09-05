from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, isfinite
from pathlib import Path
import json
import random
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from .encoders import TextEncoder
from .io import write_json, write_jsonl
from .schema import Article, Constraint, Event, ReferenceEvent
from .text import cosine_matrix, normalize_text, stable_id, word_f1, word_tokens


if TYPE_CHECKING:
    from .data import DatasetReader


SYSTEM_PROMPT = (
    "Extract one shared assignment-free set of atomic events for all requested "
    "timelines. Write concise timeline-style events and return JSON only."
)


@dataclass(frozen=True, slots=True)
class GoldGroup:
    entity_id: str
    group_id: str
    event_date: str
    summary: str
    reference_ids: tuple[str, ...]


def _bounded_probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number in [0, 1]")
    resolved = float(value)
    if not isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be a real number in [0, 1]")
    return resolved


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _constraints(values: Sequence[Constraint], entity_id: str) -> tuple[Constraint, ...]:
    resolved = tuple(values)
    if len(resolved) != 5:
        raise ValueError(f"entity {entity_id!r} must have exactly five constraints")
    if any(item.entity_id != entity_id for item in resolved):
        raise ValueError("constraint entity does not match its owner")
    if len({item.constraint_id for item in resolved}) != len(resolved):
        raise ValueError(f"entity {entity_id!r} has duplicate constraint IDs")
    return tuple(sorted(resolved, key=lambda item: item.constraint_id))


def _record_seed(seed: int, entity_id: str, article_id: str) -> int:
    raw = f"{int(seed)}\0{entity_id}\0{article_id}".encode("utf-8")
    return int.from_bytes(sha256(raw).digest()[:8], "big")


def joint_prompt(
    article: Article,
    constraints: Sequence[Constraint],
    *,
    seed: int,
    require_explicit_target_name: bool,
) -> str:
    """Render the shared full-document, five-constraint extraction prompt."""

    if type(require_explicit_target_name) is not bool:
        raise ValueError("require_explicit_target_name must be a boolean")
    values = list(_constraints(constraints, article.entity_id))
    random.Random(_record_seed(seed, article.entity_id, article.article_id)).shuffle(values)
    rendered_constraints = "\n".join(f"- {item.text}" for item in values)
    target_requirement = (
        ", explicitly name the target entity," if require_explicit_target_name else ""
    )
    return (
        "For this article, perform exactly one joint extraction pass over the complete "
        "document and the entire requested timeline set. Produce one shared, "
        "assignment-free event set; do not run separate extractions for individual "
        "requests.\n\n"
        "Return only a JSON array. Every array element must have exactly these two "
        'keys: {"date":"YYYY-MM-DD","event_summary":"..."}. Each event must be an '
        f"atomic fact written in concise timeline-style language{target_requirement} "
        "and contain no fact that the document does not state.\n\n"
        f"Target entity: {article.entity_id}\n\n"
        "Requested timelines (the complete set; order has no meaning):\n"
        f"{rendered_constraints}\n\n"
        f"Document date: {article.published_at or 'unknown'}\n"
        f"Title:\n{article.title}\n\n"
        f"Document:\n{article.text}"
    )


def assistant_target(events: Sequence[Event | GoldGroup]) -> str:
    ordered = sorted(
        events,
        key=lambda item: (
            item.event_date,
            item.summary.casefold(),
            item.summary,
            item.event_id if isinstance(item, Event) else item.group_id,
        ),
    )
    return json.dumps(
        [
            {"date": item.event_date, "event_summary": item.summary}
            for item in ordered
        ],
        ensure_ascii=False,
    )


def joint_messages(
    article: Article,
    constraints: Sequence[Constraint],
    *,
    seed: int,
    require_explicit_target_name: bool,
    targets: Sequence[Event | GoldGroup] | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": joint_prompt(
                article,
                constraints,
                seed=seed,
                require_explicit_target_name=require_explicit_target_name,
            ),
        },
    ]
    if targets is not None:
        messages.append({"role": "assistant", "content": assistant_target(targets)})
    return messages


def _complete_link(
    compatible: np.ndarray,
    affinity: np.ndarray,
) -> list[list[int]]:
    clusters = [[index] for index in range(int(compatible.shape[0]))]
    while True:
        selected: tuple[int, int] | None = None
        selected_score = -float("inf")
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                cross = np.ix_(clusters[left_index], clusters[right_index])
                if not bool(compatible[cross].all()):
                    continue
                # Complete-link selects the merge with the strongest weakest
                # cross-cluster edge.  The medoid below deliberately remains a
                # mean-affinity center.
                score = float(affinity[cross].min())
                if selected is None or score > selected_score:
                    selected = left_index, right_index
                    selected_score = score
        if selected is None:
            return clusters
        left_index, right_index = selected
        clusters[left_index] = sorted(clusters[left_index] + clusters[right_index])
        del clusters[right_index]


def deduplicate_references(
    entity_id: str,
    references: Sequence[ReferenceEvent],
    encoder: TextEncoder,
    *,
    semantic_threshold: float,
    word_f1_threshold: float,
    semantic_weight: float,
) -> tuple[GoldGroup, ...]:
    """Merge compatible same-day reference variants across all constraints."""

    semantic_cutoff = _bounded_probability(semantic_threshold, "semantic_threshold")
    lexical_cutoff = _bounded_probability(word_f1_threshold, "word_f1_threshold")
    weight = _bounded_probability(semantic_weight, "semantic_weight")
    ordered = sorted(
        references,
        key=lambda item: (
            item.event_date,
            item.summary.casefold(),
            item.summary,
            item.constraint_id,
            item.event_id,
        ),
    )
    if any(item.entity_id != entity_id for item in ordered):
        raise ValueError("reference entity does not match its owner")

    groups: list[GoldGroup] = []
    by_date: dict[str, list[ReferenceEvent]] = defaultdict(list)
    for item in ordered:
        by_date[item.event_date].append(item)
    for event_date in sorted(by_date):
        rows = by_date[event_date]
        embeddings = np.asarray(
            encoder.encode([item.summary for item in rows]), dtype=np.float32
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
            raise ValueError("encoder returned an invalid reference embedding matrix")
        semantic = cosine_matrix(embeddings)
        lexical = np.asarray(
            [
                [word_f1(left.summary, right.summary) for right in rows]
                for left in rows
            ],
            dtype=np.float32,
        )
        compatible = (semantic >= semantic_cutoff) | (lexical >= lexical_cutoff)
        affinity = weight * semantic + (1.0 - weight) * lexical
        for cluster in _complete_link(compatible, affinity):
            medoid = max(
                cluster,
                key=lambda index: (
                    float(affinity[index, cluster].mean()),
                    -len(word_tokens(rows[index].summary)),
                    rows[index].summary.casefold(),
                    rows[index].event_id,
                ),
            )
            members = tuple(rows[index] for index in cluster)
            reference_ids = tuple(
                sorted(
                    f"{item.constraint_id}:{item.event_id}" for item in members
                )
            )
            group_id = stable_id(
                "gold_",
                entity_id,
                event_date,
                reference_ids,
                tuple(sorted(item.summary for item in members)),
            )
            groups.append(
                GoldGroup(
                    entity_id=entity_id,
                    group_id=group_id,
                    event_date=event_date,
                    summary=rows[medoid].summary,
                    reference_ids=reference_ids,
                )
            )
    return tuple(
        sorted(
            groups,
            key=lambda item: (
                item.event_date,
                item.summary.casefold(),
                item.summary,
                item.group_id,
            ),
        )
    )


def retrieve_supporting_articles(
    groups: Sequence[GoldGroup],
    articles: Sequence[Article],
    encoder: TextEncoder,
    *,
    semantic_weight: float,
    top_retrieval: int,
    max_articles_per_event: int,
) -> tuple[dict[str, tuple[GoldGroup, ...]], list[dict[str, Any]]]:
    """Rank complete articles by semantic and lexical support for each event."""

    weight = _bounded_probability(semantic_weight, "retrieval_semantic_weight")
    top_k = _positive_integer(top_retrieval, "top_retrieval")
    keep = _positive_integer(max_articles_per_event, "max_articles_per_event")
    if keep > top_k:
        raise ValueError("max_articles_per_event cannot exceed top_retrieval")
    ordered_groups = tuple(groups)
    ordered_articles = tuple(sorted(articles, key=lambda item: item.article_id))
    if not ordered_groups or not ordered_articles:
        return {}, []
    entity_id = ordered_groups[0].entity_id
    if any(item.entity_id != entity_id for item in ordered_groups + ordered_articles):
        raise ValueError("retrieval accepts one entity at a time")

    article_texts = [normalize_text(f"{item.title} {item.text}") for item in ordered_articles]
    group_vectors = np.asarray(
        encoder.encode([item.summary for item in ordered_groups]), dtype=np.float32
    )
    article_vectors = np.asarray(encoder.encode(article_texts), dtype=np.float32)
    semantic = cosine_matrix(group_vectors, article_vectors)
    by_article: dict[str, dict[str, GoldGroup]] = defaultdict(dict)
    alignments: list[dict[str, Any]] = []
    for group_index, group in enumerate(ordered_groups):
        ranked: list[tuple[float, float, float, Article]] = []
        for article_index, article in enumerate(ordered_articles):
            semantic_score = float(semantic[group_index, article_index])
            lexical_score = word_f1(group.summary, article_texts[article_index])
            score = weight * semantic_score + (1.0 - weight) * lexical_score
            ranked.append((score, semantic_score, lexical_score, article))
        ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3].article_id))
        for score, semantic_score, lexical_score, article in ranked[:top_k][:keep]:
            by_article[article.article_id][group.group_id] = group
            alignments.append(
                {
                    "entity_id": entity_id,
                    "article_id": article.article_id,
                    "group_id": group.group_id,
                    "score": score,
                    "semantic_score": semantic_score,
                    "word_f1": lexical_score,
                }
            )
    resolved = {
        article_id: tuple(
            sorted(
                values.values(),
                key=lambda item: (
                    item.event_date,
                    item.summary.casefold(),
                    item.group_id,
                ),
            )
        )
        for article_id, values in by_article.items()
    }
    alignments.sort(
        key=lambda row: (str(row["entity_id"]), str(row["group_id"]), str(row["article_id"]))
    )
    return resolved, alignments


def _is_reliable_empty(target_entity_id: str, article: Article) -> bool:
    target_name = normalize_text(target_entity_id.replace("_", " ")).casefold()
    tokens = word_tokens(target_name)
    if not target_name or not tokens:
        return False
    searchable = normalize_text(f"{article.title} {article.text}").casefold()
    if target_name in searchable:
        return False
    surname = tokens[-1]
    return len(surname) < 3 or surname not in set(word_tokens(searchable))


def build_aligned_gold_records(
    reader: "DatasetReader",
    entity_ids: Sequence[str],
    encoder: TextEncoder,
    stage1: Mapping[str, Any],
    *,
    dataset_name: str,
    seed: int,
) -> dict[str, Any]:
    """Build the shared Stage-I supervision used by both datasets."""

    if dataset_name not in {"crest", "wcep_ctg"}:
        raise ValueError("dataset_name must be crest or wcep_ctg")
    entity_values = tuple(sorted(entity_ids))
    constraints = reader.constraints_for(entity_values)
    articles = reader.articles_for(entity_values)
    references = reader.references_for(entity_values)
    require_name = stage1.get("require_explicit_target_name", True)
    if type(require_name) is not bool:
        raise ValueError("stage1.require_explicit_target_name must be a boolean")

    groups_by_entity: dict[str, tuple[GoldGroup, ...]] = {}
    by_article: dict[tuple[str, str], tuple[GoldGroup, ...]] = {}
    alignment_rows: list[dict[str, Any]] = []
    for entity_id in entity_values:
        entity_references = tuple(
            item
            for constraint in _constraints(constraints[entity_id], entity_id)
            for item in references.get((entity_id, constraint.constraint_id), ())
        )
        groups = deduplicate_references(
            entity_id,
            entity_references,
            encoder,
            semantic_threshold=float(
                stage1.get("reference_dedup_semantic_threshold", 0.88)
            ),
            word_f1_threshold=float(
                stage1.get("reference_dedup_word_f1_threshold", 0.88)
            ),
            semantic_weight=float(
                stage1.get("reference_dedup_semantic_weight", 0.75)
            ),
        )
        groups_by_entity[entity_id] = groups
        retrieved, rows = retrieve_supporting_articles(
            groups,
            articles[entity_id],
            encoder,
            semantic_weight=float(stage1.get("retrieval_semantic_weight", 0.70)),
            top_retrieval=int(stage1.get("top_retrieval", 5)),
            max_articles_per_event=int(stage1.get("max_articles_per_event", 2)),
        )
        by_article.update(
            {(entity_id, article_id): values for article_id, values in retrieved.items()}
        )
        alignment_rows.extend(rows)

    article_lookup = {
        (entity_id, article.article_id): article
        for entity_id in entity_values
        for article in articles[entity_id]
    }
    records: list[dict[str, Any]] = []
    for key in sorted(by_article):
        entity_id, article_id = key
        target_groups = by_article[key]
        records.append(
            {
                "entity_id": entity_id,
                "article_id": article_id,
                "supervision": "aligned_retrieved_gold",
                "event_ids": [item.group_id for item in target_groups],
                "messages": joint_messages(
                    article_lookup[key],
                    constraints[entity_id],
                    seed=seed,
                    require_explicit_target_name=require_name,
                    targets=target_groups,
                ),
            }
        )

    positive_count = len(records)
    empty_ratio = _bounded_probability(
        stage1.get("empty_article_ratio", 0.05), "empty_article_ratio"
    )
    requested_empty = ceil(positive_count * empty_ratio) if positive_count else 0
    candidates: list[tuple[str, Article]] = []
    for target_entity_id in entity_values:
        for source_entity_id in entity_values:
            if source_entity_id == target_entity_id:
                continue
            for article in articles[source_entity_id]:
                if _is_reliable_empty(target_entity_id, article):
                    candidates.append((target_entity_id, article))
    random.Random(seed).shuffle(candidates)
    for target_entity_id, source in candidates[:requested_empty]:
        synthetic_id = f"negative::{source.entity_id}::{source.article_id}"
        prompt_article = Article(
            entity_id=target_entity_id,
            article_id=synthetic_id,
            published_at=source.published_at,
            title=source.title,
            text=source.text,
        )
        records.append(
            {
                "entity_id": target_entity_id,
                "article_id": synthetic_id,
                "source_entity_id": source.entity_id,
                "source_article_id": source.article_id,
                "supervision": "cross_entity_reliable_empty",
                "event_ids": [],
                "messages": joint_messages(
                    prompt_article,
                    constraints[target_entity_id],
                    seed=seed,
                    require_explicit_target_name=require_name,
                    targets=(),
                ),
            }
        )
    records.sort(key=lambda row: (str(row["entity_id"]), str(row["article_id"])))
    summary = {
        "dataset": dataset_name,
        "train_entities": list(entity_values),
        "training_records": len(records),
        "positive_records": positive_count,
        "empty_records": len(records) - positive_count,
        "gold_event_groups": sum(len(values) for values in groups_by_entity.values()),
        "alignment_records": len(alignment_rows),
        "input_scope": "full_article",
        "all_constraints_joint": True,
        "seed": int(seed),
    }
    return {"records": records, "alignments": alignment_rows, "summary": summary}


def prepare_stage1_records(
    config: Mapping[str, Any],
    reader: "DatasetReader",
    *,
    encoder: TextEncoder | None = None,
) -> dict[str, Any]:
    stage1 = config.get("stage1")
    if not isinstance(stage1, Mapping):
        raise ValueError("configuration must contain a stage1 mapping")
    entity_ids = reader.entity_ids("train")
    seed = int(config.get("seed", 42))
    dataset_name = config.get("dataset")
    if dataset_name not in {"crest", "wcep_ctg"}:
        raise ValueError("dataset must be 'crest' or 'wcep_ctg'")
    if encoder is None:
        raise ValueError("Stage-1 preparation requires a text encoder")
    return build_aligned_gold_records(
        reader,
        entity_ids,
        encoder,
        stage1,
        dataset_name=str(dataset_name),
        seed=seed,
    )


def write_stage1_artifacts(output_dir: str | Path, artifacts: Mapping[str, Any]) -> None:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", artifacts["records"])
    write_jsonl(output / "alignments.jsonl", artifacts.get("alignments", ()))
    write_json(output / "summary.json", artifacts["summary"])


__all__ = [
    "GoldGroup",
    "SYSTEM_PROMPT",
    "assistant_target",
    "build_aligned_gold_records",
    "deduplicate_references",
    "joint_messages",
    "joint_prompt",
    "prepare_stage1_records",
    "retrieve_supporting_articles",
    "write_stage1_artifacts",
]
