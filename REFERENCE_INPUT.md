# Reference-event input mode

This mode keeps the normal clustering, Stage-II supervision, cross-encoder
training, fusion, decoding, and evaluation stages. Stage-I language-model
training and generation are skipped. Each partition's reference events are
materialized through the normal `Mention` schema.

Run metadata records:

```json
{
  "mode": "reference_event_input",
  "uses_partition_references": true,
  "training_cross_entity_distractors": true,
  "language_model_loaded": false
}
```

Development and test results measure downstream behavior when partition
references are already present in the candidate source. They are not
end-to-end document extraction results.

## Training distractors

The unchanged balanced Stage-II objective needs positive and negative samples.
Primary training mentions come from the target entity's references. A
deterministic bounded sample of reference events from other training entities
is added as a candidate distractor pool. The existing global reliable-negative
screen decides which distractors are retained as negatives.

Development and test receive no cross-entity distractors.

## Provenance

Each partition writes:

```text
mentions/<partition>/<entity>.jsonl
mentions/<partition>/_meta/reference_provenance.jsonl
mentions/<partition>/_meta/parse_failures.jsonl
mentions/<partition>/_meta/summary.json
```

No language model or adapter is loaded. GTE and the MiniLM cross-encoder are
still used by the downstream stages.
