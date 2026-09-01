#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mecta.config import load_config  # noqa: E402
from mecta.cost import read_cost_records, summarize_cost, write_final_table  # noqa: E402
from mecta.data import DatasetReader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report average per-test-topic mecta LLM calls and tokens"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--call-records", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    reader = DatasetReader(config)
    dataset_name = "CREST" if config["dataset"] == "crest" else "WCEP-CTG"
    row = summarize_cost(
        read_cost_records(args.call_records),
        reader.entity_ids("test"),
        str(config.get("method_name", "mecta")),
        dataset_name,
    )
    write_final_table(args.output, (row,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
