#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mecta.config import load_config  # noqa: E402
from mecta.data import DatasetReader  # noqa: E402
from mecta.reference_mentions import (  # noqa: E402
    ReferenceInputSettings,
    materialize_reference_mentions,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize partition reference events as Mention records"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--partition",
        choices=("train", "development", "test"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    raw_settings = config.get("reference_input", {})
    if not isinstance(raw_settings, dict):
        raise ValueError("configuration reference_input must be a mapping")
    summary = materialize_reference_mentions(
        DatasetReader(config),
        args.partition,
        args.output_dir,
        ReferenceInputSettings.from_mapping(raw_settings),
        seed=int(config["seed"]),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
