#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mecta.cost import write_final_table  # noqa: E402
from mecta.io import read_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-dataset token-cost rows into one final table"
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for source in args.input:
        value = read_json(Path(source).expanduser().resolve())
        if not isinstance(value, list):
            raise ValueError(f"cost table must be a JSON array: {source}")
        rows.extend(value)
    write_final_table(Path(args.output).expanduser().resolve(), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
