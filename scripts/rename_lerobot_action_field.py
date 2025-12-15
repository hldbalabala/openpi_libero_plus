#!/usr/bin/env python3

"""
Utility script for migrating a LeRobot-format dataset so that the feature key
`action` gets renamed to `actions` everywhere (metadata + parquet payload).

Example:
    python scripts/rename_lerobot_action_field.py \\
        --dataset-root /home/ubuntu/.cache/huggingface/lerobot/libero_plus_lerobot

Add `--dry-run` to preview how many files would be touched before applying.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename the `action` feature to `actions` in a LeRobot dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the root directory of the LeRobot dataset (contains `meta/`, `data/`, etc.).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, only report what would change without modifying any files.",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help=(
            "Optional path to a normalization stats JSON file. "
            "If provided, keys named `action` will also be renamed to `actions`."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow rewriting parquet files in-place. Required unless --dry-run is passed. "
            "Use with caution; make sure you have a backup."
        ),
    )

    args = parser.parse_args(argv)

    if not args.dry_run and not args.overwrite:
        parser.error("Refusing to modify files without --overwrite (or use --dry-run to preview).")

    return args


def rename_info_json(info_path: Path, *, dry_run: bool) -> bool:
    data = json.loads(info_path.read_text())
    features = data.get("features", {})
    if "action" not in features:
        LOGGER.debug("`action` key not present in %s; skipping.", info_path)
        return False

    features["actions"] = features.pop("action")
    if dry_run:
        LOGGER.info("Would update feature schema in %s", info_path)
        return True

    info_path.write_text(json.dumps(data, indent=4, sort_keys=False))
    LOGGER.info("Updated feature schema in %s", info_path)
    return True


def rename_episodes_stats(stats_path: Path, *, dry_run: bool) -> bool:
    changed = False
    updated_lines: list[str] = []

    with stats_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                updated_lines.append(line)
                continue
            payload = json.loads(line)
            stats = payload.get("stats", {})
            if "action" in stats:
                stats["actions"] = stats.pop("action")
                changed = True
            updated_lines.append(json.dumps(payload))

    if not changed:
        LOGGER.debug("No `action` stats found in %s; skipping.", stats_path)
        return False

    if dry_run:
        LOGGER.info("Would rewrite %s", stats_path)
        return True

    tmp_path = stats_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for line in updated_lines:
            f.write(f"{line}\n")
    tmp_path.replace(stats_path)
    LOGGER.info("Rewrote %s", stats_path)
    return True


def rename_parquet_file(parquet_path: Path, *, dry_run: bool) -> bool:
    table = pq.read_table(parquet_path)
    column_names = table.column_names
    if "action" not in column_names:
        return False

    renamed_table = table.rename_columns(
        ["actions" if name == "action" else name for name in column_names]
    )

    if dry_run:
        LOGGER.debug("Would rewrite %s", parquet_path)
        return True

    tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    pq.write_table(renamed_table, tmp_path)
    tmp_path.replace(parquet_path)
    return True


def rename_all_parquet(data_root: Path, *, dry_run: bool) -> int:
    total_updates = 0
    if not data_root.exists():
        LOGGER.warning("Data directory %s does not exist; skipping parquet files.", data_root)
        return 0

    for parquet_path in sorted(data_root.rglob("*.parquet")):
        try:
            if rename_parquet_file(parquet_path, dry_run=dry_run):
                total_updates += 1
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to process %s: %s", parquet_path, exc)
            raise
    if total_updates:
        LOGGER.info("Processed %d parquet file(s).", total_updates)
    else:
        LOGGER.info("No parquet files required updates.")
    return total_updates


def rename_norm_stats(norm_stats_path: Path, *, dry_run: bool) -> bool:
    if not norm_stats_path.exists():
        LOGGER.warning("Normalization stats file %s not found; skipping.", norm_stats_path)
        return False

    data = json.loads(norm_stats_path.read_text())
    if "action" not in data:
        LOGGER.debug("`action` key not present in %s; skipping.", norm_stats_path)
        return False

    data["actions"] = data.pop("action")
    if dry_run:
        LOGGER.info("Would update %s", norm_stats_path)
        return True

    norm_stats_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    LOGGER.info("Updated %s", norm_stats_path)
    return True


def main(argv: Iterable[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    dataset_root: Path = args.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        LOGGER.error("Dataset root %s does not exist.", dataset_root)
        return 1

    meta_dir = dataset_root / "meta"
    data_dir = dataset_root / "data"

    if not meta_dir.exists():
        LOGGER.error("Meta directory not found at %s", meta_dir)
        return 1

    changed = False
    changed |= rename_info_json(meta_dir / "info.json", dry_run=args.dry_run)
    changed |= rename_episodes_stats(meta_dir / "episodes_stats.jsonl", dry_run=args.dry_run)
    parquet_updates = rename_all_parquet(data_dir, dry_run=args.dry_run)
    changed |= parquet_updates > 0

    if args.norm_stats:
        changed |= rename_norm_stats(args.norm_stats.expanduser().resolve(), dry_run=args.dry_run)

    if changed:
        LOGGER.info(
            "%s completed.",
            "Dry run" if args.dry_run else "Migration",
        )
    else:
        LOGGER.info("No changes needed; dataset already uses `actions`.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

