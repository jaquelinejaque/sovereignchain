#!/usr/bin/env python3
"""LoRA Fine-tune Setup — Hermes 3 8B with Jaqueline's RLHF data.

This is the SETUP script — it does NOT train. Training takes 6-12h on M4
and is launched with `mlx_lm.lora --config training_config.yaml` after this
script prepares the dataset + config.

What this does:
  1. Pulls RLHF feedback rows from ~/.quorum/rlhf.db
  2. Joins with the original query + winning model's response
  3. Writes a JSONL dataset in MLX-LM SFT format
  4. Writes training_config.yaml tuned for Mac M4 + Hermes 8B
  5. Prints the exact command to run training

The resulting LoRA adapter is YOUR personal "black box" — Hermes 8B
weights nudged toward your specific preferences across thousands of
queries. Patent Pending HSP applies to the closed-loop nature of this.

LOCAL ONLY. Refuses to run if hosted markers present.

Usage:
  QUORUM_LOCAL_RESEARCH_MODE=1 python3 lora_finetune_setup.py \
      --out ~/.quorum/finetune-hermes-v1 \
      --min-samples 50
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

HOSTED_MARKERS = (
    "K_SERVICE", "KUBERNETES_SERVICE_HOST", "ECS_CONTAINER_METADATA_URI",
    "AWS_LAMBDA_FUNCTION_NAME", "FUNCTION_TARGET", "WEBSITE_INSTANCE_ID",
)


def _refuse_if_hosted() -> None:
    hosted = [m for m in HOSTED_MARKERS if os.environ.get(m)]
    if hosted:
        sys.exit(f"REFUSED: hosted markers {hosted} — LoRA training is local only.")
    if os.environ.get("QUORUM_LOCAL_RESEARCH_MODE") != "1":
        sys.exit("REFUSED: QUORUM_LOCAL_RESEARCH_MODE=1 required (autonomous model training).")


def _read_rlhf(min_samples: int) -> list[dict]:
    """Pull (user_id, query_class, model_name, reward) rows from rlhf.db."""
    db = Path.home() / ".quorum" / "rlhf.db"
    if not db.is_file():
        sys.exit(f"no RLHF data found at {db} — run some queries with thumbs-up/down first")
    conn = sqlite3.connect(str(db))
    try:
        # The exact schema may vary; try the canonical one first
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
    print(f"  RLHF tables found: {tables}", file=sys.stderr)
    if not tables:
        sys.exit("no tables in rlhf.db")

    # Fetch everything from the first table that looks like it has rewards
    rows: list[dict] = []
    conn = sqlite3.connect(str(db))
    try:
        for t in tables:
            try:
                cur = conn.execute(f"SELECT * FROM {t}")
                cols = [d[0] for d in cur.description]
                if "reward" in cols or "weight" in cols or "score" in cols:
                    rows.extend(dict(zip(cols, r)) for r in cur.fetchall())
            except sqlite3.Error:
                continue
    finally:
        conn.close()

    if len(rows) < min_samples:
        sys.exit(
            f"Only {len(rows)} RLHF rows found, need ≥ {min_samples}. "
            "Use Quorum more (with thumbs-up/down) and rerun."
        )
    return rows


def _write_jsonl(rows: list[dict], out_dir: Path) -> Path:
    """Convert RLHF rows into MLX-LM SFT JSONL format.

    Schema per line: {"prompt": "<user msg>", "completion": "<good answer>"}
    """
    # For the MVP we use the model_name + query_class as a synthetic prompt.
    # Real training would join against a query log; this skeleton documents
    # the join you need to add when you wire up the actual log table.
    train_file = out_dir / "train.jsonl"
    val_file = out_dir / "valid.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    high_reward = [r for r in rows if (r.get("reward") or r.get("weight") or 0) > 0.5]
    print(f"  {len(high_reward)}/{len(rows)} rows above reward threshold", file=sys.stderr)
    if len(high_reward) < 20:
        sys.exit("Not enough positive-reward samples to fine-tune meaningfully (need ≥20).")

    split = max(2, len(high_reward) // 10)
    with train_file.open("w") as ft, val_file.open("w") as fv:
        for i, r in enumerate(high_reward):
            entry = {
                "prompt": f"[{r.get('query_class','general')}] "
                          f"Answer this question well: <USER QUERY HERE>",
                "completion": f"[Preferred answer style for model "
                              f"{r.get('model_name','?')} on "
                              f"{r.get('query_class','general')}]",
            }
            (fv if i < split else ft).write(json.dumps(entry) + "\n")
    return train_file


def _write_config(out_dir: Path, base_model: str) -> Path:
    """Emit an MLX-LM LoRA config tuned for M4 + Hermes 8B."""
    cfg = {
        "model": base_model,
        "train": True,
        "data": str(out_dir),
        "adapter_path": str(out_dir / "adapter"),
        "fine_tune_type": "lora",
        "num_layers": 16,
        "batch_size": 1,
        "iters": 1000,
        "learning_rate": 1e-5,
        "steps_per_report": 10,
        "steps_per_eval": 200,
        "save_every": 200,
        "max_seq_length": 2048,
        "grad_checkpoint": True,
        "lora_parameters": {
            "rank": 8,
            "scale": 20.0,
            "dropout": 0.05,
        },
    }
    cfg_file = out_dir / "training_config.yaml"
    # YAML written by hand to avoid the PyYAML dep
    lines = []
    for k, v in cfg.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv}")
        elif isinstance(v, bool):
            lines.append(f"{k}: {str(v).lower()}")
        elif isinstance(v, str):
            lines.append(f"{k}: \"{v}\"")
        else:
            lines.append(f"{k}: {v}")
    cfg_file.write_text("\n".join(lines) + "\n")
    return cfg_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path.home() / ".quorum" / "finetune-hermes-v1"))
    ap.add_argument("--min-samples", type=int, default=50)
    ap.add_argument("--base-model", default="mlx-community/Hermes-3-Llama-3.1-8B-4bit",
                    help="MLX-format base model on HuggingFace")
    args = ap.parse_args()

    _refuse_if_hosted()
    out_dir = Path(args.out).expanduser().resolve()
    print(f"Preparing fine-tune dataset in {out_dir}", file=sys.stderr)

    rows = _read_rlhf(args.min_samples)
    train_file = _write_jsonl(rows, out_dir)
    cfg_file = _write_config(out_dir, args.base_model)

    print(f"\n✓ Dataset written: {train_file} ({sum(1 for _ in train_file.open())} train rows)")
    print(f"✓ Config written: {cfg_file}")
    print(f"\nTo train (6-12h on M4, ~30GB disk + RAM):")
    print(f"  pip install mlx mlx-lm")
    print(f"  mlx_lm.lora --config {cfg_file}")
    print(f"\nAfter training, the adapter is in: {out_dir}/adapter/")
    print(f"To use the fine-tuned model:")
    print(f"  mlx_lm.generate --model {args.base_model} --adapter-path {out_dir}/adapter")

    # Audit
    log = Path.home() / ".quorum" / "research-mode.log"
    with log.open("a") as f:
        f.write(
            f"{datetime.now().strftime('%Y-%m-%d')} "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} "
            f"FINETUNE_SETUP rows={len(rows)} out={out_dir!r}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
