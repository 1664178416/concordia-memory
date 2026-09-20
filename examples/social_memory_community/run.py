"""CLI for the LLM-free social-memory community smoke experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.social_memory_community import community


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--policy",
      choices=(
          "no_memory",
          "full_context",
          "flat_episodic",
          "bidirectional_social",
      ),
      default="flat_episodic",
  )
  parser.add_argument("--rounds", type=int, default=8)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
      "--observation_mode",
      choices=("direct", "public", "mixed"),
      default="mixed",
  )
  parser.add_argument("--public_reliability", type=float, default=1.0)
  parser.add_argument(
      "--output_dir",
      type=Path,
      default=Path("results/social_memory_community"),
  )
  args = parser.parse_args()

  trace = community.run_simulation(
      args.policy,
      rounds=args.rounds,
      seed=args.seed,
      observation_mode=args.observation_mode,
      public_reliability=args.public_reliability,
  )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  (args.output_dir / "trajectory.json").write_text(
      json.dumps(trace.to_dict(), indent=2), encoding="utf-8"
  )
  (args.output_dir / "metrics.json").write_text(
      json.dumps(trace.metrics, indent=2), encoding="utf-8"
  )
  (args.output_dir / "checkpoint.json").write_text(
      json.dumps(trace.checkpoint, indent=2), encoding="utf-8"
  )
  print(json.dumps(trace.metrics, indent=2))


if __name__ == "__main__":
  main()
