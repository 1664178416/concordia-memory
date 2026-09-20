"""Replay utilities for fixed-trajectory evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from examples.social_memory_community import metrics
from examples.social_memory_community import types


def load_trace(path: str | Path) -> types.SimulationTrace:
  """Load a trajectory JSON file."""
  data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
  return types.trace_from_dict(data)


def replay_metrics(path: str | Path) -> dict[str, float]:
  """Recompute objective metrics without running any policy or LLM."""
  trace = load_trace(path)
  return metrics.compute_metrics(trace.results)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("trajectory", type=Path)
  args = parser.parse_args()
  print(json.dumps(replay_metrics(args.trajectory), indent=2))


if __name__ == "__main__":
  main()
