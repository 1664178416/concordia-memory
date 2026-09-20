"""CLI for deterministic and LLM-backed social-memory experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from concordia.contrib import language_models
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
          "llm_bidirectional",
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
      "--api_type",
      default=os.getenv("CONCORDIA_API_TYPE", "vectorengine"),
      help="Language-model registry key used by llm_bidirectional.",
  )
  parser.add_argument(
      "--model",
      default=os.getenv("VECTORENGINE_MODEL", "gpt-4o-mini"),
  )
  parser.add_argument(
      "--base_url",
      default=os.getenv("VECTORENGINE_BASE_URL"),
  )
  parser.add_argument(
      "--llm_agents",
      default="",
      help="Comma-separated agents that use the LLM; empty means all.",
  )
  parser.add_argument("--llm_temperature", type=float, default=0.0)
  parser.add_argument("--llm_max_tokens", type=int, default=128)
  parser.add_argument("--llm_timeout", type=float, default=60.0)
  parser.add_argument(
      "--fallback_policy",
      choices=("deterministic",),
      default="deterministic",
  )
  parser.add_argument(
      "--output_dir",
      type=Path,
      default=Path("results/social_memory_community"),
  )
  args = parser.parse_args()

  model = None
  llm_agents = None
  if args.policy == "llm_bidirectional":
    api_key = os.getenv("VECTORENGINE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
      parser.error(
          "Set VECTORENGINE_API_KEY or OPENAI_API_KEY for llm_bidirectional."
      )
    llm_agents = (
        tuple(
            agent.strip()
            for agent in args.llm_agents.split(",")
            if agent.strip()
        )
        or None
    )
    model = language_models.language_model_setup(
        api_type=args.api_type,
        model_name=args.model,
        api_key=api_key,
        api_base=args.base_url,
    )

  trace = community.run_simulation(
      args.policy,
      rounds=args.rounds,
      seed=args.seed,
      observation_mode=args.observation_mode,
      public_reliability=args.public_reliability,
      model=model,
      llm_agents=llm_agents,
      llm_temperature=args.llm_temperature,
      llm_max_tokens=args.llm_max_tokens,
      llm_timeout=args.llm_timeout,
      fallback_policy=args.fallback_policy,
      model_name=args.model if model is not None else None,
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
