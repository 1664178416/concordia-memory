"""Run one model-driven social-memory interaction.

This is a vertical smoke test for the current idea. It keeps the population
and the environment deterministic, while replacing one agent's decision rule
with an LLM call conditioned on episodic, belief, directed-relation, and public
memory state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from typing import Any

from concordia.contrib import language_models
from examples.social_memory_community import community
from examples.social_memory_community import memory_controller
from examples.social_memory_community import types

_DEFAULT_BASE_URL = "https://api.vectorengine.ai/v1"
_DEFAULT_MODEL = "gpt-4o-mini"
_ACTIONS: tuple[types.ActionName, ...] = ("help", "refuse", "defect")


def _seed_events() -> tuple[types.EventRecord, ...]:
  """Create prior observations that populate all four memory views."""
  return (
      types.EventRecord(
          event_id="seed-bob-direct",
          round_index=0,
          observer="Alice",
          actor="Alice",
          subject="Bob",
          target="Bob",
          action="help",
          outcome="reciprocated",
          source="Bob",
          visibility="direct",
          reliability=1.0,
      ),
      types.EventRecord(
          event_id="seed-bob-public",
          round_index=0,
          observer="Alice",
          actor="Alice",
          subject="Bob",
          target="Bob",
          action="help",
          outcome="reciprocated",
          source="public_forum",
          visibility="public",
          reliability=0.8,
      ),
      types.EventRecord(
          event_id="seed-carol-direct",
          round_index=0,
          observer="Alice",
          actor="Alice",
          subject="Carol",
          target="Carol",
          action="help",
          outcome="exploited",
          source="Carol",
          visibility="direct",
          reliability=1.0,
      ),
      types.EventRecord(
          event_id="seed-carol-public",
          round_index=0,
          observer="Alice",
          actor="Alice",
          subject="Carol",
          target="Carol",
          action="help",
          outcome="exploited",
          source="public_forum",
          visibility="public",
          reliability=0.8,
      ),
  )


def _memory_prompt(
    controller: memory_controller.MemoryController,
    candidates: tuple[str, ...],
    round_index: int,
) -> str:
  """Render the bidirectional memory context for the decision model."""
  retrieved = controller.retrieve("Alice", candidates, round_index)
  state = controller.get_structured_state()
  context = {
      "owner": "Alice",
      "round": round_index,
      "candidates": candidates,
      "E_episodic": [asdict(event) for event in retrieved],
      "B_beliefs": state["beliefs"],
      "R_directed_relations": state["relations"],
      "P_public_views": state["public"],
  }
  return (
      "You are Alice in a small social simulation. Choose the next partner "
      "and action using the memory context below. The context contains "
      "episodic events (E), private beliefs (B), directed relations (R), "
      "and public views (P). Keep the direction Alice -> partner. Return "
      "exactly one option and no explanation.\n\n"
      + json.dumps(context, ensure_ascii=False, sort_keys=True)
  )


def _decision_from_choice(
    choice: str, candidates: tuple[str, ...]
) -> types.ActionDecision:
  try:
    partner, action = choice.split("|", maxsplit=1)
  except ValueError as error:
    raise ValueError(f"Invalid model choice {choice!r}.") from error
  if partner not in candidates:
    raise ValueError(f"Model selected unknown partner {partner!r}.")
  if action not in _ACTIONS:
    raise ValueError(f"Unsupported model action {action!r}.")
  return types.ActionDecision(actor="Alice", partner=partner, action=action)


def _build_environment() -> community.CommunityEnvironment:
  return community.CommunityEnvironment(
      profiles=(
          types.AgentProfile("Alice", "cooperator"),
          types.AgentProfile("Bob", "cooperator"),
          types.AgentProfile("Carol", "defector"),
      ),
      observation_mode="mixed",
      seed=7,
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--model", default=os.getenv("VECTORENGINE_MODEL", _DEFAULT_MODEL)
  )
  parser.add_argument(
      "--base_url",
      default=os.getenv("VECTORENGINE_BASE_URL", _DEFAULT_BASE_URL),
  )
  args = parser.parse_args()

  api_key = os.getenv("VECTORENGINE_API_KEY")
  if not api_key:
    parser.error("Set VECTORENGINE_API_KEY in the current process first.")

  model = language_models.language_model_setup(
      api_type="vectorengine",
      model_name=args.model,
      api_key=api_key,
      api_base=args.base_url,
  )
  controller = memory_controller.MemoryController("Alice")
  for event in _seed_events():
    controller.observe(event)

  candidates = ("Bob", "Carol")
  memory_before = controller.get_structured_state()
  prompt = _memory_prompt(controller, candidates, round_index=1)
  options = tuple(
      f"{candidate}|{action}" for candidate in candidates for action in _ACTIONS
  )
  _, choice, choice_info = model.sample_choice(prompt, options)
  decision = _decision_from_choice(choice, candidates)

  environment = _build_environment()
  environment.round_index = 1
  result, observations = environment.resolve(decision)
  for event in observations["Alice"]:
    controller.observe(event)
  environment.advance_round()

  output: dict[str, Any] = {
      "model": args.model,
      "base_url": args.base_url.rstrip("/"),
      "memory_before": memory_before,
      "prompt_context": prompt,
      "model_choice": choice,
      "choice_info": choice_info,
      "decision": asdict(decision),
      "environment_result": asdict(result),
      "alice_observation_count": len(observations["Alice"]),
      "memory_after": controller.get_structured_state(),
  }
  print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
