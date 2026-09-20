"""Common memory-policy interface and deterministic baselines.

These baselines deliberately use the same rule-based decision layer. Only the
information exposed through ``retrieve`` differs, which makes the first
experiment useful for attributing behavior changes to memory rather than to
different LLM prompts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from typing import Any

from examples.social_memory_community import memory_controller
from examples.social_memory_community import types


class MemoryPolicy(ABC):
  """Interface shared by all memory backends."""

  name: str

  @abstractmethod
  def observe(self, event: types.EventRecord) -> None:
    """Ingest one event visible to this agent."""

  @abstractmethod
  def retrieve(
      self,
      actor: str,
      candidates: Sequence[str],
      round_index: int,
  ) -> tuple[types.EventRecord, ...]:
    """Return the context available for a partner decision."""

  def decide(
      self,
      actor: str,
      candidates: Sequence[str],
      round_index: int,
  ) -> types.ActionDecision:
    """Choose a partner and action with a shared deterministic policy."""
    if not candidates:
      raise ValueError("At least one candidate partner is required.")
    context = self.retrieve(actor, candidates, round_index)
    partner = _choose_partner(context, candidates)
    score = _candidate_score(context, partner)
    action: types.ActionName = "help" if score >= 0.0 else "refuse"
    return types.ActionDecision(actor=actor, partner=partner, action=action)

  @abstractmethod
  def get_state(self) -> dict[str, Any]:
    """Return JSON-compatible state."""

  @abstractmethod
  def set_state(self, state: dict[str, Any]) -> None:
    """Restore JSON-compatible state."""

  def get_diagnostics(self) -> dict[str, float]:
    """Return optional runtime diagnostics for experiment logging."""
    return {}


class NoMemoryPolicy(MemoryPolicy):
  """Current-observation-only baseline."""

  name = "no_memory"

  def observe(self, event: types.EventRecord) -> None:
    del event

  def retrieve(
      self,
      actor: str,
      candidates: Sequence[str],
      round_index: int,
  ) -> tuple[types.EventRecord, ...]:
    del actor, candidates, round_index
    return ()

  def get_state(self) -> dict[str, Any]:
    return {"version": 1, "policy": self.name}

  def set_state(self, state: dict[str, Any]) -> None:
    if state.get("policy", self.name) != self.name:
      raise ValueError("Checkpoint belongs to a different policy.")


class _EventPolicy(MemoryPolicy):
  """Shared event storage for retrieval baselines."""

  def __init__(self) -> None:
    self._events: list[types.EventRecord] = []

  def observe(self, event: types.EventRecord) -> None:
    self._events.append(event)

  def _events_for(self, candidates: Iterable[str]) -> list[types.EventRecord]:
    candidate_set = set(candidates)
    return [event for event in self._events if event.subject in candidate_set]

  def get_state(self) -> dict[str, Any]:
    return {
        "version": 1,
        "policy": self.name,
        "events": [asdict(event) for event in self._events],
    }

  def set_state(self, state: dict[str, Any]) -> None:
    if state.get("policy") != self.name:
      raise ValueError("Checkpoint belongs to a different policy.")
    self._events = [
        types.event_from_dict(event) for event in state.get("events", [])
    ]


class FullContextPolicy(_EventPolicy):
  """Information upper bound: expose all visible candidate events."""

  name = "full_context"

  def retrieve(
      self,
      actor: str,
      candidates: Sequence[str],
      round_index: int,
  ) -> tuple[types.EventRecord, ...]:
    del actor, round_index
    return tuple(self._events_for(candidates))


class FlatEpisodicPolicy(_EventPolicy):
  """Flat recency/relevance retrieval without relation or belief state."""

  name = "flat_episodic"

  def __init__(self, top_k: int = 6) -> None:
    super().__init__()
    if top_k < 1:
      raise ValueError("top_k must be positive.")
    self._top_k = top_k

  def retrieve(
      self,
      actor: str,
      candidates: Sequence[str],
      round_index: int,
  ) -> tuple[types.EventRecord, ...]:
    del actor
    events = self._events_for(candidates)
    ranked = sorted(
        events,
        key=lambda event: (
            event.round_index <= round_index,
            event.round_index,
            event.reliability,
        ),
        reverse=True,
    )
    return tuple(ranked[: self._top_k])

  def get_state(self) -> dict[str, Any]:
    state = super().get_state()
    state["top_k"] = self._top_k
    return state

  def set_state(self, state: dict[str, Any]) -> None:
    super().set_state(state)
    self._top_k = int(state.get("top_k", self._top_k))


class BidirectionalSocialMemoryPolicy(MemoryPolicy):
  """Deterministic E/B/R/P policy with bidirectional memory routing."""

  name = "bidirectional_social"

  def __init__(self, top_k: int = 6) -> None:
    self._top_k = top_k
    self._memory_by_agent: dict[str, memory_controller.MemoryController] = {}

  def _memory(self, actor: str) -> memory_controller.MemoryController:
    if actor not in self._memory_by_agent:
      self._memory_by_agent[actor] = memory_controller.MemoryController(
          owner=actor, top_k=self._top_k
      )
    return self._memory_by_agent[actor]

  def observe(self, event: types.EventRecord) -> None:
    self._memory(event.observer).observe(event)

  def retrieve(
      self,
      actor: str,
      candidates: Sequence[str],
      round_index: int,
  ) -> tuple[types.EventRecord, ...]:
    return self._memory(actor).retrieve(actor, candidates, round_index)

  def decide(
      self,
      actor: str,
      candidates: Sequence[str],
      round_index: int,
  ) -> types.ActionDecision:
    if not candidates:
      raise ValueError("At least one candidate partner is required.")
    memory = self._memory(actor)
    memory.retrieve(actor, candidates, round_index)
    partner = memory.choose_partner(candidates)
    action: types.ActionName = (
        "help" if memory.candidate_score(partner) >= 0.0 else "refuse"
    )
    return types.ActionDecision(actor=actor, partner=partner, action=action)

  def get_state(self) -> dict[str, Any]:
    return {
        "version": 1,
        "policy": self.name,
        "top_k": self._top_k,
        "memory_by_agent": {
            agent: memory.get_state()
            for agent, memory in self._memory_by_agent.items()
        },
    }

  def set_state(self, state: dict[str, Any]) -> None:
    if state.get("policy") != self.name:
      raise ValueError("Checkpoint belongs to a different policy.")
    self._top_k = int(state.get("top_k", self._top_k))
    self._memory_by_agent = {}
    for agent, memory_state in state.get("memory_by_agent", {}).items():
      memory = memory_controller.MemoryController(
          owner=agent, top_k=self._top_k
      )
      memory.set_state(memory_state)
      self._memory_by_agent[agent] = memory


def _event_value(event: types.EventRecord) -> float:
  if event.outcome == "reciprocated":
    return 1.0
  if event.outcome in ("exploited", "defected"):
    return -1.0
  return 0.0


def _candidate_score(
    events: Iterable[types.EventRecord], candidate: str
) -> float:
  relevant = [event for event in events if event.subject == candidate]
  if not relevant:
    return 0.0
  weighted_values = [
      event.reliability * _event_value(event) for event in relevant
  ]
  return sum(weighted_values) / len(weighted_values)


def _choose_partner(
    events: Iterable[types.EventRecord], candidates: Sequence[str]
) -> str:
  scores = {
      candidate: _candidate_score(events, candidate) for candidate in candidates
  }
  return max(
      candidates,
      key=lambda candidate: (scores[candidate], -candidates.index(candidate)),
  )


def make_policy(
    name: str,
    *,
    model: Any | None = None,
    top_k: int = 6,
    llm_temperature: float = 0.0,
    llm_max_tokens: int = 128,
    llm_timeout: float = 60.0,
    fallback_policy: str = "deterministic",
    model_name: str | None = None,
) -> MemoryPolicy:
  """Create a baseline by its CLI name."""
  if name == NoMemoryPolicy.name:
    return NoMemoryPolicy()
  if name == FullContextPolicy.name:
    return FullContextPolicy()
  if name == FlatEpisodicPolicy.name:
    return FlatEpisodicPolicy()
  if name == BidirectionalSocialMemoryPolicy.name:
    return BidirectionalSocialMemoryPolicy(top_k=top_k)
  if name == "llm_bidirectional":
    if model is None:
      raise ValueError("llm_bidirectional requires a language model instance.")
    from examples.social_memory_community import llm_policy

    return llm_policy.LlmBidirectionalSocialMemoryPolicy(
        model,
        top_k=top_k,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
        timeout=llm_timeout,
        fallback_policy=fallback_policy,
        model_name=model_name,
    )
  raise ValueError(
      f"Unknown policy {name!r}; expected no_memory, full_context, "
      "flat_episodic, bidirectional_social, or llm_bidirectional."
  )
