"""Deterministic E/B/R/P memory management for the smoke environment."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from examples.social_memory_community import types


class MemoryController:
  """Maintain one agent's episodic, belief, relation, and public state.

  This first implementation is deliberately deterministic.  A learned memory
  manager can later implement the same interface while the environment and
  action-policy comparisons remain unchanged.
  """

  def __init__(
      self,
      owner: str,
      *,
      top_k: int = 6,
      learning_rate: float = 0.25,
  ) -> None:
    if not owner:
      raise ValueError("owner must be non-empty.")
    if top_k < 1:
      raise ValueError("top_k must be positive.")
    if not 0.0 < learning_rate <= 1.0:
      raise ValueError("learning_rate must be in (0, 1].")
    self.owner = owner
    self.top_k = top_k
    self.learning_rate = learning_rate
    self._events: list[types.EventRecord] = []
    self._beliefs: dict[str, types.BeliefState] = {}
    self._relations: dict[str, types.RelationState] = {}
    self._public: dict[str, types.PublicState] = {}
    self._best_observation_reliability: dict[
        tuple[int, str, str, str], float
    ] = {}
    self._public_observation_keys: set[tuple[int, str, str, str]] = set()

  def observe(self, event: types.EventRecord) -> None:
    """Store an observation and perform a bottom-up state update.

    In mixed visibility mode the same interaction can arrive as direct and
    public observations.  B/R use the most reliable view, while P records
    each deduplicated public view.  All event records remain available for
    replay and auditing.
    """
    self._events.append(event)
    partner = self._partner_for_observer(event)
    key = (event.round_index, event.actor, partner, event.observer)
    reliability = _clip(event.reliability, 0.0, 1.0)
    previous_reliability = self._best_observation_reliability.get(key, -1.0)
    outcome_value = _outcome_value(event.outcome)
    if reliability > previous_reliability:
      self._best_observation_reliability[key] = reliability
      self._update_private_states(
          event,
          partner=partner,
          reliability=reliability,
          outcome_value=outcome_value,
      )

    if event.visibility == "public":
      if key in self._public_observation_keys:
        return
      self._public_observation_keys.add(key)
      public = self._public.get(
          partner,
          types.PublicState(owner=self.owner, subject=partner),
      )
      self._public[partner] = types.PublicState(
          owner=self.owner,
          subject=partner,
          reputation=_clip(
              public.reputation
              + 0.5 * self.learning_rate * reliability * outcome_value
          ),
          support=public.support + max(0.0, outcome_value),
          count=public.count + 1,
          last_update_round=event.round_index,
      )

  def _update_private_states(
      self,
      event: types.EventRecord,
      *,
      partner: str,
      reliability: float,
      outcome_value: float,
  ) -> None:
    belief = self._beliefs.get(
        partner,
        types.BeliefState(owner=self.owner, subject=partner),
    )
    if outcome_value > 0.0:
      alpha = belief.alpha + self.learning_rate * reliability
      beta = belief.beta
    elif outcome_value < 0.0:
      alpha = belief.alpha
      beta = belief.beta + self.learning_rate * reliability
    else:
      alpha = belief.alpha
      beta = belief.beta
    self._beliefs[partner] = types.BeliefState(
        owner=self.owner,
        subject=partner,
        context=belief.context,
        alpha=alpha,
        beta=beta,
        last_update_round=event.round_index,
    )

    relation = self._relations.get(
        partner,
        types.RelationState(owner=self.owner, partner=partner),
    )
    self._relations[partner] = types.RelationState(
        owner=self.owner,
        partner=partner,
        trust=_clip(
            relation.trust + self.learning_rate * reliability * outcome_value
        ),
        reciprocity=_clip(
            relation.reciprocity
            + (
                self.learning_rate * reliability * outcome_value
                if event.action == "help"
                else 0.0
            )
        ),
        familiarity=_clip(relation.familiarity + 0.1, 0.0, 1.0),
        interaction_count=relation.interaction_count + 1,
        last_update_round=event.round_index,
    )

  def retrieve(
      self,
      actor: str,
      candidates: Sequence[str],
      round_index: int,
  ) -> tuple[types.EventRecord, ...]:
    """Perform top-down routing and return relevant episodic records."""
    del actor
    candidate_set = set(candidates)
    events = [
        event
        for event in self._events
        if self._partner_for_observer(event) in candidate_set
    ]
    ranked = sorted(
        events,
        key=lambda event: (
            self.candidate_score(self._partner_for_observer(event)),
            event.round_index <= round_index,
            event.round_index,
            event.reliability,
        ),
        reverse=True,
    )
    return tuple(ranked[: self.top_k])

  def get_decision_context(
      self,
      actor: str,
      candidates: Sequence[str],
      round_index: int,
  ) -> dict[str, Any]:
    """Return a bounded, read-only E/B/R/P view for one decision.

    The context is intentionally restricted to the candidate set.  This keeps
    prompt growth bounded and makes the top-down query explicit: the current
    actor and candidate partners determine which episodic records and states
    are exposed to the decision layer.
    """
    if actor != self.owner:
      raise ValueError(
          f"Context owner is {self.owner!r}, but actor is {actor!r}."
      )
    candidate_list = tuple(dict.fromkeys(candidates))
    structured = self.get_structured_state()

    def candidate_states(
        state_name: str, default_factory: Any
    ) -> dict[str, dict[str, Any]]:
      state_by_subject = structured[state_name]
      return {
          candidate: state_by_subject.get(
              candidate, asdict(default_factory(candidate))
          )
          for candidate in candidate_list
      }

    return {
        "schema_version": 1,
        "owner": self.owner,
        "round_index": round_index,
        "candidates": list(candidate_list),
        "E": [
            asdict(event)
            for event in self.retrieve(self.owner, candidate_list, round_index)
        ],
        "B": candidate_states(
            "beliefs",
            lambda candidate: types.BeliefState(
                owner=self.owner, subject=candidate
            ),
        ),
        "R": candidate_states(
            "relations",
            lambda candidate: types.RelationState(
                owner=self.owner, partner=candidate
            ),
        ),
        "P": candidate_states(
            "public",
            lambda candidate: types.PublicState(
                owner=self.owner, subject=candidate
            ),
        ),
    }

  def choose_partner(self, candidates: Sequence[str]) -> str:
    """Choose the candidate with the strongest B/R/P state."""
    if not candidates:
      raise ValueError("At least one candidate partner is required.")
    return max(
        candidates,
        key=lambda candidate: (
            self.candidate_score(candidate),
            -candidates.index(candidate),
        ),
    )

  def candidate_score(self, candidate: str) -> float:
    """Combine belief, directed relation, and public reputation."""
    belief = self._beliefs.get(
        candidate,
        types.BeliefState(owner=self.owner, subject=candidate),
    )
    relation = self._relations.get(
        candidate,
        types.RelationState(owner=self.owner, partner=candidate),
    )
    public = self._public.get(
        candidate,
        types.PublicState(owner=self.owner, subject=candidate),
    )
    belief_signal = 2.0 * belief.cooperation_probability - 1.0
    return (
        0.45 * relation.trust
        + 0.20 * relation.reciprocity
        + 0.25 * belief_signal
        + 0.10 * public.reputation
    )

  def get_state(self) -> dict[str, Any]:
    """Return JSON-compatible state for checkpoints."""
    return {
        "version": 1,
        "owner": self.owner,
        "top_k": self.top_k,
        "learning_rate": self.learning_rate,
        "events": [asdict(event) for event in self._events],
        "beliefs": {
            subject: asdict(state) for subject, state in self._beliefs.items()
        },
        "relations": {
            partner: asdict(state) for partner, state in self._relations.items()
        },
        "public": {
            subject: asdict(state) for subject, state in self._public.items()
        },
        "observation_index": [
            {
                "key": list(key),
                "reliability": reliability,
            }
            for key, reliability in self._best_observation_reliability.items()
        ],
        "public_observation_index": [
            list(key) for key in self._public_observation_keys
        ],
    }

  def set_state(self, state: dict[str, Any]) -> None:
    """Restore state produced by :meth:`get_state`."""
    if int(state.get("version", 0)) != 1:
      raise ValueError("Unsupported memory-controller checkpoint version.")
    if state.get("owner", self.owner) != self.owner:
      raise ValueError("Checkpoint belongs to a different agent.")
    self.top_k = int(state.get("top_k", self.top_k))
    self.learning_rate = float(state.get("learning_rate", self.learning_rate))
    self._events = [
        types.event_from_dict(item) for item in state.get("events", [])
    ]
    self._beliefs = {
        subject: types.BeliefState(**item)
        for subject, item in state.get("beliefs", {}).items()
    }
    self._relations = {
        partner: types.RelationState(**item)
        for partner, item in state.get("relations", {}).items()
    }
    self._public = {
        subject: types.PublicState(**item)
        for subject, item in state.get("public", {}).items()
    }
    self._best_observation_reliability = {
        tuple(item["key"]): float(item["reliability"])
        for item in state.get("observation_index", [])
    }
    self._public_observation_keys = {
        tuple(item) for item in state.get("public_observation_index", [])
    }

  def get_structured_state(self) -> dict[str, dict[str, dict[str, Any]]]:
    """Return B/R/P state for diagnostics without exposing mutable objects."""
    return {
        "beliefs": {key: asdict(value) for key, value in self._beliefs.items()},
        "relations": {
            key: asdict(value) for key, value in self._relations.items()
        },
        "public": {key: asdict(value) for key, value in self._public.items()},
    }

  def _partner_for_observer(self, event: types.EventRecord) -> str:
    """Normalize the smoke environment's observer-specific event schema."""
    if event.observer == event.subject:
      return event.actor
    return event.subject


def _outcome_value(outcome: types.OutcomeName) -> float:
  if outcome == "reciprocated":
    return 1.0
  if outcome in ("exploited", "defected"):
    return -1.0
  return 0.0


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
  return min(upper, max(lower, value))
