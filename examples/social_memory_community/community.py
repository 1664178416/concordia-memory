"""Objective environment and runner for the first social-memory milestone."""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
from typing import Any, Mapping, Sequence

from concordia.language_model import language_model
from examples.social_memory_community import metrics
from examples.social_memory_community import policies
from examples.social_memory_community import types

DEFAULT_PROFILES = (
    types.AgentProfile("Alice", "cooperator"),
    types.AgentProfile("Carol", "defector"),
    types.AgentProfile("Bob", "cooperator"),
    types.AgentProfile("Dana", "reverser", reversal_round=4),
)


class CommunityEnvironment:
  """Programmatic interaction environment with objective ground truth."""

  def __init__(
      self,
      profiles: Sequence[types.AgentProfile] = DEFAULT_PROFILES,
      *,
      observation_mode: types.ObservationMode = "mixed",
      public_reliability: float = 1.0,
      seed: int = 42,
  ) -> None:
    if len(profiles) < 2:
      raise ValueError("At least two agents are required.")
    if not 0.0 <= public_reliability <= 1.0:
      raise ValueError("public_reliability must be in [0, 1].")
    names = [profile.name for profile in profiles]
    if len(set(names)) != len(names):
      raise ValueError("Agent names must be unique.")
    self.profiles = tuple(profiles)
    self._profiles_by_name = {profile.name: profile for profile in profiles}
    self.observation_mode = observation_mode
    self.public_reliability = public_reliability
    self.seed = seed
    self.round_index = 0
    self.results: list[types.InteractionResult] = []
    self.observations: list[types.EventRecord] = []
    self.decisions: list[types.ActionDecision] = []

  @property
  def agent_names(self) -> tuple[str, ...]:
    return tuple(self._profiles_by_name)

  def resolve(
      self, decision: types.ActionDecision
  ) -> tuple[types.InteractionResult, dict[str, tuple[types.EventRecord, ...]]]:
    """Resolve an action and return observer-specific event records."""
    if decision.actor not in self._profiles_by_name:
      raise ValueError(f"Unknown actor {decision.actor!r}.")
    if decision.partner not in self._profiles_by_name:
      raise ValueError(f"Unknown partner {decision.partner!r}.")
    if decision.actor == decision.partner:
      raise ValueError("An agent cannot select itself as a partner.")

    partner_reliable = self._profiles_by_name[decision.partner].is_reliable(
        self.round_index
    )
    if decision.action == "help":
      if partner_reliable:
        outcome: types.OutcomeName = "reciprocated"
        actor_payoff, partner_payoff = 2.0, 1.0
      else:
        outcome = "exploited"
        actor_payoff, partner_payoff = -1.0, 2.0
    elif decision.action == "refuse":
      outcome = "withheld"
      actor_payoff, partner_payoff = 0.0, 0.0
    else:
      outcome = "defected"
      actor_payoff, partner_payoff = 1.0, -1.0

    result = types.InteractionResult(
        round_index=self.round_index,
        actor=decision.actor,
        partner=decision.partner,
        action=decision.action,
        outcome=outcome,
        partner_reliable=partner_reliable,
        actor_payoff=actor_payoff,
        partner_payoff=partner_payoff,
    )
    observations = self._make_observations(result)
    self.decisions.append(decision)
    self.results.append(result)
    for events in observations.values():
      self.observations.extend(events)
    return result, observations

  def advance_round(self) -> None:
    self.round_index += 1

  def _make_observations(
      self, result: types.InteractionResult
  ) -> dict[str, tuple[types.EventRecord, ...]]:
    events_by_observer: dict[str, list[types.EventRecord]] = {
        name: [] for name in self.agent_names
    }
    direct_observers = {result.actor, result.partner}
    if self.observation_mode in ("direct", "mixed"):
      for observer in direct_observers:
        events_by_observer[observer].append(
            self._event_for_observer(
                result, observer=observer, visibility="direct", reliability=1.0
            )
        )

    if self.observation_mode in ("public", "mixed"):
      for observer in self.agent_names:
        events_by_observer[observer].append(
            self._event_for_observer(
                result,
                observer=observer,
                visibility="public",
                reliability=self.public_reliability,
            )
        )
    return {name: tuple(events) for name, events in events_by_observer.items()}

  def _event_for_observer(
      self,
      result: types.InteractionResult,
      *,
      observer: str,
      visibility: types.VisibilityName,
      reliability: float,
  ) -> types.EventRecord:
    reported_outcome = result.outcome
    if visibility == "public" and reliability < 1.0:
      if _deterministic_flip(
          self.seed,
          result.round_index,
          result.actor,
          result.partner,
          observer,
      ):
        reported_outcome = _flip_outcome(reported_outcome)
    event_id = (
        f"r{result.round_index}-{result.actor}-{result.partner}-"
        f"{observer}-{visibility}"
    )
    return types.EventRecord(
        event_id=event_id,
        round_index=result.round_index,
        observer=observer,
        actor=result.actor,
        subject=result.partner,
        target=result.partner,
        action=result.action,
        outcome=reported_outcome,
        source=result.actor if visibility == "direct" else "public_forum",
        visibility=visibility,
        reliability=reliability,
    )

  def get_state(self) -> dict[str, Any]:
    return {
        "version": 1,
        "round_index": self.round_index,
        "seed": self.seed,
        "observation_mode": self.observation_mode,
        "public_reliability": self.public_reliability,
        "profiles": [asdict(profile) for profile in self.profiles],
        "decisions": [asdict(decision) for decision in self.decisions],
        "results": [asdict(result) for result in self.results],
        "observations": [asdict(event) for event in self.observations],
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    if int(state.get("version", 0)) != 1:
      raise ValueError("Unsupported environment checkpoint version.")
    self.round_index = int(state["round_index"])
    self.decisions = [
        types.ActionDecision(**item) for item in state.get("decisions", [])
    ]
    self.results = [
        types.InteractionResult(**item) for item in state.get("results", [])
    ]
    self.observations = [
        types.event_from_dict(item) for item in state.get("observations", [])
    ]


def run_simulation(
    policy_name: str,
    *,
    rounds: int = 8,
    seed: int = 42,
    observation_mode: types.ObservationMode = "mixed",
    public_reliability: float = 1.0,
    profiles: Sequence[types.AgentProfile] = DEFAULT_PROFILES,
    model: language_model.LanguageModel | None = None,
    llm_agents: Sequence[str] | None = None,
    llm_temperature: float = 0.0,
    llm_max_tokens: int = 128,
    llm_timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
    fallback_policy: str = "deterministic",
    model_name: str | None = None,
) -> types.SimulationTrace:
  """Run one policy population through the objective environment.

  When policy_name is llm_bidirectional, model is used only for the agents
  listed in llm_agents. If the list is omitted, every agent uses the LLM
  policy. Other agents use the deterministic bidirectional controller, which
  makes one-agent live smoke tests inexpensive.
  """
  if rounds < 1:
    raise ValueError("rounds must be positive.")
  environment = CommunityEnvironment(
      profiles,
      observation_mode=observation_mode,
      public_reliability=public_reliability,
      seed=seed,
  )
  if policy_name == "llm_bidirectional":
    if model is None:
      raise ValueError("llm_bidirectional requires a language model instance.")
    selected_llm_agents = set(
        environment.agent_names if llm_agents is None else llm_agents
    )
    unknown_agents = selected_llm_agents.difference(environment.agent_names)
    if unknown_agents:
      raise ValueError(f"Unknown llm_agents: {sorted(unknown_agents)}")
    agent_policies = {
        name: policies.make_policy(
            "llm_bidirectional"
            if name in selected_llm_agents
            else "bidirectional_social",
            model=model,
            top_k=6,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_timeout=llm_timeout,
            fallback_policy=fallback_policy,
            model_name=model_name,
        )
        for name in environment.agent_names
    }
  else:
    agent_policies = {
        name: policies.make_policy(policy_name)
        for name in environment.agent_names
    }

  for _ in range(rounds):
    for actor in environment.agent_names:
      candidates = tuple(
          name for name in environment.agent_names if name != actor
      )
      decision = agent_policies[actor].decide(
          actor, candidates, environment.round_index
      )
      _, observations = environment.resolve(decision)
      for observer, events in observations.items():
        for event in events:
          agent_policies[observer].observe(event)
    environment.advance_round()

  run_metrics = metrics.compute_metrics(environment.results)
  for policy in agent_policies.values():
    for key, value in policy.get_diagnostics().items():
      run_metrics[key] = run_metrics.get(key, 0.0) + value
  checkpoint = {
      "environment": environment.get_state(),
      "policies": {
          name: policy.get_state() for name, policy in agent_policies.items()
      },
      "policy_by_agent": {
          name: policy.name for name, policy in agent_policies.items()
      },
  }
  return types.SimulationTrace(
      policy=policy_name,
      seed=seed,
      rounds=rounds,
      observation_mode=observation_mode,
      public_reliability=public_reliability,
      decisions=tuple(environment.decisions),
      results=tuple(environment.results),
      observations=tuple(environment.observations),
      profiles=tuple(copy.deepcopy(profiles)),
      metrics=run_metrics,
      checkpoint=checkpoint,
  )


def _deterministic_flip(
    seed: int, round_index: int, actor: str, partner: str, observer: str
) -> bool:
  payload = f"{seed}:{round_index}:{actor}:{partner}:{observer}".encode()
  digest = hashlib.sha256(payload).digest()
  return digest[0] % 2 == 0


def _flip_outcome(outcome: types.OutcomeName) -> types.OutcomeName:
  if outcome == "reciprocated":
    return "exploited"
  if outcome == "exploited":
    return "reciprocated"
  return outcome
