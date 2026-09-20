"""Serializable data structures for the social-memory smoke environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

ActionName = Literal["help", "refuse", "defect"]
AgentKind = Literal["cooperator", "defector", "reverser"]
ObservationMode = Literal["direct", "public", "mixed"]
OutcomeName = Literal["reciprocated", "exploited", "withheld", "defected"]
VisibilityName = Literal["direct", "public"]


@dataclass(frozen=True, slots=True)
class AgentProfile:
  """Latent behavior used by the objective environment.

  Agents never receive ``kind`` or ``reversal_round`` directly. They are
  ground truth used to score partner selection and adaptation.
  """

  name: str
  kind: AgentKind
  reversal_round: int | None = None

  def is_reliable(self, round_index: int) -> bool:
    if self.kind == "cooperator":
      return True
    if self.kind == "defector":
      return False
    if self.reversal_round is None:
      raise ValueError("A reverser must define reversal_round.")
    return round_index < self.reversal_round


@dataclass(frozen=True, slots=True)
class ActionDecision:
  """A structured action produced by a memory policy."""

  actor: str
  partner: str
  action: ActionName
  message: str = ""


@dataclass(frozen=True, slots=True)
class InteractionResult:
  """Objective result of one interaction."""

  round_index: int
  actor: str
  partner: str
  action: ActionName
  outcome: OutcomeName
  partner_reliable: bool
  actor_payoff: float
  partner_payoff: float


@dataclass(frozen=True, slots=True)
class EventRecord:
  """One observer's record of an objective interaction.

  ``outcome`` is the observer's reported outcome. It may differ from the
  objective outcome when a public report is noisy. ``reliability`` describes
  the source confidence available to the observer, rather than truth itself.
  """

  event_id: str
  round_index: int
  observer: str
  actor: str
  subject: str
  target: str
  action: ActionName
  outcome: OutcomeName
  source: str
  visibility: VisibilityName
  reliability: float


@dataclass(frozen=True, slots=True)
class BeliefState:
  """An agent's uncertain belief about a partner's cooperation."""

  owner: str
  subject: str
  context: str = "cooperation"
  alpha: float = 1.0
  beta: float = 1.0
  last_update_round: int = -1

  @property
  def cooperation_probability(self) -> float:
    total = self.alpha + self.beta
    return self.alpha / total if total else 0.5

  @property
  def confidence(self) -> float:
    return min(1.0, (self.alpha + self.beta - 2.0) / 8.0)


@dataclass(frozen=True, slots=True)
class RelationState:
  """A directed social state owned by one agent."""

  owner: str
  partner: str
  trust: float = 0.0
  reciprocity: float = 0.0
  familiarity: float = 0.0
  interaction_count: int = 0
  last_update_round: int = -1


@dataclass(frozen=True, slots=True)
class PublicState:
  """An agent's local view of a partner's public reputation."""

  owner: str
  subject: str
  reputation: float = 0.0
  support: float = 0.0
  count: int = 0
  last_update_round: int = -1


@dataclass(frozen=True, slots=True)
class SimulationTrace:
  """Complete, JSON-compatible result of one policy run."""

  policy: str
  seed: int
  rounds: int
  observation_mode: ObservationMode
  public_reliability: float
  decisions: tuple[ActionDecision, ...]
  results: tuple[InteractionResult, ...]
  observations: tuple[EventRecord, ...]
  profiles: tuple[AgentProfile, ...]
  metrics: Mapping[str, float]
  checkpoint: Mapping[str, Any]

  def to_dict(self) -> dict[str, Any]:
    """Return a JSON-compatible representation."""
    return asdict(self)


def event_from_dict(data: Mapping[str, Any]) -> EventRecord:
  """Restore an event from a checkpoint or replay file."""
  return EventRecord(**dict(data))


def trace_from_dict(data: Mapping[str, Any]) -> SimulationTrace:
  """Restore a trace written by ``SimulationTrace.to_dict``."""
  return SimulationTrace(
      policy=str(data["policy"]),
      seed=int(data["seed"]),
      rounds=int(data["rounds"]),
      observation_mode=data["observation_mode"],
      public_reliability=float(data["public_reliability"]),
      decisions=tuple(
          ActionDecision(**item) for item in data.get("decisions", [])
      ),
      results=tuple(
          InteractionResult(**item) for item in data.get("results", [])
      ),
      observations=tuple(
          event_from_dict(item) for item in data.get("observations", [])
      ),
      profiles=tuple(AgentProfile(**item) for item in data.get("profiles", [])),
      metrics=dict(data.get("metrics", {})),
      checkpoint=dict(data.get("checkpoint", {})),
  )
