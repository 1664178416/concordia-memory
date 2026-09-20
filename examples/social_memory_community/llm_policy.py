"""LLM action selection on top of the deterministic social-memory updater."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from typing import Any, override

from concordia.language_model import language_model
from examples.social_memory_community import memory_controller
from examples.social_memory_community import policies
from examples.social_memory_community import types

_VALID_ACTIONS = frozenset(("help", "refuse", "defect"))


class InvalidActionResponse(ValueError):
  """Raised when a model response cannot be mapped to a legal action."""


def build_decision_prompt(context: dict[str, Any]) -> str:
  """Build a model prompt from a controller-produced E/B/R/P context."""
  return (
      "You are an agent in a repeated social interaction. Choose one partner "
      "and one action using only the visible memory context below. E is "
      "episodic experience, B is the owner's belief about partner behavior, "
      "R is the directed relation from owner to partner, and P is the "
      "owner's public view. Do not infer hidden profiles or unavailable "
      "events. Return exactly one JSON object with this schema and no "
      'explanation: {"partner": "<candidate>", "action": '
      '"help|refuse|defect"}.\n\n'
      + json.dumps(context, ensure_ascii=False, sort_keys=True)
  )


def parse_action_response(
    raw_response: str,
    *,
    actor: str,
    candidates: Sequence[str],
) -> types.ActionDecision:
  """Parse and validate a structured action response from a model."""
  if not isinstance(raw_response, str) or not raw_response.strip():
    raise InvalidActionResponse("Model returned an empty response.")

  decoder = json.JSONDecoder()
  parsed: object | None = None
  for index, character in enumerate(raw_response):
    if character != "{":
      continue
    try:
      candidate, _ = decoder.raw_decode(raw_response[index:])
    except json.JSONDecodeError:
      continue
    if isinstance(candidate, dict):
      parsed = candidate
      break
  if not isinstance(parsed, dict):
    raise InvalidActionResponse("Response did not contain a JSON object.")

  partner = parsed.get("partner")
  action = parsed.get("action")
  if not isinstance(partner, str) or not isinstance(action, str):
    raise InvalidActionResponse(
        "Response must contain string fields 'partner' and 'action'."
    )
  partner = partner.strip()
  action = action.strip().lower()
  if partner not in candidates:
    raise InvalidActionResponse(f"Unknown partner {partner!r}.")
  if action not in _VALID_ACTIONS:
    raise InvalidActionResponse(f"Unknown action {action!r}.")
  return types.ActionDecision(
      actor=actor,
      partner=partner,
      action=action,
  )


class LlmBidirectionalSocialMemoryPolicy(
    policies.BidirectionalSocialMemoryPolicy
):
  """Use an LLM for top-down action selection and rules for memory updates."""

  name = "llm_bidirectional"

  def __init__(
      self,
      model: language_model.LanguageModel,
      *,
      top_k: int = 6,
      temperature: float = 0.0,
      max_tokens: int = 128,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      fallback_policy: str = "deterministic",
      model_name: str | None = None,
  ) -> None:
    super().__init__(top_k=top_k)
    if not 0.0 <= temperature <= 2.0:
      raise ValueError("temperature must be in [0, 2].")
    if max_tokens < 1:
      raise ValueError("max_tokens must be positive.")
    if timeout <= 0.0:
      raise ValueError("timeout must be positive.")
    if fallback_policy != "deterministic":
      raise ValueError("Only deterministic fallback is currently supported.")
    self._model = model
    self._temperature = temperature
    self._max_tokens = max_tokens
    self._timeout = timeout
    self._fallback_policy = fallback_policy
    self._model_name = model_name
    self._llm_calls = 0
    self._llm_successes = 0
    self._parse_failures = 0
    self._fallbacks = 0
    self._last_status = "not_called"
    self._last_error = ""
    self._last_prompt_hash = ""

  @override
  def decide(
      self,
      actor: str,
      candidates: Sequence[str],
      round_index: int,
  ) -> types.ActionDecision:
    if not candidates:
      raise ValueError("At least one candidate partner is required.")
    memory = self._memory(actor)
    context = memory.get_decision_context(actor, candidates, round_index)
    prompt = build_decision_prompt(context)
    self._llm_calls += 1
    self._last_prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    try:
      raw_response = self._model.sample_text(
          prompt,
          max_tokens=self._max_tokens,
          temperature=self._temperature,
          timeout=self._timeout,
          seed=None,
      )
      decision = parse_action_response(
          raw_response, actor=actor, candidates=candidates
      )
    except InvalidActionResponse as error:
      self._parse_failures += 1
      return self._fallback(
          actor, candidates, round_index, f"parse_error: {error}"
      )
    except Exception as error:  # pylint: disable=broad-exception-caught
      return self._fallback(
          actor,
          candidates,
          round_index,
          f"model_error: {type(error).__name__}: {error}",
      )
    self._llm_successes += 1
    self._last_status = "ok"
    self._last_error = ""
    return decision

  def _fallback(
      self,
      actor: str,
      candidates: Sequence[str],
      round_index: int,
      reason: str,
  ) -> types.ActionDecision:
    self._fallbacks += 1
    self._last_status = "fallback"
    self._last_error = reason[:300]
    return policies.BidirectionalSocialMemoryPolicy.decide(
        self, actor, candidates, round_index
    )

  @override
  def get_diagnostics(self) -> dict[str, float]:
    return {
        "llm_calls": float(self._llm_calls),
        "llm_successes": float(self._llm_successes),
        "llm_parse_failures": float(self._parse_failures),
        "llm_fallbacks": float(self._fallbacks),
    }

  @override
  def get_state(self) -> dict[str, Any]:
    state = super().get_state()
    state["llm"] = {
        "model_name": self._model_name,
        "temperature": self._temperature,
        "max_tokens": self._max_tokens,
        "timeout": self._timeout,
        "fallback_policy": self._fallback_policy,
        "calls": self._llm_calls,
        "successes": self._llm_successes,
        "parse_failures": self._parse_failures,
        "fallbacks": self._fallbacks,
        "last_status": self._last_status,
        "last_error": self._last_error,
        "last_prompt_hash": self._last_prompt_hash,
    }
    return state

  @override
  def set_state(self, state: dict[str, Any]) -> None:
    super().set_state(state)
    llm_state = state.get("llm", {})
    self._llm_calls = int(llm_state.get("calls", 0))
    self._llm_successes = int(llm_state.get("successes", 0))
    self._parse_failures = int(llm_state.get("parse_failures", 0))
    self._fallbacks = int(llm_state.get("fallbacks", 0))
    self._last_status = str(llm_state.get("last_status", self._last_status))
    self._last_error = str(llm_state.get("last_error", self._last_error))
    self._last_prompt_hash = str(
        llm_state.get("last_prompt_hash", self._last_prompt_hash)
    )
