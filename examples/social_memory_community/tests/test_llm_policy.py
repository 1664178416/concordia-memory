"""Offline tests for the LLM-backed social-memory policy."""

from collections.abc import Collection, Sequence
import json
import unittest

from concordia.language_model import language_model
from examples.social_memory_community import community
from examples.social_memory_community import llm_policy
from examples.social_memory_community import types


class _FakeLanguageModel(language_model.LanguageModel):

  def __init__(
      self, response: str = '{"partner": "Bob", "action": "help"}'
  ) -> None:
    self.response = response
    self.prompts: list[str] = []
    self.calls = 0

  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    del (
        max_tokens,
        terminators,
        temperature,
        top_p,
        top_k,
        timeout,
        seed,
    )
    self.calls += 1
    self.prompts.append(prompt)
    return self.response

  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    del prompt, responses, seed
    raise AssertionError("The JSON action policy should call sample_text.")


class _FailingLanguageModel(_FakeLanguageModel):

  def sample_text(self, *args, **kwargs) -> str:
    del args, kwargs
    self.calls += 1
    raise TimeoutError("synthetic timeout")


class LlmPolicyTest(unittest.TestCase):

  def test_parser_accepts_markdown_wrapped_json(self) -> None:
    fenced = chr(96) * 3
    decision = llm_policy.parse_action_response(
        fenced + 'json\n{"partner": "Bob", "action": "defect"}\n' + fenced,
        actor="Alice",
        candidates=("Bob", "Carol"),
    )
    self.assertEqual(decision, types.ActionDecision("Alice", "Bob", "defect"))

  def test_policy_prompt_contains_e_b_r_and_p(self) -> None:
    model = _FakeLanguageModel()
    policy = llm_policy.LlmBidirectionalSocialMemoryPolicy(model)

    decision = policy.decide("Alice", ("Bob", "Carol"), round_index=0)

    self.assertEqual(decision.partner, "Bob")
    self.assertEqual(decision.action, "help")
    prompt = model.prompts[0]
    for key in ('"E"', '"B"', '"R"', '"P"'):
      self.assertIn(key, prompt)
    self.assertNotIn('"kind"', prompt)
    self.assertNotIn('"reversal_round"', prompt)
    self.assertEqual(policy.get_diagnostics()["llm_successes"], 1.0)

  def test_invalid_response_falls_back_and_is_recorded(self) -> None:
    model = _FakeLanguageModel("not-json")
    policy = llm_policy.LlmBidirectionalSocialMemoryPolicy(model)

    decision = policy.decide("Alice", ("Bob", "Carol"), round_index=0)

    self.assertEqual(decision.partner, "Bob")
    self.assertEqual(decision.action, "help")
    diagnostics = policy.get_diagnostics()
    self.assertEqual(diagnostics["llm_parse_failures"], 1.0)
    self.assertEqual(diagnostics["llm_fallbacks"], 1.0)
    self.assertIn("parse_error", policy.get_state()["llm"]["last_error"])

  def test_model_error_falls_back_without_leaking_credentials(self) -> None:
    policy = llm_policy.LlmBidirectionalSocialMemoryPolicy(
        _FailingLanguageModel()
    )

    policy.decide("Alice", ("Bob", "Carol"), round_index=0)
    state = policy.get_state()

    self.assertEqual(policy.get_diagnostics()["llm_fallbacks"], 1.0)
    self.assertNotIn("api_key", json.dumps(state).lower())

  def test_checkpoint_restores_memory_and_diagnostics(self) -> None:
    model = _FakeLanguageModel()
    policy = llm_policy.LlmBidirectionalSocialMemoryPolicy(model)
    policy.decide("Alice", ("Bob", "Carol"), round_index=0)
    state = policy.get_state()

    restored = llm_policy.LlmBidirectionalSocialMemoryPolicy(
        _FakeLanguageModel()
    )
    restored.set_state(state)

    self.assertEqual(restored.get_state(), state)
    self.assertEqual(restored.get_diagnostics()["llm_calls"], 1.0)

  def test_runner_can_limit_llm_to_one_agent(self) -> None:
    model = _FakeLanguageModel()
    trace = community.run_simulation(
        "llm_bidirectional",
        rounds=1,
        profiles=(
            types.AgentProfile("Alice", "cooperator"),
            types.AgentProfile("Bob", "cooperator"),
            types.AgentProfile("Carol", "defector"),
        ),
        model=model,
        llm_agents=("Alice",),
        model_name="fake-model",
    )

    self.assertEqual(trace.metrics["llm_calls"], 1.0)
    self.assertEqual(trace.metrics["llm_successes"], 1.0)
    self.assertEqual(
        trace.checkpoint["policy_by_agent"]["Alice"], "llm_bidirectional"
    )
    self.assertEqual(
        trace.checkpoint["policy_by_agent"]["Bob"], "bidirectional_social"
    )
    self.assertEqual(len(trace.results), 3)


if __name__ == "__main__":
  unittest.main()
