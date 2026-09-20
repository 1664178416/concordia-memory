"""Smoke and serialization tests for the first social-memory milestone."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from examples.social_memory_community import community
from examples.social_memory_community import memory_controller
from examples.social_memory_community import metrics
from examples.social_memory_community import policies
from examples.social_memory_community import types


class SocialMemorySmokeTest(unittest.TestCase):

  def test_all_baselines_produce_a_complete_trace(self) -> None:
    for policy_name in (
        "no_memory",
        "full_context",
        "flat_episodic",
        "bidirectional_social",
    ):
      trace = community.run_simulation(policy_name, rounds=3, seed=7)
      self.assertEqual(len(trace.results), 3 * len(trace.profiles))
      self.assertEqual(trace.metrics["interactions"], 12.0)
      json.dumps(trace.to_dict())

  def test_public_visibility_exposes_more_records(self) -> None:
    direct = community.run_simulation(
        "flat_episodic", rounds=2, observation_mode="direct"
    )
    mixed = community.run_simulation(
        "flat_episodic", rounds=2, observation_mode="mixed"
    )
    self.assertGreater(len(mixed.observations), len(direct.observations))

  def test_checkpoint_restores_policy_state(self) -> None:
    trace = community.run_simulation("flat_episodic", rounds=2, seed=3)
    policy = policies.make_policy("flat_episodic")
    policy.set_state(trace.checkpoint["policies"]["Alice"])
    restored = policy.get_state()
    self.assertEqual(restored["policy"], "flat_episodic")
    self.assertEqual(
        len(restored["events"]),
        len(trace.checkpoint["policies"]["Alice"]["events"]),
    )

  def test_trajectory_round_trip_recomputes_metrics(self) -> None:
    trace = community.run_simulation("full_context", rounds=2, seed=5)
    restored = types.trace_from_dict(json.loads(json.dumps(trace.to_dict())))
    self.assertEqual(
        metrics.compute_metrics(restored.results), dict(trace.metrics)
    )

  def test_metrics_distinguish_selective_cooperation(self) -> None:
    reliable = types.InteractionResult(
        round_index=0,
        actor="A",
        partner="B",
        action="help",
        outcome="reciprocated",
        partner_reliable=True,
        actor_payoff=2.0,
        partner_payoff=1.0,
    )
    unreliable = types.InteractionResult(
        round_index=0,
        actor="A",
        partner="C",
        action="refuse",
        outcome="withheld",
        partner_reliable=False,
        actor_payoff=0.0,
        partner_payoff=0.0,
    )
    result = metrics.compute_metrics([reliable, unreliable])
    self.assertEqual(result["selective_cooperation_gap"], 1.0)

  def test_memory_changes_partner_selection_after_negative_evidence(
      self,
  ) -> None:
    no_memory = community.run_simulation("no_memory", rounds=8, seed=11)
    full_context = community.run_simulation("full_context", rounds=8, seed=11)
    self.assertGreater(
        full_context.metrics["average_actor_payoff"],
        no_memory.metrics["average_actor_payoff"],
    )

  def test_bidirectional_policy_checkpoint_contains_structured_state(
      self,
  ) -> None:
    trace = community.run_simulation("bidirectional_social", rounds=3, seed=13)
    alice_state = trace.checkpoint["policies"]["Alice"]
    self.assertIn("memory_by_agent", alice_state)
    memory_state = alice_state["memory_by_agent"]["Alice"]
    self.assertIn("beliefs", memory_state)
    self.assertIn("relations", memory_state)
    self.assertIn("public", memory_state)
    json.dumps(trace.to_dict())

  def test_social_memory_updates_directed_relation_score(self) -> None:
    policy = policies.BidirectionalSocialMemoryPolicy()
    policy.observe(
        types.EventRecord(
            event_id="positive",
            round_index=0,
            observer="Alice",
            actor="Bob",
            subject="Bob",
            target="Bob",
            action="help",
            outcome="reciprocated",
            source="Bob",
            visibility="direct",
            reliability=1.0,
        )
    )
    positive = policy._memory("Alice").candidate_score("Bob")
    policy.observe(
        types.EventRecord(
            event_id="negative",
            round_index=1,
            observer="Alice",
            actor="Bob",
            subject="Bob",
            target="Bob",
            action="help",
            outcome="exploited",
            source="Bob",
            visibility="direct",
            reliability=1.0,
        )
    )
    after_negative = policy._memory("Alice").candidate_score("Bob")
    self.assertGreater(positive, 0.0)
    self.assertLess(after_negative, positive)

  def test_mixed_visibility_updates_once_and_keeps_public_state_separate(
      self,
  ) -> None:
    controller = memory_controller.MemoryController("Alice")
    direct = types.EventRecord(
        event_id="direct",
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
    )
    public = types.EventRecord(
        event_id="public",
        round_index=0,
        observer="Alice",
        actor="Alice",
        subject="Bob",
        target="Bob",
        action="help",
        outcome="reciprocated",
        source="public_forum",
        visibility="public",
        reliability=0.5,
    )
    controller.observe(direct)
    controller.observe(public)
    state = controller.get_structured_state()
    self.assertEqual(state["relations"]["Bob"]["interaction_count"], 1)
    self.assertEqual(state["public"]["Bob"]["count"], 1)

  def test_relation_state_is_directional(self) -> None:
    alice = memory_controller.MemoryController("Alice")
    bob = memory_controller.MemoryController("Bob")
    alice.observe(
        types.EventRecord(
            event_id="alice-observes",
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
        )
    )
    bob.observe(
        types.EventRecord(
            event_id="bob-observes",
            round_index=0,
            observer="Bob",
            actor="Alice",
            subject="Bob",
            target="Bob",
            action="help",
            outcome="reciprocated",
            source="Alice",
            visibility="direct",
            reliability=1.0,
        )
    )
    self.assertIn("Bob", alice.get_structured_state()["relations"])
    self.assertIn("Alice", bob.get_structured_state()["relations"])


if __name__ == "__main__":
  unittest.main()
