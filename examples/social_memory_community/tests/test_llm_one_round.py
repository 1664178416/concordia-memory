"""Offline tests for the one-round model-driven smoke harness."""

import unittest

from examples.social_memory_community import llm_one_round
from examples.social_memory_community import memory_controller


class LlmOneRoundTest(unittest.TestCase):

  def test_prompt_contains_all_memory_views(self) -> None:
    controller = memory_controller.MemoryController("Alice")
    for event in llm_one_round._seed_events():
      controller.observe(event)

    prompt = llm_one_round._memory_prompt(
        controller, ("Bob", "Carol"), round_index=1
    )

    self.assertIn("E_episodic", prompt)
    self.assertIn("B_beliefs", prompt)
    self.assertIn("R_directed_relations", prompt)
    self.assertIn("P_public_views", prompt)

  def test_decision_parser_validates_partner_and_action(self) -> None:
    decision = llm_one_round._decision_from_choice(
        "Carol|defect", ("Bob", "Carol")
    )
    self.assertEqual(decision.partner, "Carol")
    self.assertEqual(decision.action, "defect")

    with self.assertRaises(ValueError):
      llm_one_round._decision_from_choice("Dana|help", ("Bob", "Carol"))
    with self.assertRaises(ValueError):
      llm_one_round._decision_from_choice("Bob|unknown", ("Bob", "Carol"))


if __name__ == "__main__":
  unittest.main()
