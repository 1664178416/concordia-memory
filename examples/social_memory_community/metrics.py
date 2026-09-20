"""Ground-truth metrics for the deterministic community environment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from examples.social_memory_community import types


def compute_metrics(
    results: Sequence[types.InteractionResult],
) -> dict[str, float]:
  """Compute metrics that cannot be gamed by increasing cooperation rate."""
  if not results:
    return {
        "interactions": 0.0,
        "average_actor_payoff": 0.0,
        "partner_selection_accuracy": 0.0,
        "reliable_help_rate": 0.0,
        "unreliable_help_rate": 0.0,
        "selective_cooperation_gap": 0.0,
    }
  help_results = [result for result in results if result.action == "help"]
  reliable_results = [result for result in results if result.partner_reliable]
  unreliable_results = [
      result for result in results if not result.partner_reliable
  ]
  reliable_help_rate = _help_rate(reliable_results)
  unreliable_help_rate = _help_rate(unreliable_results)
  counts = Counter(result.action for result in results)
  return {
      "interactions": float(len(results)),
      "average_actor_payoff": sum(
          result.actor_payoff for result in results
      ) / len(results),
      "partner_selection_accuracy": sum(
          result.partner_reliable for result in results
      ) / len(results),
      "reliable_help_rate": reliable_help_rate,
      "unreliable_help_rate": unreliable_help_rate,
      "selective_cooperation_gap": reliable_help_rate - unreliable_help_rate,
      "help_count": float(counts["help"]),
      "refuse_count": float(counts["refuse"]),
      "defect_count": float(counts["defect"]),
      "helped_interactions": float(len(help_results)),
  }


def _help_rate(results: Sequence[types.InteractionResult]) -> float:
  if not results:
    return 0.0
  return sum(result.action == "help" for result in results) / len(results)
