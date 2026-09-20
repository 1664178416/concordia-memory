# Social Memory Community

This is the first, LLM-free milestone for the social-memory project. It keeps
the environment deterministic so that memory effects can be tested before an
LLM acting layer and a Concordia Game Master are added.

Each round every agent chooses a partner and one action (`help`, `refuse`, or
`defect`). The environment resolves the outcome from a hidden partner profile.
Agents receive direct and/or public observations, which are then passed to one
of the memory policies.

Run a smoke experiment from the repository root:

```powershell
python -m examples.social_memory_community.run --policy flat_episodic --rounds 8 --seed 42
```

The command writes `trajectory.json`, `metrics.json`, and `checkpoint.json`.
The current policies are `no_memory`, `full_context`, `flat_episodic`, and
`bidirectional_social`. The latter is the first deterministic E/B/R/P memory
controller: observations update episodic, belief, directed-relation, and
public-view state, and the resulting state guides partner selection.

This environment is intentionally separate from the existing resource-dilemma
example. The next milestone will add a Concordia component and an LLM-backed
memory controller using the same policy interface.
