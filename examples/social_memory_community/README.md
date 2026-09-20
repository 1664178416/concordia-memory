# Social Memory Community

This is the first controlled milestone for the social-memory project. The
environment and objective outcomes remain deterministic, while an optional LLM
policy can make the top-down partner/action decision.

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
`bidirectional_social`, and `llm_bidirectional`. The deterministic policy
updates episodic, belief, directed-relation, and public-view state; the LLM
policy reads the same bounded context and falls back to the deterministic
decision when a request or parse fails.

## Calling an OpenAI-compatible model

The repository also exposes a dependency-free wrapper for chat-completions
compatible services. VectorEngine can be used through the `vectorengine`
`api_type`; the key is read only from the current process environment.

```powershell
$env:VECTORENGINE_API_KEY = "<your-key>"
$env:VECTORENGINE_BASE_URL = "https://api.vectorengine.ai/v1"
$env:VECTORENGINE_MODEL = "gpt-4o-mini"
python -m examples.social_memory_community.vectorengine_smoke
```

Run one model-driven interaction with all four memory views in the prompt:

```powershell
python -m examples.social_memory_community.llm_one_round
```

This command performs one E/B/R/P -> model choice -> environment outcome ->
memory update cycle and prints the structured state before and after the call.

Run the formal policy path with only Alice using the LLM (the other agents use
the deterministic controller):

```powershell
python -m examples.social_memory_community.run --policy llm_bidirectional --rounds 1 --llm_agents Alice --model gpt-4o-mini
```

The consolidated research specification is in IDEA.md. It distinguishes the
cross-level BBSM mechanism from the four state views and separates the current
development smoke test from the paper-scale evaluation protocol.

The same wrapper can be used with another provider through
`api_type="openai_compatible"`, passing its `api_base` and `api_key` to
`language_model_setup`. Keys are never written to experiment outputs.

This environment is intentionally separate from the existing resource-dilemma
example. The next research milestone is replayable paper-scale scenarios and
cross-level ablations before adding a Concordia component.
