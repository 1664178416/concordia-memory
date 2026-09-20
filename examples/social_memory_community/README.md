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

The same wrapper can be used with another provider through
`api_type="openai_compatible"`, passing its `api_base` and `api_key` to
`language_model_setup`. Keys are never written to experiment outputs.

This environment is intentionally separate from the existing resource-dilemma
example. The next milestone will add a Concordia component and an LLM-backed
memory controller using the same policy interface.
