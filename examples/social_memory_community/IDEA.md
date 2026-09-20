# Belief-Mediated Bidirectional Social Memory

## One-sentence statement

We study how an agent in a partially observable, long-running social world
turns experiences into beliefs about partner behavior, turns those beliefs into
directed relationship choices, and lets objective outcomes revise the memory.

The contribution is a cross-level update and retrieval loop. It is not the
claim that four independent memory stores are new.

## Core state

For agent i, the social memory state is:

    M_i = (E_i, B_i, R_i, P_i)

* E_i: episodic records of events that i experienced or observed.
* B_i: beliefs about a partner or social norm, including uncertainty.
* R_i: directed relation state. R(i -> j) and R(j -> i) are independent and
  can contain trust, reciprocity, familiarity, hostility, or obligation.
* P_i: the public information visible to i, such as public reputation and
  public norms. It is a local view, not a forced global truth.

Each event keeps the fields needed to explain asymmetry:

    event_id, round_index, observer, actor, subject, target,
    action, outcome, source, visibility, reliability

The first implementation uses Beta-style cooperation belief state, a directed
relation state, and a local public reputation state. More dimensions can be
added after the evaluation protocol is stable.

## The mechanism

The bottom-up path is:

    observed event
      -> update B_i(j)
      -> update R(i -> j)
      -> update P_i when the event is public

The top-down path is:

    current task and candidate set
      -> retrieve candidate-related E_i
      -> expose B_i, R(i -> j), and P_i
      -> choose a partner and action

The environment, rather than an agent, computes the objective outcome and
payoff. The outcome becomes a new observation. This closes the loop:

    observation -> memory update -> social action -> objective result
    -> new observation

Source and visibility are metadata and experimental controls. Rumor and
correction scenarios are robustness conditions, not the primary task.

## What the paper claims

The paper should make three focused claims.

1. Separating behavior prediction (belief) from willingness to interact
   (directed relation) makes social decisions more interpretable than a flat
   event store or one undifferentiated trust score.
2. A typed, partially observable memory loop connects memory quality to
   partner choice, selective cooperation, payoff, and adaptation after a
   partner changes behavior.
3. A common replayable environment and cross-level ablations identify whether
   gains come from belief, directionality, public views, or the update/retrieval
   loop itself.

The work should not claim that event memory, reflection, relationship fields,
or four-way schemas individually appeared for the first time. Generative
Agents, Affordable Generative Agents, summary memories, and relation ledgers
already motivate parts of this design.

## Main research questions

* RQ1: Does the bidirectional loop improve long-horizon partner selection and
  selective cooperation over flat episodic or single-score memories?
* RQ2: Does separating B and R improve prediction calibration and adaptation
  when a partner reverses behavior?
* RQ3: When public observations are noisy, does P help cooperation while
  preserving reasonable private disagreement instead of collapsing every agent
  to one reputation?
* RQ4: What memory and API cost is paid for the behavioral improvement?

## Experimental protocol

The target experiment uses one controlled Concordia-style community:

* 16 agents, 60 rounds, and 10 paired seeds for the main table.
* Four profile types, balanced across the population: cooperator, defector,
  reverser, and conditional partner.
* Sequential execution with programmatic objective outcomes.
* Conditions: stable partners, mid-horizon reversal, partner turnover, and
  partial/noisy visibility.
* Public misinformation and correction are robustness conditions.

The current four-agent, short-run community is a development smoke test. It is
not yet the paper-scale benchmark.

The focused main comparison is:

1. No Memory.
2. Flat Episodic Retrieval.
3. One-way summary memory.
4. Single-score relation ledger.
5. BBSM, the proposed mechanism.
6. Full Context as an information/cost upper bound.

Generative Agents and Affordable Generative Agents should first be adapted to
the same environment and reported in an appendix or an additional comparison,
unless their prompt and cost budgets can be controlled fairly.

## Metrics

Objective social metrics:

* average and total actor payoff;
* partner-selection utility and reliable-partner accuracy;
* reliable-help rate, unreliable-help rate, and selective cooperation gap;
* reciprocity and directed edge persistence;
* recovery lag after a partner reverses behavior.

Mechanism metrics:

* Brier score and calibration error for cooperation probability;
* relation update error against an objective exponentially smoothed target;
* public reputation accuracy and private/public disagreement;
* retrieval hit@k and cross-level context consistency.

Efficiency metrics:

* memory size;
* prompt tokens and model calls;
* latency;
* fallback and parse-failure rate.

Higher cooperation alone is not a success criterion. A method that helps every
partner can be worse than one that selectively helps reliable partners.

## Ablations

Each ablation removes one mechanism while keeping the environment and model
budget fixed:

* remove top-down candidate routing;
* remove bottom-up updates;
* remove B;
* merge directed relations into one undirected score;
* remove P;
* replace multi-dimensional R with one trust score;
* expose the same observations to every agent.

The LLM policy currently uses the LLM only for top-down action selection. The
bottom-up E -> B/R/P update is deterministic and auditable. Learned memory
updates and LoRA are later milestones, after trajectories and replay checks are
stable.

## Code map and status

* memory_controller.py: deterministic E/B/R/P state and bounded decision
  context.
* policies.py: deterministic baselines and common policy interface.
* llm_policy.py: JSON action selection, fallback, diagnostics, and checkpoint
  support.
* community.py: objective environment, mixed visibility, and selectable LLM
  agents.
* metrics.py: current payoff and selective-cooperation metrics.
* replay.py: trajectory loading and objective metric recomputation.
* run.py: reproducible CLI configuration.

The current milestone proves the vertical path and the LLM policy interface.
The next research milestone is a replay protocol with paper-scale scenarios,
one-way/single-relation adapters, richer metrics, and fixed-seed comparisons.
