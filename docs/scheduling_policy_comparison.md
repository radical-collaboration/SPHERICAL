# Why Downstream-First Is Hard to Beat — A Scheduling-Policy Study

**Status:** Findings from the dreamer-campaign ADR benchmark
**Scope:** Comparing four cross-stage scheduling policies (`none`, `rule`,
`bandit`, `llm`) on the 5-stage antigen-discovery cascade, and explaining why a
simple hand-coded heuristic is the one to beat.

---

## TL;DR

We A/B-tested four policies that drive the Campaign Manager's cross-stage
scheduling priority, on a **deadline-yield** objective: *how many terminal
leads does the pipeline produce within a fixed 60 s wall-clock window?*
(Higher is better — this is the realistic HPC framing of a fixed allocation.)

| Policy   | leads (median) | mean | [min..max] | what it does |
|----------|:--:|:--:|:--:|--------------|
| `none`   | 8  | 7.6  | [6..10]  | static priorities (no adaptation) |
| `bandit` | 14 | 14.4 | [11..18] | Thompson-sampling, learns from reward |
| `rule`   | 19 | 19.2 | [18..21] | **deterministic downstream-first** |
| `llm`    | 22 | 21.2 | [16..25] | GPT-4o-mini, told downstream-first is the default |

Two robust conclusions (the small `llm`-vs-`rule` gap is within run-to-run noise;
the others are not):

1. **`rule` (downstream-first) is near-optimal and stable.** Across three
   different objectives and several hand-tuned configs explicitly designed to
   favour adaptation, no policy reliably beats it. The LLM, given full freedom,
   *converges to the same downstream-first ladder* — it cannot out-schedule the
   heuristic, only reproduce it.
2. **The LLM beats the bandit — but not by out-thinking the rule.** It wins
   because it is *told* the good policy in its prompt and so schedules well from
   cycle 0, while the bandit must *discover* that policy through costly
   exploration and is still wandering when the 60 s window closes.

The interesting result is not "LLM wins" — it's *why a domain heuristic beats
both a learned and an LLM scheduler*, and *what the LLM's real advantage is*.

---

## The use case

The campaign is a linear **cascade** (`plot_optimizations.py` display names in
parentheses):

```
s1 ligand_filter  →  s2 ml_affinity  →  s3 docking  →  s4 md_refinement  →  s5 fep_ranking
(Initial Screening) (Active Learning)  (Structural)    (Refinement Sim)     (Affinity Ranking)
```

Three properties of this cascade are what make downstream-first so strong:

- **Only the terminal stage produces value.** A "hit" (lead) is an `s5`
  completion. Work finished at `s1`–`s4` is worthless until it reaches `s5`.
- **The drain path is cheap, the bottleneck is expensive.** In the benchmark
  config (`config_deadline.yaml`): `s1` is a fast CPU screen that floods the
  pipeline; `s2` is the expensive GPU bottleneck (10× the per-task cost);
  `s3`–`s5` are cheap 1 s GPU stages. GPUs are oversubscribed (~24 available vs
  ~50 demanded).
- **Scheduling is a tight control loop.** The CM re-schedules on every replica
  completion (sub-second); the ADR policy nudges priorities every ~1–3 s.

---

## The ADR agent: inputs and outputs

The Campaign Manager keeps owning scheduling, execution, and resources. A
`radical.adr` **agent** (`CampaignOperator` + a `Policy`) runs *alongside* a live
CM in an Observe → Decide → Act loop and only nudges the CM's levers — the ADR
"sacred boundary". All CM coupling lives in `CampaignView` (`src/campaign/adr/view.py`).

### Input — the observation (`CampaignView.observe()`)

Once per decision cycle (every `tick_s` ≈ 1–3 s) the agent receives a snapshot:

**Campaign-level**

| field | meaning |
|-------|---------|
| `cycle` | decision-cycle index |
| `terminal` | id of the deepest (hit-producing) stage |
| `hits` / `target` | terminal completions so far / campaign goal |
| `free_cpus` / `free_gpus` | currently unallocated resources |

**Per stage** (one entry for every workflow group)

| field | meaning |
|-------|---------|
| `running` / `cap` | replicas executing now / max concurrent |
| `pending` | replicas triggered but **waiting** on resources (the backlog) |
| `starved` | `true` when `pending>0` and `running<cap` — a priority boost can help |
| `is_source` | `true` for the root stage (its `pending` is the raw library, not a stall) |
| `finished` | replicas completed |
| `queue_depth` | sharder-buffer depth (candidates not yet dispatched) |
| `bp_state` | backpressure: `HOLD` / `THROTTLE` / `WIDEN` |
| `priority` / `ready` / `deps` | current priority, dependency-readiness, upstreams |

**On real HPC runs only** — when a `TelemetrySubscriber` is wired to
`asyncflow.start_telemetry()`, the observation also carries live hardware
signals: `gpu_util`, `cpu_util`, `mem_util`, `gpu_utils_per_device`,
`task_fail_rate`, `avg_task_duration_s`. (In the local dreamer emulation these
are absent; the policy works without them.)

### Output — the decision (two levers)

Each cycle the policy returns a decision that the operator translates into CM
lever calls. There are two adaptive controls:

1. **Scheduling — `set_priority(stage, p)`.** A `priorities` map
   `{stage_id: priority}` covering every stage. The CM's two-pass greedy
   scheduler orders eligible groups by this number (higher = next free
   GPU/CPU slot). This is how the agent steers *which stage runs next*.
2. **Sharding — `set_batch_size(stage, n)`.** An optional `batch_sizes` map
   `{stage_id: target_size}` that adjusts a downstream stage's sharder
   `target_size` (how many buffered candidates it dispatches per cycle). The
   sharder then applies its fixed backpressure multiplier on top
   (`THROTTLE ×0.5 / HOLD ×1.0 / WIDEN ×1.5`). This is how the agent controls
   *how fast each stage's queue fills*.

A third lever, `trigger(stage, n)` (queue *n* replicas of a dependent stage),
exists on the view but is not used by the scheduling policies here. The decision
also carries a `stop` flag (end the campaign once `hits >= target`).

The structured output object is `ScheduleDecision`
(`src/campaign/adr/policies.py`): `{priorities, batch_sizes, stop}`.

### Which agents we compared

All three are interchangeable `Policy` implementations behind the same
view/operator (`make_scheduling_policy(op, kind=...)`):

| kind | class | how it decides |
|------|-------|----------------|
| `rule` | `DownstreamFirstPolicy` | deterministic: priority = dependency depth, every cycle |
| `bandit` | `BanditSchedulingPolicy` | Thompson-sampling `SchedulingBandit`; reward from downstream backpressure; ranks stages by posterior sample |
| `llm` | `LLMSchedulingPolicy` | an LLM reads the observation as JSON and returns a `ScheduleDecision` |

The **LLM agent** is **GPT-4o-mini**, called through an OpenAI-compatible
endpoint (OpenRouter) via the `instructor` library, which forces the model's
reply into the `ScheduleDecision` schema. It is composed as
`Policy(primary=LLM, fallback=rule)`, so a slow or malformed LLM call degrades
to the deterministic rule for that cycle rather than stalling the campaign. (Any
OpenAI-compatible model works by swapping `cm.adr.base_url` / `model` — e.g. a
local Ollama model — but the results in this document use GPT-4o-mini.)

---

## Why `rule` (downstream-first) is hard to beat

`DownstreamFirstPolicy` assigns priority by dependency depth every cycle:
`s5 > s4 > s3 > s2 > s1`. That single rule is remarkably robust here:

1. **It converts finished work into hits immediately.** Keeping the terminal
   stages highest means any candidate that reaches `s4`/`s5` is run at once,
   rather than waiting behind upstream work. For a "produce leads" objective,
   rushing the leading edge to the exit is exactly right.
2. **It never wastes slots on the deep stages.** A high priority only matters
   when a stage *has* work. When `s5` is empty it simply doesn't run, and the
   CM scheduler's two-pass greedy fill hands those GPUs to whatever stage *does*
   have work — automatically flowing them upstream to the bottleneck. So
   "prioritise the terminal stage" costs nothing when the terminal stage is idle.
3. **It strikes the throughput balance by construction.** The naive "feed the
   bottleneck" instinct (give the expensive `s2` the most GPUs) *backfires*:
   it starves the cheap drain path, so `s2` output piles up at `s3` and never
   becomes hits. We measured this directly — an aggressive bottleneck-boosting
   prompt produced **0 leads** on most runs. Downstream-first avoids the trap:
   the drain stages keep their slots, `s2` gets the (still ample) remainder, and
   leads flow steadily.

In short, downstream-first encodes the **correct inductive bias** for a cascade
with terminal-only value. There is little headroom above it, and the obvious
"smarter" moves (chase the bottleneck) make things worse.

> A `concurrency_floor` on the cheap drain stages (`s3`/`s4`/`s5` reserve a few
> GPUs each) is what makes *any* bottleneck-feeding safe — it guarantees the
> drain path can never be fully starved. Without it, an over-aggressive policy
> can deadlock the pipeline at 0 leads.

---

## Why `llm` ties `rule` but beats `bandit`

The LLM policy (`LLMSchedulingPolicy`, GPT-4o-mini) is given the live per-stage
state (running, pending, `starved`, `is_source`, backpressure) and a prompt that
sets **downstream-first as the strong default**, to be nudged only on clear
evidence of a starved non-source stage.

- **vs. `rule`: a tie.** Inspecting the decision log, the LLM emits the exact
  `s5>s4>s3>s2>s1` ladder on essentially every cycle — it recognises that the
  default is right and rarely deviates. Its leads (median 22) and the rule's
  (median 19) overlap within noise. The honest reading: *the LLM rediscovers the
  heuristic rather than improving on it.*
- **vs. `bandit`: a real, explainable win.** `BanditSchedulingPolicy` starts
  with no knowledge of the cascade and must learn stage value from a
  backpressure-derived reward. Its decision trace wanders through random
  orderings (`s3>s5>s2>s4>s1`, `s4>s2>s3>s1>s5`, …) well past cycle 11 — it is
  still **exploring** when the 60 s deadline closes, so a large fraction of the
  window is spent scheduling sub-optimally. The LLM pays no exploration cost: it
  is *told* the policy and applies it from cycle 0. That is the LLM's genuine
  advantage here — **warm-start from domain knowledge, not superior
  per-cycle reasoning.**

`policy_comparison.png` shows this directly: the `rule` and `llm` priority
panels are flat, stable downstream-first ladders from cycle 0, while the
`bandit` panel is chaotic — its per-stage priorities keep reshuffling across the
whole window as it explores, never settling into the ordering the other two had
from the start.

---

## When *would* adaptive scheduling help?

This study is a fair test of *per-cycle priority assignment in a balanced linear
cascade*, where the answer is "use the heuristic." Adaptive (bandit/LLM)
scheduling is expected to pay off when the structure breaks the assumptions that
make downstream-first optimal:

- **Non-linear topology** (branching/join DAGs) where "depth" no longer uniquely
  orders stages and the right call is to *balance* parallel branches.
- **Shifting/unknown bottlenecks** that the heuristic's fixed ranking cannot
  anticipate (we built a shifting-bottleneck config; downstream-first still won
  on time-to-target because rushing the leading edge dominates).
- **Higher-altitude decisions** — regime detection, replanning, budget
  reallocation — rather than tight-loop priority nudging. This is where an LLM's
  reasoning is more likely to add value than at the per-cycle control level.

The takeaway for SPHERICAL: **keep `rule` as the default scheduler.** Use the
ADR `bandit`/`llm` policies for research and for topologies where the heuristic's
assumptions don't hold — and remember that the LLM's measured edge over the
bandit is its ability to be *seeded* with the right policy, not to invent a
better one.

---

## Reproduce

```bash
cd workflows/run_campaign/dreamer_campaign
export OPENROUTER_API_KEY=sk-or-v1-...        # for the llm policy

# deadline-yield benchmark (leads in a fixed 60 s window; higher = better)
python benchmark_adr.py --config config_deadline.yaml --mode deadline-yield \
    --deadline 60 --policies none rule bandit llm --runs 5 --out benchmark_deadline.json

# outcome plot (leads per policy, with per-run spread)
python plot_deadline_yield.py --results benchmark_deadline.json --out plots/deadline_yield.png

# decision-trace plot (why llm > bandit: stable ladder vs exploration)
python plot_policy_comparison.py \
    adr-logs/rule-run0.jsonl adr-logs/bandit-run0.jsonl adr-logs/llm-run0.jsonl \
    --out plots/policy_comparison.png
```

The time-to-target objective (wall-clock to the *N*-th lead, the original
metric) is the default mode (`--mode time-to-target`); on it `rule` wins
outright because rushing the leading edge to the exit is precisely
downstream-first.
