# Sentinel-Eval

Interpretability-first evaluation harness for agentic systems. It diagnoses *why* an agent failed instead of
only scoring it, and stress-tests agents with prompt-injection attacks.

| Piece | Module | What it does |
|---|---|---|
| Traces | `trace.py` | Structured per-step JSON traces (JSONL) |
| Taxonomy | `taxonomy.py` | 7 report categories + SQL-logic, medical-code lookup, injection-compliance |
| Judges | `judge.py` | Binary verdicts, jury majority vote, order-swapped pairwise comparison |
| Attribution | `attribution.py` | "first wrong step" localisation (all-at-once and step-by-step) |
| Counterfactual replay | `replay.py` | Patch a suspected step with the reference output, re-run, check whether the failure disappears; builds the Causal Failure Graph |
| Who&When eval | `whowhen.py` | Agent-level / step-level accuracy on the 184 annotated failures |
| Testbed | `ehr.py`, `agent.py` | Synthetic SQLite EHR (no credentialing), ReAct SQL agent, anomaly injection, injectable notes |
| Attacks | `attacks.py` | Indirect prompt injection; reports attack-success rate *and* task success (catches "refuse everything") |
| Regression | `regression.py` | Diagnosed failures become permanent cases; `sentinel-eval regress` exits 1 on regression |
| Dashboard | `dashboard/app.py` | Streamlit trace viewer with the failure graph and intervention table |
| Attacker training | `scripts/train_attacker_grpo.py` | Single-turn QLoRA + GRPO attacker (TRL) |

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                          # 11 tests, fully offline
sentinel-eval demo                 # buggy agent -> traces -> root cause by replay -> regression cases
sentinel-eval regress --path runs/demo/regressions.jsonl            # fixed agent: passes
sentinel-eval regress --bug latest mean --path runs/demo/regressions.jsonl   # buggy agent: exit 1
sentinel-eval attack               # injection ASR vs task success; add --hardened to compare
pip install streamlit && streamlit run dashboard/app.py -- runs/demo
```

Real models: `sentinel-eval whowhen --subset algo --method step --judges anthropic:<model>,anthropic:<model>,ollama:<model>`
(needs `ANTHROPIC_API_KEY`; OpenAI-compatible servers such as Ollama work via `ollama:<model>` / `openai:<model>@<url>`).
Plug in your own agent with `--agent module:factory` where `factory(ehr) -> ReActAgent`.

## Status (what has and has not been measured)

Measured here, offline:
- Random baseline on Who&When (seed 0): **algo** agent 31.7% / step 8.7%; **hand** agent 12.1% / step 5.2%.
- Counterfactual replay localises the injected SQL bug to the query step in 24/24 failed runs of the simulated agent.
- Simulated defender: template attacks reach 17% ASR; the hardened variant 0% with no loss of task success.

**Not yet run:** the LLM-judge numbers on Who&When (no API key was available), the GRPO attacker training, and any
MedAgentBench / FHIR-AgentBench agent runs. The simulated agent (`sim.py`) is a deterministic policy used to verify
the machinery against known root causes; its numbers say nothing about real LLMs.

## Data notes

- Downloaded to `data/raw/` (git-ignored): Who&When (`sentinel-eval download`), `medagentbench/`, `fhir_agentbench/` (git clones).
- Who&When step labels are 0-indexed into the log. In Algorithm-Generated, 123/126 labelled agents match the agent at the labelled step.
  Hand-Crafted is messier: 3/58 labels are out of range and only 34/55 in-range labels match. Raw labels are used as published.
- MIMIC-IV is credentialed; its terms restrict sending data to external LLM APIs. Nothing here needs it.
- Counterfactual replay needs a re-executable agent, so it runs on the EHR testbed, not on the static Who&When logs.
