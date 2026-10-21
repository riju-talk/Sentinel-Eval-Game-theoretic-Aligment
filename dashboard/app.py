"""Trace viewer + Causal Failure Graph.  Run: streamlit run dashboard/app.py -- runs/demo"""
import json
import sys
from pathlib import Path

import streamlit as st

from sentinel_eval.trace import load_traces

run = Path(sys.argv[-1] if len(sys.argv) > 1 and not sys.argv[-1].endswith(".py") else "runs/demo")
st.set_page_config(page_title="Sentinel-Eval", layout="wide")
st.title("Sentinel-Eval trace viewer")

traces = load_traces(run / "traces.jsonl")
graphs = {g["task_id"]: g for g in json.loads((run / "graphs.json").read_text())} if (run / "graphs.json").exists() else {}

ok = sum(bool(t.success) for t in traces)
c1, c2, c3 = st.columns(3)
c1.metric("Runs", len(traces))
c2.metric("Success", f"{ok}/{len(traces)}")
c3.metric("Root-caused by replay", sum(g["root_step"] is not None for g in graphs.values()))

only_fail = st.checkbox("Only failures", True)
pool = [t for t in traces if not (only_fail and t.success)]
if not pool:
    st.success("No failures in this run.")
    st.stop()
t = st.selectbox("Run", pool, format_func=lambda t: f"{'PASS' if t.success else 'FAIL'}  {t.task_id}")

left, right = st.columns([3, 2])
g = graphs.get(t.task_id)
with left:
    st.subheader("Trace")
    st.caption(f"{t.task}  |  gold: {t.gold}  |  answer: {t.final_answer}")
    for s in t.steps:
        root = g and s.idx == g["root_step"]
        with st.container(border=True):
            st.markdown(f"**{'ROOT CAUSE  ' if root else ''}step {s.idx}** `{s.agent}/{s.kind}`")
            st.code(s.content, language="sql" if s.kind == "tool_call" else None)
with right:
    if g:
        st.subheader("Causal failure graph")
        st.graphviz_chart(g["dot"])
        st.subheader("Interventions (counterfactual replay)")
        st.table([{"patched step": i["step"], "failure resolved": i["fixed"], "new answer": i["new_answer"]}
                  for i in g["interventions"]])
