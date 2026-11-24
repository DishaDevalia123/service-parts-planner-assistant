import pandas as pd
import streamlit as st

import assistant as a

st.set_page_config(page_title="Planner Assistant", page_icon="🔧", layout="wide")

BADGES = {
    "high": ("🟢 High confidence", "success"),
    "medium": ("🟡 Check this", "warning"),
    "low": ("🔴 Not sure", "error"),
    "out_of_scope": ("🔴 Not in the data", "error"),
    "clarify": ("💬 Need one detail", "info"),
}

EXAMPLES = [
    "What is the overall fill rate?",
    "Show the shortage in units for each part",
    "How much of the shortage comes from bulk orders?",
    "How many units did warehouse 4 supply to other warehouses?",
    "How much Part A do we have at warehouse 4?",
    "What is the supplier lead time for Part B?",
]

if "history" not in st.session_state:
    st.session_state.history = []
if "pending" not in st.session_state:
    st.session_state.pending = None


@st.cache_data
def overview():
    _, r = a.run_sql("""SELECT (SELECT as_of_date FROM data_info), COUNT(*), SUM(requested_qty),
                        SUM(filled_qty), ROUND(100.0*SUM(filled_qty)/SUM(requested_qty),1) FROM line_status""")
    return r[0]


with st.sidebar:
    as_of, n_lines, demand, filled, rate = overview()
    st.subheader("Case data")
    st.caption(f"As of {as_of} · 4 parts · 4 warehouses")
    c1, c2 = st.columns(2)
    c1.metric("Order lines", n_lines)
    c2.metric("Demand (units)", demand)
    c1.metric("Filled", filled)
    c2.metric("Fill rate", f"{rate}%")
    st.divider()
    st.subheader("Try a question")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    st.divider()
    if st.button("Clear cache and chat", use_container_width=True):
        a._cache.clear()
        st.session_state.history = []
    st.caption(f"SQL model: {a.SQL_MODEL}\n\nSummary model: {a.SUMMARY_MODEL}")

st.title("Service Parts Planner Assistant")
st.caption("Ask in plain English. Every number comes from a SQL query on the case data, "
           "and every answer shows its query, its checks and how confident it is.")


def render(res, idx):
    label, kind = BADGES.get(res.status, ("⚪", "info"))
    with st.chat_message("assistant"):
        getattr(st, kind)(f"**{label}** — {res.answer}")
        if res.options:
            cols = st.columns(len(res.options))
            for col, opt in zip(cols, res.options):
                if col.button(opt, key=f"opt-{idx}-{opt}"):
                    st.session_state.pending = f"{res.question} ({opt})"
                    st.rerun()
        if res.rows:
            st.dataframe(pd.DataFrame(res.rows, columns=res.columns), hide_index=True, use_container_width=True)
        if res.assumption:
            st.caption(f"Assumption: {res.assumption}")
        with st.expander("SQL and checks"):
            if res.sql:
                st.code(res.sql, language="sql")
            for c in res.checks:
                st.markdown(f"- {c}")
        cost = f"${res.cost:.5f}" if (a.PRICE_IN or a.PRICE_OUT) else "free tier"
        tag = " · ⚡ cached" if res.cached else ""
        st.caption(f"⏱ {res.seconds}s · {res.llm_calls} AI calls · "
                   f"{res.tokens_in + res.tokens_out} tokens · {cost}{tag}")


for i, res in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(res.question)
    render(res, i)

typed = st.chat_input("e.g. Which part has the biggest shortage at warehouse 3?")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.spinner("Writing SQL, running checks..."):
        try:
            res = a.ask(question)
        except Exception as e:
            res = a.Result(question, "low", answer=f"Something went wrong: {e}")
    st.session_state.history.append(res)
    render(res, len(st.session_state.history) - 1)
