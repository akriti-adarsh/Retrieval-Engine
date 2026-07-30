"""Streamlit demo client.

This talks to the running API over HTTP and never imports the retrieval pipeline. That is
deliberate: the demo exists to show what a consumer of this service actually sees, and a UI
that reached past the API into the library would prove nothing about the API.

The layout puts the answer and its numbered sources next to each other, because the claim
this project makes is that every sentence can be checked against a passage. A demo that shows
a confident answer without the passages behind it is the thing this repository argues against.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("RE_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 180.0

st.set_page_config(page_title="retrieval-engine", page_icon="*", layout="wide")


def call_api(base_url: str, payload: dict[str, Any], debug: bool) -> tuple[dict[str, Any], str]:
    """Query the API, returning (body, error_message).

    Every failure comes back as a readable message rather than a traceback: a demo that shows
    a stack trace when the server is down tells a viewer nothing about the service.
    """
    url = f"{base_url.rstrip('/')}/v1/query" + ("?debug=true" if debug else "")
    try:
        response = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    except httpx.ConnectError:
        return {}, f"Cannot reach the API at {base_url}. Is it running?"
    except httpx.TimeoutException:
        return {}, "The API did not respond in time. The first query loads the models."
    except httpx.RequestError as exc:
        return {}, f"Request failed: {type(exc).__name__}: {exc}"

    try:
        body = response.json()
    except ValueError:
        return {}, f"HTTP {response.status_code}: the response was not JSON."

    if response.status_code >= 400:
        # The API returns one error envelope for every failure, so this is the only shape to
        # handle.
        error = body.get("error", {}) if isinstance(body, dict) else {}
        code = error.get("code", "unknown_error")
        message = error.get("message", "no message")
        return {}, f"{code}: {message}"
    return body, ""


def render_answer(body: dict[str, Any]) -> None:
    """The answer, with refusal treated as a first-class outcome rather than a failure."""
    answer_type = body.get("answer_type", "unknown")
    answer = body.get("answer", "")

    if answer_type == "refused":
        # Refusing is correct behaviour when the evidence is weak, so it is shown as an
        # informational result, not an error.
        st.info(f"**The service declined to answer.**\n\n{answer}")
        st.caption(
            "This is a deliberate outcome, not a failure. The retrieval scores were below the "
            "configured confidence threshold, so the service reports that rather than guessing."
        )
        return

    st.markdown(answer)
    label = {"generated": "Written by the language model", "extractive": "Extracted verbatim"}
    st.caption(
        f"{label.get(answer_type, answer_type)} - model `{body.get('model', '?')}`, "
        f"prompt `{body.get('prompt_version', '?')}`"
    )
    if answer_type == "extractive":
        st.caption(
            "The model server was unreachable, so every sentence above is quoted directly from "
            "a source. This path cannot hallucinate."
        )


def render_sources(body: dict[str, Any]) -> None:
    """Numbered sources, so each [n] marker in the answer can be checked against its passage."""
    sources = body.get("sources", [])
    citations = {c.get("marker") for c in body.get("citations", []) if c.get("resolved")}
    unresolved = [c.get("marker") for c in body.get("citations", []) if not c.get("resolved")]

    if unresolved:
        st.error(
            f"The answer cited source(s) {unresolved} that were never retrieved. "
            "That is a defect, and it is surfaced rather than hidden."
        )

    if not sources:
        st.write("No sources were retrieved.")
        return

    for source in sources:
        index = source.get("index")
        cited = " (cited)" if index in citations else ""
        section = " > ".join(source.get("section_path") or []) or source.get("title", "")
        with st.expander(
            f"[{index}]{cited}  {source.get('title', '')[:70]}  ({source.get('score', 0):.3f})",
            expanded=index == 1,
        ):
            st.caption(f"`{source.get('doc_id')}` - {section}")
            st.write(source.get("text", ""))


def render_grounding(body: dict[str, Any]) -> None:
    """The grounding report, shown whether or not it passed."""
    grounding = body.get("grounding", {})
    sentences = grounding.get("sentences", [])
    flagged = grounding.get("flagged_sentences", [])

    if grounding.get("grounded"):
        st.success(f"Grounded. Every sentence scored at or above {grounding.get('threshold')}.")
    else:
        st.warning(
            f"Not fully grounded: {len(flagged)} sentence(s) fell below "
            f"{grounding.get('threshold')} similarity to any retrieved passage."
        )

    if sentences:
        st.dataframe(
            [
                {
                    "supported": "yes" if s.get("grounded") else "NO",
                    "similarity": round(s.get("max_similarity", 0.0), 3),
                    "sentence": s.get("sentence", ""),
                }
                for s in sentences
            ],
            width="stretch",
            hide_index=True,
        )


def render_timings(body: dict[str, Any]) -> None:
    timings = body.get("timings", {})
    stages = ["dense", "lexical", "fusion", "rerank", "generation", "grounding"]
    rows = [{"stage": stage, "ms": round(timings.get(f"{stage}_ms", 0.0), 1)} for stage in stages]
    st.bar_chart(rows, x="stage", y="ms", height=220)
    st.caption(f"Total {timings.get('total_ms', 0.0):.0f} ms")


def render_debug(body: dict[str, Any]) -> None:
    debug = body.get("debug")
    if not debug:
        st.write("Enable the debug toggle to see the retrieval trail.")
        return

    st.write("**Candidate counts per stage**")
    st.json(debug.get("candidate_counts", {}))

    expanded = debug.get("expanded_queries", [])
    if len(expanded) > 1:
        st.write("**Expanded queries**")
        for query in expanded:
            st.write(f"- {query}")

    stages = debug.get("stages", [])
    movement = debug.get("rank_movement", [])
    if stages:
        st.write("**Per-source stage scores**")
        st.dataframe(
            [
                {
                    "final": stage.get("final_rank"),
                    "fused": stage.get("fused_rank"),
                    "moved": movement[i] if i < len(movement) else None,
                    "dense": stage.get("dense_score"),
                    "lexical": stage.get("lexical_score"),
                    "rerank": stage.get("rerank_score"),
                }
                for i, stage in enumerate(stages)
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "`moved` is places gained by reranking. A positive number means the "
            "cross-encoder promoted that passage above where fusion had placed it."
        )

    st.write("**Exact configuration that produced this**")
    st.json(debug.get("config", {}))


def main() -> None:
    st.title("retrieval-engine")
    st.caption(
        "Hybrid dense and BM25 retrieval, reciprocal rank fusion, cross-encoder reranking, "
        "and grounding verification. This page is an ordinary HTTP client of the API."
    )

    with st.sidebar:
        st.header("Settings")
        base_url = st.text_input("API base URL", value=DEFAULT_API)
        top_k = st.slider("Sources to return (top_k)", min_value=1, max_value=20, value=5)
        debug = st.toggle("Show retrieval trail (debug)", value=False)
        st.divider()
        st.caption(
            "Start the API with:\n\n"
            "`uv run uvicorn retrieval_engine.api.app:create_app --factory --port 8000`"
        )
        if st.button("Check readiness"):
            try:
                ready = httpx.get(f"{base_url.rstrip('/')}/health/ready", timeout=120.0).json()
                st.json(ready)
            except httpx.RequestError as exc:
                st.error(f"Cannot reach the API: {type(exc).__name__}")

    question = st.text_input(
        "Question",
        placeholder="how does reciprocal rank fusion combine two ranked lists?",
    )
    if not st.button("Ask", type="primary") or not question.strip():
        st.stop()

    with st.spinner("Retrieving, then verifying the answer against the sources..."):
        body, error = call_api(base_url, {"query": question, "top_k": top_k}, debug)

    if error:
        st.error(error)
        st.stop()

    answer_column, sources_column = st.columns([3, 2], gap="large")
    with answer_column:
        st.subheader("Answer")
        render_answer(body)
        st.subheader("Grounding")
        render_grounding(body)
    with sources_column:
        st.subheader("Sources")
        st.caption("Check each [n] in the answer against the passage it points at.")
        render_sources(body)

    st.divider()
    timing_tab, debug_tab, raw_tab = st.tabs(["Timings", "Retrieval trail", "Raw response"])
    with timing_tab:
        render_timings(body)
    with debug_tab:
        render_debug(body)
    with raw_tab:
        st.json(body)


if __name__ == "__main__":
    main()
