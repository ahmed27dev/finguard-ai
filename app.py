"""
app.py - FinGuard AI dashboard.

Presents what the pipeline found: each flagged concern, the explanation, the
passage it was drawn from with its page, and whether verification confirmed
the explanation against that passage.

Three display decisions worth knowing about:

  UNEXPLAINED IS SHOWN, NOT HIDDEN
      An anomaly the filing never accounts for is the most useful thing here,
      not a gap in the output. Hiding it would invert the point of the tool.

  THE ARITHMETIC IS SHOWN ALONGSIDE THE CLAIM
      Every flag displays the figures it was computed from, so a reader can
      recompute it rather than trust it.

  VERIFICATION IS A LABEL, NOT A GATE
      A failed check is displayed with its reason rather than suppressed.
      Hiding failures would make the checker look infallible.

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

PROCESSED = Path("data/processed")
FILINGS = Path("data/filings")

st.set_page_config(page_title="FinGuard AI", page_icon="•", layout="wide")


# ---------------------------------------------------------------------------
# Caching. Streamlit re-runs this whole script on every interaction, so
# without these the models reload and the filing re-parses on each click.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def warm_models() -> None:
    from retrieval.indexer import get_model as bi
    from retrieval.reranker import get_model as ce
    bi()
    ce()


@st.cache_data(show_spinner=False)
def load_verified(filing_id: str) -> list[dict] | None:
    path = PROCESSED / f"{filing_id}.verified.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_metrics(filing_id: str) -> list[dict]:
    path = PROCESSED / f"{filing_id}.metrics.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def md_safe(text: str) -> str:
    """
    Escape dollar signs before rendering.

    Streamlit reads $...$ as LaTeX math, so a passage containing two dollar
    amounts has everything between them swallowed into an equation. On a
    Coca-Cola filing "$6,805 million and $11,599 million" rendered as italic
    mathematics with the words between them lost. Silently corrupting figures
    is not acceptable in a tool whose whole claim is verifiable numbers.
    """
    return text.replace("$", r"\$")


def available_filings() -> list[str]:
    return sorted(p.stem.replace(".verified", "")
                  for p in PROCESSED.glob("*.verified.json"))


def run_full_pipeline(filing_id: str, force: bool = False) -> None:
    """Run every stage, reporting progress. Slow - a minute or more."""
    from pipeline import investigate
    from flagging.generator import explain
    from flagging.faithfulness_check import verify

    progress = st.progress(0.0, "Parsing filing...")
    anomalies = investigate(filing_id, force)
    progress.progress(0.5, f"{len(anomalies)} anomalies found. Explaining...")

    explained = [explain(a, a.get("evidence", [])) for a in anomalies]
    progress.progress(0.8, "Verifying explanations...")

    verified = [verify(e) for e in explained]
    (PROCESSED / f"{filing_id}.verified.json").write_text(
        json.dumps(verified, indent=2), encoding="utf-8")

    progress.progress(1.0, "Done")
    load_verified.clear()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("FinGuard AI")
st.sidebar.caption("Anomaly detection over 10-K filings, with verified "
                   "citation-backed explanations.")

filings = available_filings()

# After analysing an upload, show it. Without this the dropdown keeps
# whatever was selected before, so a filing that processed correctly looks
# like nothing happened.
if "filing_choice" in st.session_state and \
        st.session_state["filing_choice"] not in filings:
    del st.session_state["filing_choice"]

choice = st.sidebar.selectbox(
    "Analysed filings", filings or ["(none yet)"],
    key="filing_choice", disabled=not filings)

st.sidebar.divider()
uploaded = st.sidebar.file_uploader(
    "Or analyse a new filing", type=["htm", "html"],
    help="SEC EDGAR serves 10-K filings as HTML. Download the primary "
         "document from the filing index page.")

if uploaded is not None:
    name = Path(uploaded.name).stem
    FILINGS.mkdir(parents=True, exist_ok=True)
    (FILINGS / f"{name}.htm").write_bytes(uploaded.getvalue())
    if st.sidebar.button(f"Analyse {name}", type="primary"):
        warm_models()
        run_full_pipeline(name, force=True)
        st.session_state["filing_choice"] = name
        st.rerun()

if filings and st.sidebar.button("Re-run analysis"):
    warm_models()
    run_full_pipeline(choice, force=True)
    st.rerun()

st.sidebar.divider()
show = st.sidebar.radio(
    "Show", ["All concerns", "Explained by the filing", "Unexplained"],
    help="Unexplained means the arithmetic found a real change and the "
         "filing contains no passage accounting for it.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if not filings:
    st.title("FinGuard AI")
    st.info("No filings analysed yet. Upload a 10-K in the sidebar, or run "
            "`python pipeline.py <filing-id> --investigate` first.")
    st.stop()

items = load_verified(choice) or []
metrics = {m["metric_id"]: m for m in load_metrics(choice)}

explained = [i for i in items if i.get("supported")]
unexplained = [i for i in items if not i.get("supported")]
verified_ok = [i for i in items if i.get("verdict") == "supported"]

st.title(choice)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Concerns flagged", len(items))
c2.metric("Explained by filing", len(explained))
c3.metric("Unexplained", len(unexplained))
c4.metric("Verified faithful", len(verified_ok))

st.caption("Concerns are detected by arithmetic over the filing's own tables. "
           "No language model decides what is flagged.")
st.divider()

selected = {"All concerns": items,
            "Explained by the filing": explained,
            "Unexplained": unexplained}[show]

if not selected:
    st.info("Nothing in this category for this filing.")

for item in selected:
    citation = item.get("citation")
    verdict = item.get("verdict")

    if verdict == "supported":
        badge, colour = "verified", "green"
    elif verdict == "not_supported":
        badge, colour = "verification failed", "red"
    else:
        badge, colour = "no explanation in filing", "orange"

    with st.container(border=True):
        head, tag = st.columns([5, 1])
        head.markdown(f"**{md_safe(item['finding'])}**")
        tag.markdown(f":{colour}[{badge}]")

        body = item["explanation"]
        if body.startswith(item["finding"]):
            body = body[len(item["finding"]):].lstrip(". ")
        st.write(md_safe(body or "No explanation available."))

        # The figures behind the flag, so a reader can recompute it.
        ids = item.get("metric_ids") or []
        rows = [metrics[i] for i in ids if i in metrics]
        if rows:
            with st.expander("Figures this was computed from"):
                for m in rows:
                    periods = sorted(m["values"], reverse=True)
                    series = "  ".join(
                        f"**{p}** {m['values'][p]:,.0f}" for p in periods)
                    unit = f" ({m['scale']})" if m.get("scale") else ""
                    st.markdown(md_safe(f"{m['label']}{unit} — {series}"))
                    st.caption(f"page {m['page']} · {m['section']}")

        if citation:
            with st.expander(f"Source — page {citation['page']}"):
                st.markdown(f"> {md_safe(citation['text_span'])}")
                st.caption(f"{citation['section']} · {citation['chunk_id']}")
                score = item.get("faithfulness_score")
                if score is not None:
                    st.caption(f"Entailment {score:.2f} "
                               f"({item.get('nli_label')})")
                for f in item.get("failures") or []:
                    st.warning(f)

st.divider()
st.caption("FinGuard flags areas for review. It is not an audit, a fraud "
           "detector, or investment advice; every concern needs a human "
           "judgement.")