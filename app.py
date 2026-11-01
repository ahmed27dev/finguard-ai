"""
app.py - FinGuard AI dashboard.

Presents what the pipeline found: each flagged concern, the explanation, the
passage it was drawn from with its page, and whether verification confirmed
the explanation against that passage.

Presentation follows the subject matter. A 10-K is a working paper, so
concerns are set as ruled entries rather than cards, figures are right
aligned in tabular numerals under year columns, and citations name a page
and an Item the way an analyst would write them.

Three decisions worth knowing about:

  THE FIGURES ARE THE HERO
      Every concern shows the series it was computed from, as a statement
      table. The claim this project makes is that flags are recomputable by
      hand, so the numbers cannot sit behind a collapsed panel.

  UNEXPLAINED IS SHOWN, NOT HIDDEN
      An anomaly the filing never accounts for is the most useful thing
      here. Hiding it would invert the point of the tool.

  VERIFICATION IS A LABEL, NOT A GATE
      A failed check is displayed with its reason rather than suppressed.
      Hiding failures would make the checker look infallible.

    streamlit run app.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

PROCESSED = Path("data/processed")
FILINGS = Path("data/filings")

st.set_page_config(page_title="FinGuard", page_icon="§", layout="wide")


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Spectral:wght@400;500;600&display=swap');

:root {
  --ink:        #17212B;
  --ink-soft:   #5A6B7B;
  --ink-faint:  #8E9CA8;
  --rule:       #E2E7EB;
  --rule-firm:  #C4CDD5;
  --used:       #DCE6F2;
  --used-edge:  #B9CCE3;
  --verified:   #0F6B4F;
  --unexplained:#8A5115;
  --failed:     #9B1C1C;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', system-ui, sans-serif; }

/* Streamlit's default top padding pushes everything below the fold. */
.block-container { padding-top: 2.4rem; max-width: 60rem; }

/* --- masthead --- */
.masthead { border-bottom: 2px solid var(--ink); padding-bottom: .7rem; margin-bottom: .5rem; }
.masthead h1 {
  font-family: 'Spectral', Georgia, serif;
  font-weight: 600; font-size: 2.1rem; line-height: 1.1;
  color: var(--ink); margin: 0; letter-spacing: -0.01em;
}
/* Document identity, lifted from the filing's own dei tags. A labelled
   strip rather than tiles: it carries facts found nowhere else on the
   page, so it adds information instead of restating the line below it. */
.facts {
  display: flex; flex-wrap: wrap; gap: 0 2.4rem;
  margin: .9rem 0 0 0; padding: 0;
}
.facts .fact { margin: 0; }
.facts dt {
  font-size: .68rem; color: var(--ink-faint);
  letter-spacing: .02em; margin-bottom: .1rem;
}
.facts dd {
  margin: 0; font-size: .9rem; color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.masthead .scope {
  font-size: .9rem; color: var(--ink-soft); margin-top: .8rem;
  font-variant-numeric: tabular-nums;
}
.masthead .basis {
  font-size: .82rem; color: var(--ink-faint); margin-top: .9rem;
  margin-bottom: 0; max-width: 46rem; line-height: 1.5;
}

/* --- one concern --- */
.entry {
  border-top: 1px solid var(--rule);
  padding: 1.15rem 0 .95rem 1.1rem;
  border-left: 6px solid var(--rule-firm);
  padding-left: 1.1rem;
}
.entry--verified    { border-left-color: var(--verified); }
.entry--unexplained { border-left-color: var(--unexplained); }
.entry--failed      { border-left-color: var(--failed); }

.entry__head {
  display: flex; align-items: center; gap: .9rem;
  margin-bottom: .4rem; font-size: .78rem; color: var(--ink-soft);
}
.entry__rule { font-weight: 500; }
.entry__verdict { font-weight: 500; }
.entry--verified    .entry__verdict { color: var(--verified); }
.entry--unexplained .entry__verdict { color: var(--unexplained); }
.entry--failed      .entry__verdict { color: var(--failed); }

/* Severity, named rather than drawn: a bare bar means nothing on paper. */
.entry__sev { margin-left: auto; color: var(--ink-faint); }

.entry__finding {
  font-family: 'Spectral', Georgia, serif;
  font-size: 1.16rem; line-height: 1.34; font-weight: 500;
  color: var(--ink); margin: 0 0 .4rem 0;
  font-variant-numeric: tabular-nums;
}
.entry__account {
  font-size: .94rem; line-height: 1.5; color: var(--ink);
  margin: 0 0 .75rem 0; max-width: 44rem;
}
.entry__account.is-absent { color: var(--ink-soft); }

/* --- the figures: a statement, not a list --- */
/* Built from divs, not <table>: Streamlit styles bare table elements and
   its vertical rules were cutting across the statement. display:table on
   divs keeps the automatic column sizing a statement needs. */
.ledger {
  display: table; border-collapse: collapse;
  width: 100%; max-width: 44rem;
  margin: .2rem 0 .85rem 0; font-size: .86rem;
}
.ledger__r { display: table-row; }
.ledger__l, .ledger__v {
  display: table-cell; border-bottom: 1px solid var(--rule);
  padding: .26rem 0 .26rem 1.6rem; vertical-align: bottom;
}
.ledger__l {
  text-align: left; padding-left: 0; color: var(--ink-soft); width: 100%;
}
.ledger__v {
  text-align: right; white-space: nowrap; padding-right: .45rem;
  font-variant-numeric: tabular-nums; color: var(--ink);
}
/* One tint, one meaning: these are the cells the arithmetic used. Slate
   rather than the verdict greens and ambers, so it cannot be mistaken
   for a judgement, and it survives greyscale printing. */
.ledger__v.is-used { background: var(--used); }
/* An edge on the block as well as a fill: a tint alone can vanish when a
   screenshot is scaled down and compressed, and the shading carries the
   claim that these are the figures behind the finding. */
.ledger__r--head .ledger__v.is-used { border-bottom-color: var(--used-edge); }
.ledger__r--head .ledger__v.is-used { color: var(--ink); }
.ledger__r--head .ledger__l,
.ledger__r--head .ledger__v {
  font-size: .78rem; font-weight: 500; color: var(--ink-faint);
  padding-top: 0; padding-bottom: .35rem;
}
.ledger .unit { font-size: .76rem; color: var(--ink-faint); }
.formula {
  font-size: .8rem; color: var(--ink-faint); font-style: italic;
  margin: -.55rem 0 .85rem 0; max-width: 44rem;
}
.ledger__src {
  display: table-cell; border-bottom: none; padding-top: .3rem;
  font-size: .76rem; color: var(--ink-faint);
}

/* --- the cited passage --- */
details.cite { margin-top: .2rem; }
details.cite > summary {
  cursor: pointer; font-size: .84rem; color: var(--ink-soft);
  list-style: none; padding: .2rem 0;
}
details.cite > summary::-webkit-details-marker { display: none; }
details.cite > summary::before { content: '+ '; color: var(--ink-faint); }
details.cite[open] > summary::before { content: '– '; }
details.cite > summary:hover { color: var(--ink); }
.cite blockquote {
  font-family: 'Spectral', Georgia, serif;
  font-size: .95rem; line-height: 1.6; color: var(--ink);
  border-left: 2px solid var(--rule-firm);
  margin: .6rem 0 .5rem 0; padding: 0 0 0 .95rem; max-width: 44rem;
}
.cite .prov {
  font-size: .8rem; color: var(--ink-faint);
  line-height: 1.5; max-width: 44rem; margin: 0;
}
.cite .flaw { font-size: .8rem; color: var(--failed); margin-top: .35rem; }

.footnote {
  border-top: 1px solid var(--rule); margin-top: 2rem; padding-top: .9rem;
  font-size: .8rem; color: var(--ink-faint); max-width: 44rem; line-height: 1.5;
}
</style>
"""


# ---------------------------------------------------------------------------
# Caching. Streamlit re-runs this script on every interaction, so without
# these the models reload and the filing re-parses on each click.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def warm_models() -> None:
    from retrieval.indexer import get_model as bi
    from retrieval.reranker import get_model as ce
    bi()
    ce()


@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> list | dict | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_verified(filing_id: str) -> list[dict]:
    return load_json(str(PROCESSED / f"{filing_id}.verified.json")) or []


def load_metrics(filing_id: str) -> dict[str, dict]:
    rows = load_json(str(PROCESSED / f"{filing_id}.metrics.json")) or []
    return {m["metric_id"]: m for m in rows}


def filing_scale(filing_id: str) -> tuple[int, int]:
    """Chunks indexed and pages covered, for the scope line."""
    chunks = load_json(str(PROCESSED / f"{filing_id}.chunks.json")) or []
    pages = [c.get("page", 0) for c in chunks if c.get("page")]
    return len(chunks), (max(pages) if pages else 0)


@st.cache_data(show_spinner=False)
def filing_identity(filing_id: str) -> dict[str, str]:
    """Registrant, document type and period, read from the filing itself.

    Every inline XBRL filing tags these under the SEC's dei namespace, so
    they can be lifted without a lookup table and without hardcoding a
    company name - an uploaded filing gets a real masthead too. Read by
    regex rather than BeautifulSoup because these files run to megabytes
    and only five short values are wanted. Any failure returns nothing and
    the page falls back to the filing id.
    """
    import re

    try:
        source = next(iter(FILINGS.glob(f"{filing_id}.*")), None)
        if source is None:
            return {}
        raw = source.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}

    wanted = {
        "registrant": "EntityRegistrantName",
        "symbol": "TradingSymbol",
        "doc_type": "DocumentType",
        "fiscal_year": "DocumentFiscalYearFocus",
        "period_end": "DocumentPeriodEndDate",
    }

    def clean(fragment: str) -> str:
        """Tags out, entities decoded, whitespace collapsed.

        Entities matter here: filings write the date as January&#160;25,
        and an undecoded &#160; survives escaping and shows up raw.
        """
        return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())

    found: dict[str, str] = {}
    for key, tag in wanted.items():
        hit = re.search(rf'name=["\']dei:{tag}["\'][^>]*>', raw, re.I)
        if not hit:
            continue
        window = raw[hit.end():hit.end() + 400]
        # Most values end at the element close. A date does not: filings
        # split it across nested elements ("January 25" then ", 2026"),
        # so the close of the first one lands mid-value. For the date the
        # whole window is read and the date pulled out of the text.
        if key != "period_end":
            window = re.split(r"</ix:non", window, maxsplit=1, flags=re.I)[0]
        value = clean(window)
        if value:
            found[key] = value[:120]

    end = found.get("period_end", "")
    iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", end)
    named = re.search(
        r"(January|February|March|April|May|June|July|August|September"
        r"|October|November|December)\s+(\d{1,2})\s*,?\s*(\d{4})", end, re.I)
    try:
        from datetime import date
        if iso:
            y, m, d = (int(g) for g in iso.groups())
        elif named:
            months = ["january", "february", "march", "april", "may", "june",
                      "july", "august", "september", "october", "november",
                      "december"]
            m = months.index(named.group(1).lower()) + 1
            d, y = int(named.group(2)), int(named.group(3))
        else:
            raise ValueError
        found["period_end"] = date(y, m, d).strftime("%d %b %Y").lstrip("0")
    except ValueError:
        found.pop("period_end", None)

    return found


def available_filings() -> list[str]:
    return sorted(p.name.replace(".verified.json", "")
                  for p in PROCESSED.glob("*.verified.json"))


def run_full_pipeline(filing_id: str, force: bool = False) -> None:
    """Run every stage, reporting progress. Slow - a minute or more."""
    from pipeline import investigate
    from flagging.generator import explain
    from flagging.faithfulness_check import verify

    progress = st.progress(0.0, "Reading the filing")
    anomalies = investigate(filing_id, force)
    progress.progress(0.5, f"{len(anomalies)} concerns found. Looking for explanations")

    explained = [explain(a, a.get("evidence", [])) for a in anomalies]
    progress.progress(0.8, "Checking explanations against their sources")

    verified = [verify(e) for e in explained]
    (PROCESSED / f"{filing_id}.verified.json").write_text(
        json.dumps(verified, indent=2), encoding="utf-8")

    progress.progress(1.0, "Done")
    load_json.clear()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def esc(text: str) -> str:
    return html.escape(str(text))


def fmt(value: float) -> str:
    """Accounting format: thousands separated, negatives in parentheses."""
    if value is None:
        return "—"
    whole = abs(value) == int(abs(value))
    body = f"{abs(value):,.0f}" if whole else f"{abs(value):,.2f}"
    return f"({body})" if value < 0 else body


def verdict_of(item: dict) -> tuple[str, str]:
    """(css modifier, label shown to the reader).

    The wording is deliberately narrow. What the entailment check establishes
    is that the explanation follows from the passage it cited - not that the
    filing accounts for the change adequately. "Verified against source"
    invited the second reading, which the system cannot support.
    """
    verdict = item.get("verdict")
    if verdict in {"supported", "verbatim"}:
        return "verified", "explanation matches source"
    if verdict == "not_supported":
        return "failed", "explanation unsupported"
    return "unexplained", "no explanation in filing"


def formula_line(item: dict, rows: list[dict], latest: str) -> str:
    """Name the arithmetic behind a ratio, but only if it reproduces.

    The numerator is assumed to be the first metric. Rather than trust that,
    the division is carried out here and compared against the value the
    detector recorded. If it does not reproduce, nothing is shown - a wrong
    formula printed under the figures would be worse than no formula.
    """
    if item.get("rule") != "ratio_shift" or len(rows) != 2:
        return ""

    target = (item.get("detail") or {}).get("to")
    num = rows[0]["values"].get(latest)
    den = rows[1]["values"].get(latest)
    if target is None or not num or not den:
        return ""

    ratio = num / den
    if abs(ratio - target) < 0.01:
        shape = "{a} ÷ {b}"
    elif abs(ratio * 100 - target) < 0.01:
        shape = "{a} ÷ {b} × 100"
    else:
        return ""

    expr = shape.format(a=esc(rows[0]["label"]).lower(),
                        b=esc(rows[1]["label"]).lower())
    return f"<p class='formula'>Computed as {expr}.</p>"


def ledger_table(rows: list[dict], used: set[str] | None = None) -> str:
    """The series behind a concern, set as a statement.

    ``used`` names the periods the finding was computed from. A filing
    often reports three years while a concern compares two, so without
    this the reader cannot tell which columns produced the number.
    """
    used = used or set()
    if not rows:
        return ""

    periods: list[str] = []
    for m in rows:
        for p in m["values"]:
            if p not in periods:
                periods.append(p)
    periods.sort(reverse=True)

    def mark(period: str) -> str:
        return " is-used" if period in used else ""

    head = "".join(
        f"<div class='ledger__v{mark(p)}'>{esc(p)}</div>" for p in periods)
    body = [f"<div class='ledger__r ledger__r--head'>"
            f"<div class='ledger__l'>Figures behind this</div>{head}</div>"]

    for m in rows:
        cells = "".join(
            f"<div class='ledger__v{mark(p)}'>{fmt(m['values'].get(p))}</div>"
            for p in periods)
        unit = f" <span class='unit'>{esc(m['scale'])}</span>" if m.get("scale") else ""
        body.append(f"<div class='ledger__r'>"
                    f"<div class='ledger__l'>{esc(m['label'])}{unit}</div>"
                    f"{cells}</div>")

    pages = sorted({m["page"] for m in rows if m.get("page")})
    if pages:
        body.append(
            f"<div class='ledger__r'><div class='ledger__src' "
            f"style='padding-left:0'>Reported on "
            f"{'pages' if len(pages) > 1 else 'page'} "
            f"{esc(', '.join(str(p) for p in pages))} of the filing.</div>"
            + "".join(f"<div class='ledger__src{mark(p)}'></div>"
                      for p in periods)
            + "</div>")

    return f"<div class='ledger'>{''.join(body)}</div>"


def citation_block(item: dict) -> str:
    citation = item.get("citation")
    if not citation:
        return ""

    section = str(citation.get("section", "")).strip()
    page = citation.get("page")
    where = f"Page {esc(page)}" if page else "Source"
    if section:
        where += f", {esc(section)}"

    score = item.get("faithfulness_score")
    label = item.get("nli_label")
    if score is not None:
        check = (f"An entailment model scored this passage {score:.2f} "
                 f"against the explanation ({esc(label)}).")
    elif item.get("verdict") == "verbatim":
        check = "The explanation quotes this passage directly."
    else:
        check = ""

    flaws = "".join(
        f"<p class='flaw'>{esc(f)}</p>" for f in item.get("failures") or [])

    return (
        f"<details class='cite'><summary>Read the passage this came from</summary>"
        f"<blockquote>{esc(citation.get('text_span', ''))}</blockquote>"
        f"<p class='prov'>{where}. {check}</p>{flaws}</details>"
    )


def render_entry(item: dict, metrics: dict[str, dict]) -> str:
    modifier, verdict_label = verdict_of(item)

    body = item.get("explanation", "")
    finding = item.get("finding", "")
    if body.startswith(finding):
        body = body[len(finding):].lstrip(". ")
    absent = "" if item.get("supported") else " is-absent"

    severity = float(item.get("severity") or 0)
    rule = str(item.get("rule", "")).replace("_", " ")

    rows = [metrics[i] for i in item.get("metric_ids") or [] if i in metrics]
    periods = sorted({p for m in rows for p in m["values"]}, reverse=True)
    latest = periods[0] if periods else ""
    used = {str(p) for p in (item.get("detail") or {}).get("periods") or []}

    return (
        f"<article class='entry entry--{modifier}'>"
        f"<div class='entry__head'>"
        f"<span class='entry__rule'>{esc(rule)}</span>"
        f"<span class='entry__sev'>severity {severity:.2f}</span>"
        f"<span class='entry__verdict'>{esc(verdict_label)}</span>"
        f"</div>"
        f"<p class='entry__finding'>{esc(finding)}</p>"
        f"<p class='entry__account{absent}'>{esc(body) or 'No explanation available.'}</p>"
        f"{ledger_table(rows, used)}"
        f"{formula_line(item, rows, latest)}"
        f"{citation_block(item)}"
        f"</article>"
    )


def remove_filing(filing_id: str) -> list[str]:
    """Delete everything stored for one filing; return what was removed.

    Matches on the filing id as a whole name or as the stem before a dot,
    so removing "ko-20241231" cannot touch "ko-20241231-b" or another
    company. Looks under data/ because the pipeline writes the source,
    chunks, metrics and verified results to separate places.
    """
    import shutil
    removed: list[str] = []
    root = Path("data")
    if not root.exists() or not filing_id:
        return removed
    for p in sorted(root.rglob("*")):
        stem = p.name.split(".")[0] if p.is_file() else p.name
        if stem != filing_id:
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
            removed.append(str(p))
        except OSError:
            pass
    load_json.clear()
    filing_identity.clear()
    return removed


def report_html(title: str, masthead: str, entries: list[str],
                view_label: str) -> str:
    """One self-contained page the reader can print or save as PDF.

    Cited passages are opened because a collapsed <details> prints as
    nothing. The sidebar and controls are not part of it.
    """
    body = "".join(e.replace("<details class='cite'",
                             "<details class='cite' open")
                   for e in entries)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>FinGuard - {esc(title)}</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"{STYLES}"
        "<style>"
        "body { margin: 0; background: #fff; color: var(--ink);"
        " font-family: 'IBM Plex Sans', system-ui, sans-serif; }"
        ".page { max-width: 60rem; margin: 0 auto; padding: 2.4rem 1.6rem; }"
        ".printbar { display: flex; gap: 1rem; align-items: center;"
        " margin-bottom: 1.4rem; font-size: .85rem; color: var(--ink-soft); }"
        ".printbar button { font: inherit; padding: .4rem .9rem;"
        " border: 1px solid var(--rule-firm); background: #fff;"
        " border-radius: 4px; cursor: pointer; }"
        "@media print {"
        " .printbar { display: none; }"
        " .page { padding: 0; max-width: none; }"
        " .entry { break-inside: avoid; }"
        " @page { margin: 16mm 14mm; }"
        "}"
        "</style></head><body><div class='page'>"
        "<div class='printbar'>"
        "<button onclick='window.print()'>Print or save as PDF</button>"
        f"<span>{esc(view_label)}</span></div>"
        f"{masthead}{body}"
        "<p class='footnote'>FinGuard flags areas for review. It is not an "
        "audit, a fraud detector, or investment advice; every concern needs "
        "a human judgement.</p>"
        "</div></body></html>"
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.markdown(STYLES, unsafe_allow_html=True)

st.sidebar.markdown("### FinGuard")
st.sidebar.caption(
    "Finds financial changes worth a second look in a 10-K filing, then "
    "retrieves what management said about each one.")

filings = available_filings()

# A new filing is selected by leaving a note here and rerunning; Streamlit
# does not allow changing the dropdown's value after it has been drawn.
if "pending_choice" in st.session_state:
    st.session_state["filing_choice"] = st.session_state.pop("pending_choice")

if "filing_choice" in st.session_state and \
        st.session_state["filing_choice"] not in filings:
    del st.session_state["filing_choice"]

choice = st.sidebar.selectbox(
    "Analysed filings", filings or ["(none yet)"],
    key="filing_choice", disabled=not filings)

st.sidebar.divider()
uploader_key = f"uploader_{st.session_state.get('uploader_n', 0)}"
uploaded = st.sidebar.file_uploader(
    "Analyse another filing", type=["htm", "html"], key=uploader_key,
    help="SEC EDGAR serves 10-K filings as HTML. Download the primary "
         "document from the filing index page.")

if uploaded is not None:
    name = Path(uploaded.name).stem
    if st.sidebar.button(f"Analyse {name}", type="primary"):
        FILINGS.mkdir(parents=True, exist_ok=True)
        (FILINGS / f"{name}.htm").write_bytes(uploaded.getvalue())
        warm_models()
        run_full_pipeline(name, force=True)
        st.session_state["pending_choice"] = name
        # A fresh key empties the uploader, so the file is not re-saved
        # on every rerun and the box is clear for the next filing.
        st.session_state["uploader_n"] = st.session_state.get("uploader_n", 0) + 1
        st.rerun()

if filings and st.sidebar.button("Run the analysis again"):
    warm_models()
    run_full_pipeline(choice, force=True)
    st.rerun()

if filings:
    with st.sidebar.expander("Remove this filing"):
        st.caption(f"Deletes {choice} and its results from this computer. "
                   "The analysis would have to be run again.")
        sure = st.checkbox("Yes, remove it", key=f"sure_{choice}")
        if st.button("Remove", disabled=not sure, key=f"rm_{choice}"):
            remove_filing(choice)
            st.session_state.pop("filing_choice", None)
            st.rerun()

if not filings:
    st.markdown(
        "<div class='masthead'><h1>FinGuard</h1></div>"
        "<p class='basis'>Upload a 10-K filing in the sidebar to begin. "
        "FinGuard reads the filing's tables, finds changes worth a second "
        "look, and retrieves what management wrote about each one.</p>",
        unsafe_allow_html=True)
    st.stop()

items = load_verified(choice)
metrics = load_metrics(choice)
n_chunks, n_pages = filing_scale(choice)

explained = [i for i in items if i.get("supported")]
unexplained = [i for i in items if not i.get("supported")]

scope = f"{len(items)} concerns raised"

ident = filing_identity(choice)
title = ident.get("registrant") or choice

facts: list[tuple[str, str]] = []
if ident.get("symbol"):
    facts.append(("Ticker", ident["symbol"]))
if ident.get("doc_type"):
    facts.append(("Filing", f"Form {ident['doc_type']}"))
if ident.get("fiscal_year"):
    facts.append(("Fiscal year", ident["fiscal_year"]))
if ident.get("period_end"):
    facts.append(("Period ended", ident["period_end"]))
if n_pages:
    facts.append(("Pages", str(n_pages)))
if n_chunks:
    facts.append(("Passages indexed", f"{n_chunks:,}"))

strip = "".join(
    f"<div class='fact'><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>"
    for k, v in facts)
strip = f"<dl class='facts'>{strip}</dl>" if strip else ""

masthead_html = (
    f"<div class='masthead'><h1>{esc(title)}</h1>{strip}"
    f"<p class='scope'>{esc(scope)}. {len(explained)} explained by the "
    f"filing, {len(unexplained)} left unexplained.</p>"
    f"<p class='basis'>Concerns come from arithmetic over the filing's own "
    f"tables. No language model decides what is flagged; it only reads the "
    f"passage retrieved for a concern already raised. Shaded figures are "
    f"the periods a concern was computed from.</p></div>")
st.markdown(masthead_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Charts, as a separate view rather than mixed into the entries. The reading
# page is built around the figures themselves, and a chart beside them would
# compete with the numbers the project asks a reader to check by hand.
# ---------------------------------------------------------------------------
st.sidebar.divider()
view = st.sidebar.radio("View", ["Concerns", "Charts"], key="view_mode")

if view == "Charts":
    import altair as alt
    import pandas as pd

    TONE = {"explained": "#0F6B4F", "unexplained": "#8A5115",
            "unsupported": "#9B1C1C"}

    def clip(text, n):
        """Trim to n characters on a word boundary, with an ellipsis."""
        text = (text or "").strip()
        if len(text) <= n:
            return text
        cut = text[:n].rsplit(" ", 1)[0]
        return (cut or text[:n]) + "\u2026"

    def metric_label(item, index, fallback):
        """The metric's own name, looked up in the extracted metrics.

        The saved results carry metric_ids but not the labels, so the name
        has to be fetched rather than read off the finding.
        """
        ids = item.get("metric_ids") or []
        if index < len(ids):
            row = metrics.get(ids[index]) or {}
            if row.get("label"):
                return row["label"]
        return fallback

    def verdict_word(item):
        v = item.get("verdict")
        if v in {"supported", "verbatim"}:
            return "explained"
        return "unsupported" if v == "not_supported" else "unexplained"

    st.markdown("##### Concerns by severity")
    st.caption("Severity orders findings inside this filing. Colour is the "
               "verdict. Hover for the rule.")
    df = pd.DataFrame([{
        "concern": clip(i.get("finding"), 64),
        "severity": round(float(i.get("severity") or 0), 2),
        "verdict": verdict_word(i),
        "rule": str(i.get("rule", "")).replace("_", " "),
    } for i in items])
    present = [v for v in TONE if v in set(df["verdict"])]
    st.altair_chart(
        alt.Chart(df).mark_bar(size=15, cornerRadius=2).encode(
            x=alt.X("severity:Q", title="severity",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("concern:N", title=None, sort="-x",
                    axis=alt.Axis(labelLimit=360, labelFontSize=12,
                                  labelPadding=8)),
            color=alt.Color("verdict:N", title=None,
                            scale=alt.Scale(domain=present,
                                            range=[TONE[v] for v in present]),
                            legend=alt.Legend(orient="bottom")),
            tooltip=["concern", "rule", "severity", "verdict"],
        ).properties(height=max(150, 40 * len(df))),
        use_container_width=True)

    st.markdown("##### Which rules fired")
    st.caption("Five arithmetic rules run on every filing. The mix differs by "
               "company.")
    counts: dict[str, int] = {}
    for i in items:
        name = str(i.get("rule", "")).replace("_", " ")
        counts[name] = counts.get(name, 0) + 1
    rules_df = pd.DataFrame({"rule": list(counts),
                             "concerns": list(counts.values())})
    st.altair_chart(
        alt.Chart(rules_df).mark_bar(size=15, cornerRadius=2,
                                     color="#5A6B7B").encode(
            x=alt.X("concerns:Q", title="concerns",
                    axis=alt.Axis(tickMinStep=1, format="d")),
            y=alt.Y("rule:N", title=None, sort="-x",
                    axis=alt.Axis(labelFontSize=12, labelPadding=8)),
            tooltip=["rule", "concerns"],
        ).properties(height=max(110, 40 * len(rules_df))),
        use_container_width=True)

    moves = []
    for i in items:
        detail = i.get("detail") or {}
        rule = i.get("rule")

        def num(key, d=detail):
            v = d.get(key)
            return float(v) if isinstance(v, (int, float)) else None

        pairs = []
        if rule == "divergence":
            for key, value in detail.items():
                if key in {"periods", "gap_pp"} or not isinstance(value, (int, float)):
                    continue
                pairs.append((key.replace("_", " "), float(value)))
        elif rule in {"segment_vs_total", "sustained_trend"}:
            a_key, b_key = (("segment", "total") if rule == "segment_vs_total"
                            else ("segment_span", "total_span"))
            a, b = num(a_key), num(b_key)
            if a is not None and b is not None:
                pairs = [(clip(metric_label(i, 0, "this segment"), 34), a),
                         ("company total", b)]
        elif rule == "magnitude":
            a, b = num("change"), num("revenue_change")
            if a is not None and b is not None:
                pairs = [(clip(metric_label(i, 0, "this metric"), 34), a),
                         ("revenue", b)]
        if len(pairs) != 2:
            continue
        for name, value in pairs:
            moves.append({"finding": clip(i.get("finding"), 78),
                          "metric": name, "change": round(value, 1)})

    st.markdown("##### What moved against what")
    st.caption("Each pair of changes either side of zero. The gap is what the "
               "rule fired on.")
    if moves:
        moves_df = pd.DataFrame(moves)
        st.altair_chart(
            alt.Chart(moves_df).mark_bar(size=13, cornerRadius=2).encode(
                x=alt.X("change:Q", title="change over the period (%)"),
                y=alt.Y("metric:N", title=None, sort=None,
                        axis=alt.Axis(labelFontSize=11, labelLimit=240)),
                color=alt.condition(alt.datum.change >= 0,
                                    alt.value("#2F5D8A"), alt.value("#8A5115")),
                tooltip=["finding", "metric", "change"],
            ).properties(height=46, width=430).facet(
                row=alt.Row("finding:N", title=None,
                            header=alt.Header(labelFontSize=11,
                                              labelOrient="top",
                                              labelAnchor="start",
                                              labelLimit=600, labelPadding=0,
                                              labelFontWeight="bold")),
            ).resolve_scale(y="independent"))
    else:
        st.caption("No comparative findings in this filing.")
    st.stop()

groups = {
    f"Everything ({len(items)})": items,
    f"Explained by the filing ({len(explained)})": explained,
    f"Left unexplained ({len(unexplained)})": unexplained,
}

st.sidebar.divider()
show = st.sidebar.radio(
    "Show", list(groups),
    help="Unexplained means the arithmetic found a real change and the "
         "filing contains no passage accounting for it.")

selected = groups[show]

if selected:
    st.sidebar.divider()
    st.sidebar.download_button(
        "Download printable report",
        data=report_html(title, masthead_html,
                         [render_entry(i, metrics) for i in selected], show),
        file_name=f"finguard-{choice}.html", mime="text/html",
        help="Opens in your browser. Press Print, then choose Save as PDF.")

if not selected:
    st.info("Nothing in this category for this filing.")

for item in selected:
    st.markdown(render_entry(item, metrics), unsafe_allow_html=True)

st.markdown(
    "<p class='footnote'>FinGuard flags areas for review. It is not an "
    "audit, a fraud detector, or investment advice; every concern needs a "
    "human judgement.</p>",
    unsafe_allow_html=True)