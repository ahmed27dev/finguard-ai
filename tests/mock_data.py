"""
mock_data.py - fixtures for the flagging tests.

Built from figures that actually appear in the filings the system was
developed against, not invented numbers. A test that passes on made-up data
and fails on a real filing is worse than no test, and several of the bugs
these fixtures protect against were only visible in real table shapes.
"""

from __future__ import annotations


def linear_row(label: str, values: dict, *, page: int = 31,
               section: str = "Item 8 - Financial Statements",
               layout: str = "wide", scale: str | None = "millions",
               row_id: str | None = None) -> dict:
    """One linearized table row, as linearizer.py emits it."""
    return {
        "row_id": row_id or f"r_{abs(hash(label)) % 10000:04d}",
        "table_id": "tb_test",
        "page": page,
        "section": section,
        "label": label,
        "layout": layout,
        "period": None,
        "scale": scale,
        "scale_source": "preceding" if scale else None,
        "values": values,
        "linear": f"{label}: " + " | ".join(f"{k}: {v}" for k, v in values.items()),
    }


def metric(canonical: str | None, label: str, values: dict, *,
           page: int = 31, layout: str = "wide",
           metric_id: str = "m_0000") -> dict:
    """One extracted metric, as metric_extractor.py emits it."""
    return {
        "metric_id": metric_id,
        "canonical": canonical,
        "label": label,
        "values": values,
        "absolute": {k: v * 1_000_000 for k, v in values.items()},
        "scale": "millions",
        "scale_source": "preceding",
        "layout": layout,
        "page": page,
        "section": "Item 8 - Financial Statements",
        "source_row_id": "r_0001",
        "also_on_pages": [],
        "is_primary": bool(canonical),
        "role": "company" if canonical else "unnamed",
    }


def evidence(text: str, *, page: int = 27, supported: bool = True,
             score: float = 8.0, source_type: str = "text") -> dict:
    """One retrieved passage, as retrieve.py emits it."""
    return {
        "text": text,
        "source_type": source_type,
        "chunk_id": "c_00001",
        "page": page,
        "section": "Item 7 - Management's Discussion and Analysis",
        "score": score,
        "supported": supported,
    }


# --- Apple FY2025, segment operating income across three per-year tables ---
# Greater China fell in every period while the company total rose. This is
# only visible because the linearizer merged three separate tables.
APPLE_SEGMENT_OP_INCOME = {
    "Greater China": {"2025": 26917.0, "2024": 27082.0, "2023": 30328.0},
    "Americas": {"2025": 72480.0, "2024": 67656.0, "2023": 60508.0},
    "Europe": {"2025": 47739.0, "2024": 41790.0, "2023": 36098.0},
    "Total": {"2025": 133050.0, "2024": 123216.0, "2023": 114301.0},
}

# --- Microsoft FY2026, every headline metric grew with the company ---
# A bare year-over-year threshold flagged all six and said nothing.
MSFT_UNIFORM_GROWTH = {
    "revenue": {"2026": 331839.0, "2025": 281724.0},
    "operating_income": {"2026": 155237.0, "2025": 128528.0},
    "gross_profit": {"2026": 225465.0, "2025": 193893.0},
    "net_income": {"2026": 133749.0, "2025": 101832.0},
    "equity": {"2026": 442387.0, "2025": 343479.0},
    "total_assets": {"2026": 758376.0, "2025": 619003.0},
}

# --- Nvidia FY2026, hypergrowth ---
# Gross profit sits 8.7 points from revenue - over an absolute cutoff, but
# proportionally 0.87, which is plainly just tracking revenue.
NVDA_HYPERGROWTH = {
    "revenue": {"2026": 215938.0, "2025": 130497.0},
    "gross_profit": {"2026": 153463.0, "2025": 97858.0},
    "total_assets": {"2026": 206803.0, "2025": 111601.0},
    "equity": {"2026": 157293.0, "2025": 79327.0},
}

APPLE_TAX_EVIDENCE = (
    "The Company's effective tax rate for 2025 was lower compared to 2024 due "
    "to a $10.7 billion year-over-year decrease in the provision for income "
    "taxes related to the State Aid Decision."
)

APPLE_CHINA_EVIDENCE = (
    "Greater China net sales decreased during 2025 compared to 2024 primarily "
    "due to lower net sales of iPhone, partially offset by higher net sales "
    "of Mac."
)