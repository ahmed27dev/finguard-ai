"""
fetch_filing.py - download 10-K filings from SEC EDGAR.

Two ways to call it:

    # by ticker - looks up the latest 10-K automatically
    python -m retrieval.fetch_filing MSFT
    python -m retrieval.fetch_filing NVDA --year 2024

    # by direct URL, when you already have one
    python -m retrieval.fetch_filing msft-2025 --url https://www.sec.gov/Archives/...

SEC rejects requests without a descriptive User-Agent, so set this first:

    $env:FINGUARD_UA = "FinGuard AI Student Project you@example.com"

Two CIK conventions, and mixing them up is the usual mistake:
    data.sec.gov API   -> zero-padded to 10 digits   CIK0000789019.json
    Archives URL path  -> no leading zeros           /edgar/data/789019/
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

FILINGS_DIR = Path("data/filings")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


def _headers() -> dict:
    ua = os.environ.get("FINGUARD_UA")
    if not ua:
        raise SystemExit(
            "Set FINGUARD_UA first:\n"
            '  $env:FINGUARD_UA = "FinGuard AI Student Project you@example.com"\n'
            "SEC returns 403 without it."
        )
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def _get(url: str) -> bytes:
    with urlopen(Request(url, headers=_headers()), timeout=60) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    time.sleep(0.15)  # SEC asks for under 10 requests/second
    return raw


def lookup_cik(ticker: str) -> tuple[str, str]:
    """Resolve a ticker to (cik_no_zeros, company_name)."""
    data = json.loads(_get(TICKERS_URL))
    target = ticker.upper()
    for entry in data.values():
        if entry["ticker"].upper() == target:
            return str(entry["cik_str"]), entry["title"]
    raise SystemExit(f"ticker {target} not found in EDGAR")


def find_10k(cik: str, year: int | None = None) -> tuple[str, str, str]:
    """
    Find a 10-K for this company.

    filings.recent holds parallel arrays: index i across form, filingDate,
    accessionNumber and primaryDocument describes one filing. Returns
    (accession_without_dashes, primary_document, filing_date).
    """
    data = json.loads(_get(SUBMISSIONS_URL.format(cik10=cik.zfill(10))))
    recent = data["filings"]["recent"]

    for i, form in enumerate(recent["form"]):
        if form != "10-K":
            continue
        filed = recent["filingDate"][i]
        if year and not filed.startswith(str(year)):
            continue
        return (
            recent["accessionNumber"][i].replace("-", ""),
            recent["primaryDocument"][i],
            filed,
        )
    raise SystemExit(f"no 10-K found for CIK {cik}" + (f" in {year}" if year else ""))


def fetch_by_ticker(ticker: str, year: int | None = None) -> Path:
    cik, name = lookup_cik(ticker)
    accession, document, filed = find_10k(cik, year)
    url = ARCHIVE_URL.format(cik=cik, accession=accession, document=document)

    print(f"{name}  (CIK {cik})")
    print(f"10-K filed {filed}")
    print(f"{url}")

    return save(url, f"{ticker.lower()}-{filed[:4]}")


def save(url: str, name: str) -> Path:
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    dest = FILINGS_DIR / f"{name}.htm"
    raw = _get(url)
    dest.write_bytes(raw)
    print(f"saved {dest}  ({len(raw) / 1_000_000:.1f} MB)\n")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="ticker (MSFT), or output name when --url is given")
    ap.add_argument("--url", help="direct filing URL, skips EDGAR lookup")
    ap.add_argument("--year", type=int, help="filing year, e.g. 2024")
    args = ap.parse_args()

    if args.url:
        save(args.url, args.target)
    else:
        fetch_by_ticker(args.target, args.year)


if __name__ == "__main__":
    main()