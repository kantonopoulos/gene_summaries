#!/usr/bin/env python3
"""Fetch the Human Protein Atlas gene summary for an Ensembl gene ID.

Pulls two things:
  1. summary_page  - every key/value shown in the summary cards at the top of
                     https://www.proteinatlas.org/<ENSG_ID> (what is visually there)
  2. structured_json - the full machine-readable payload from the .json endpoint
                     (superset of numbers behind the page: specificity scores,
                     per-tissue expression dicts, cancer prognostics, etc.)

Usage:
    python scripts/fetch_hpa.py ENSG00000146648
    python scripts/fetch_hpa.py ENSG00000146648 --outdir data
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.proteinatlas.org"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/json",
}
ENSG_RE = re.compile(r"^ENSG\d{11}$")


def normalize_id(raw: str) -> str:
    """`ENSG00000146648-EGFR` or `ensg00000146648` -> `ENSG00000146648`."""
    clean = raw.strip().split("-")[0].upper()
    if not ENSG_RE.match(clean):
        raise ValueError(f"Not a valid Ensembl gene ID: {raw!r}")
    return clean


def _clean(text: str) -> str:
    text = re.sub(r"\(\s*all genes\s*\)", "", text)
    text = re.sub(r"\.\.\.\s*show\s+(more|less)", "", text, flags=re.I)
    text = re.sub(r"\[provided by RefSeq[^\]]*\]", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .…")


def _value(td) -> str:
    """Text of a value cell.

    HPA splits long paragraphs with <span class="protein_function_text"> and
    injects <a> PubMed citation markers; joining with a space would break words
    ("extracel lular"), so those cells are joined with no separator and the
    citation links dropped.
    """
    td = BeautifulSoup(str(td), "html.parser")
    is_paragraph = td.find("span", class_=re.compile(r"_text$")) is not None
    if is_paragraph:
        for a in td.find_all("a"):
            a.decompose()
        return _clean(td.get_text("", strip=True))
    return _clean(td.get_text(" ", strip=True))


def _label(th) -> str:
    """Text of a header cell with the (i) tooltip helper stripped out."""
    th = BeautifulSoup(str(th), "html.parser")
    for junk in th.select("sup.help_i, div[id^='help_'], .helpText"):
        junk.decompose()
    return _clean(th.get_text(" ", strip=True))


def fetch_summary_page(ensembl_id: str) -> "OrderedDict[str, OrderedDict[str, str]]":
    """Scrape the summary cards (`table.summary_info`) from the gene page."""
    url = f"{BASE}/{ensembl_id}"
    resp = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    sections: "OrderedDict[str, OrderedDict[str, str]]" = OrderedDict()
    for table in soup.select("table.summary_info"):
        rows = table.find_all("tr")
        if not rows:
            continue
        head = rows[0].find("th", class_="head")
        title = _clean(head.get_text(" ", strip=True)) if head else "GENERAL"
        entries: "OrderedDict[str, str]" = OrderedDict()
        for tr in rows[1:]:
            th = tr.find("th")
            td = tr.find("td")  # first data cell; image cell (rowspan) comes later
            if not th or not td:
                continue
            key = _label(th)
            val = _value(td)
            if key and val:
                entries[key] = val
        if entries:
            sections[title] = entries
    return sections


def fetch_structured_json(ensembl_id: str) -> dict:
    """The full HPA .json record (list with one gene object)."""
    resp = requests.get(f"{BASE}/{ensembl_id}.json", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) and data else data


def fetch_gene(raw_id: str) -> dict:
    ensembl_id = normalize_id(raw_id)
    structured = fetch_structured_json(ensembl_id)
    summary = fetch_summary_page(ensembl_id)
    return {
        "ensembl_id": ensembl_id,
        "gene_symbol": structured.get("Gene"),
        "page_url": f"{BASE}/{ensembl_id}",
        "fetched_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "summary_page": summary,
        "structured_json": structured,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ensembl_id", help="e.g. ENSG00000146648")
    ap.add_argument("--outdir", default=".", help="directory to write <ID>_hpa.json")
    ap.add_argument("--print", action="store_true", help="also print the summary_page to stdout")
    args = ap.parse_args(argv)

    payload = fetch_gene(args.ensembl_id)
    outpath = Path(args.outdir) / f"{payload['ensembl_id']}_hpa.json"
    outpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {outpath}  ({outpath.stat().st_size:,} bytes)")

    if args.print:
        for section, entries in payload["summary_page"].items():
            print(f"\n=== {section} ===")
            for k, v in entries.items():
                print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
