#!/usr/bin/env python3
"""Build the text batch input (one request per gene x section) and load it into BigQuery.

Does NOT submit a batch job. Uses the exact same section builders, instructions and
generation settings as summarize_gene.py (via summarize_gene.batch_request); sections
with no real data are skipped, as in the live pipeline.

Each row: key = "<ENSG>__<section-slug>", request = GenerateContentRequest JSON.

Steps
  1. output/*_hpa_clean.json  ->  data/batch/text_requests.jsonl  (local copy)
  2. --load: upload that file to BigQuery table <dataset>.<table> (replaced if it exists)

Usage
  python scripts/batch_text_prepare.py                 # build JSONL only
  python scripts/batch_text_prepare.py --load          # build + load into BigQuery
  python scripts/batch_text_prepare.py --limit 20      # first 20 genes (smoke test)
  python scripts/batch_text_prepare.py --only-keys data/batch/text_failures.txt \
      --out data/batch/text_retry.jsonl --load --table text_retry_in   # retry batch
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bq  # noqa: E402
from summarize_gene import (  # noqa: E402
    DEFAULT_THINKING, SECTIONS, _first_info, batch_request, has_real_data,
)


def section_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def gene_rows(clean_path: Path, thinking: str):
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    ensembl_id = clean_path.name.split("_")[0]
    info = _first_info(clean)
    gene = (info.get("Gene name") or ensembl_id).split(" ")[0]
    anchor = f"{gene} ({info.get('Protein') or ''})"
    for title, builder in SECTIONS:
        data, instr = builder(clean)
        if not has_real_data(data):
            continue
        yield title, {
            "key": f"{ensembl_id}__{section_slug(title)}",
            "request": batch_request(instr, data, anchor=anchor, thinking=thinking),
        }


def build(clean_dir: Path, out: Path, thinking: str, limit: int | None,
          only_keys: set[str] | None = None) -> None:
    files = sorted(clean_dir.glob("ENSG*_hpa_clean.json"))
    if only_keys is not None:
        wanted = {k.split("__")[0] for k in only_keys}
        files = [p for p in files if p.name.split("_")[0] in wanted]
    files = files[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    per_section: collections.Counter = collections.Counter()
    per_gene: collections.Counter = collections.Counter()
    chars = 0
    with out.open("w", encoding="utf-8") as f:
        for p in files:
            n = 0
            for title, row in gene_rows(p, thinking):
                if only_keys is not None and row["key"] not in only_keys:
                    continue
                line = json.dumps(row, ensure_ascii=False)
                f.write(line + "\n")
                chars += len(line)
                per_section[title] += 1
                n += 1
            per_gene[n] += 1
    total = sum(per_section.values())
    print(f"wrote {out}: {total} requests from {len(files)} genes "
          f"({out.stat().st_size / 1e6:.0f} MB, ~{chars / 4 / 1e6:.0f}M prompt tokens est.)")
    for title, _ in SECTIONS:
        print(f"  {title:32} {per_section[title]}")
    print("  sections per gene:", dict(sorted(per_gene.items())))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dir", default="output")
    ap.add_argument("--out", default="data/batch/text_requests.jsonl")
    ap.add_argument("--thinking", default=DEFAULT_THINKING)
    ap.add_argument("--limit", type=int, default=None, help="only the first N genes")
    ap.add_argument("--only-keys", metavar="FILE",
                    help="only these request keys (one per line), e.g. data/batch/text_failures.txt")
    ap.add_argument("--load", action="store_true", help="also load the JSONL into BigQuery")
    ap.add_argument("--table", default="text_in")
    args = ap.parse_args(argv)

    out = Path(args.out)
    only = None
    if args.only_keys:
        only = {k.strip() for k in Path(args.only_keys).read_text().splitlines() if k.strip()}
    build(Path(args.clean_dir), out, args.thinking, args.limit, only)
    if args.load:
        bq.load_jsonl(out, args.table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
