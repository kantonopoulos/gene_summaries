#!/usr/bin/env python3
"""Build the image batch input (one infographic request per gene) and load it into BigQuery.

Does NOT submit a batch job. Reads the finished summaries (<ID>_summary.md, written by
batch_text_collect.py or summarize_gene.py) and builds each prompt exactly as
visualize_gene.py does (build_prompt + batch_request_image).

Each row: key = "<ENSG>", request = GenerateContentRequest JSON.

Usage
  python scripts/batch_image_prepare.py                 # build data/batch/image_requests.jsonl
  python scripts/batch_image_prepare.py --load          # build + load into BigQuery (image_in)
  python scripts/batch_image_prepare.py --only-keys data/batch/image_failures.txt \\
      --out data/batch/image_retry.jsonl --load --table image_retry_in   # retry batch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bq  # noqa: E402
from visualize_gene import batch_request_image, build_prompt  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-dir", default="output")
    ap.add_argument("--out", default="data/batch/image_requests.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="only the first N genes")
    ap.add_argument("--only-keys", metavar="FILE", help="only these gene ids (one per line)")
    ap.add_argument("--load", action="store_true", help="also load the JSONL into BigQuery")
    ap.add_argument("--table", default="image_in")
    args = ap.parse_args(argv)

    files = sorted(Path(args.summary_dir).glob("ENSG*_summary.md"))
    if args.only_keys:
        wanted = {k.strip() for k in Path(args.only_keys).read_text().splitlines() if k.strip()}
        files = [p for p in files if p.name.split("_")[0] in wanted]
    files = files[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in files:
            row = {
                "key": p.name.split("_")[0],
                "request": batch_request_image(build_prompt(p.read_text(encoding="utf-8"))),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out}: {len(files)} requests ({out.stat().st_size / 1e6:.1f} MB)")

    n_clean = len(list(Path(args.summary_dir).glob("ENSG*_hpa_clean.json")))
    if not args.only_keys and args.limit is None and len(files) < n_clean:
        print(f"note: only {len(files)} summaries for {n_clean} cleaned genes "
              f"(missing ones are not in this batch)")
    if args.load:
        bq.load_jsonl(out, args.table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
