#!/usr/bin/env python3
"""Download the infographic JPEGs from the image batch results (BigQuery) into a folder.

1. One query extracts just (key, image bytes as base64) from image_out into a slim
   table image_export — prompts, metadata and thought parts are left behind.
2. That table is read in parallel pages (free) and each image is written as
   <outdir>/<ENSG>_<SYMBOL>.jpg (just <ENSG>.jpg for genes without a symbol).

Resumable: genes that already have a file in <outdir> are left out of the extraction,
so a rerun only downloads what is missing. Genes whose response contained no image are
listed in data/batch/image_failures.txt (batch_image_prepare.py --only-keys can retry them).

Usage
  python scripts/batch_image_export.py                    # -> output/images/
  python scripts/batch_image_export.py --outdir ~/Desktop/gene_images
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bq  # noqa: E402

# first non-thought image part of each response
EXPORT_SQL = """
SELECT key,
  (SELECT JSON_VALUE(p, '$.inlineData.data')
     FROM UNNEST(JSON_QUERY_ARRAY(response, '$.candidates[0].content.parts')) AS p WITH OFFSET i
     WHERE JSON_VALUE(p, '$.inlineData.mimeType') = 'image/jpeg'
       AND COALESCE(LAX_BOOL(p.thought), FALSE) = FALSE
     ORDER BY i LIMIT 1) AS jpeg_b64
FROM `{project}.{dataset}.{src}`
WHERE key NOT IN UNNEST(ARRAY<STRING>{done})
"""
PAGE = 20  # rows per request (~0.4 MB each as base64; keeps responses well under 10 MB)


def gene_symbols(summary_dir: Path) -> dict[str, str]:
    out = {}
    for p in summary_dir.glob("ENSG*_summary.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[d["ensembl_id"]] = d.get("gene") or d["ensembl_id"]
    return out


def filename(ensembl_id: str, symbol: str | None) -> str:
    if not symbol or symbol == ensembl_id:
        return f"{ensembl_id}.jpg"
    return f"{ensembl_id}_{re.sub(r'[^A-Za-z0-9.-]+', '-', symbol)}.jpg"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="image_out", help="batch result table")
    ap.add_argument("--outdir", default="output/images")
    ap.add_argument("--summary-dir", default="output", help="for gene symbols in file names")
    ap.add_argument("--failures", default="data/batch/image_failures.txt")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    symbols = gene_symbols(Path(args.summary_dir))

    done = sorted({p.name[:15] for p in outdir.glob("ENSG*.jpg")})
    print(f"{len(done)} images already in {outdir}/; extracting the rest from {args.table} ...", flush=True)
    n = bq.query_to_table(
        EXPORT_SQL.format(project=bq.PROJECT, dataset=bq.DATASET, src=args.table, done=json.dumps(done)),
        "image_export")
    print(f"{n} rows to download ...", flush=True)

    def fetch(start: int) -> tuple[int, int, list[str]]:
        written = skipped = 0
        missing = []
        for row in bq.list_rows("image_export", start, PAGE):
            key = row["key"]
            if not row["jpeg_b64"]:
                missing.append(key)
                continue
            path = outdir / filename(key, symbols.get(key))
            if path.exists():
                skipped += 1
                continue
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(base64.b64decode(row["jpeg_b64"]))
            tmp.replace(path)
            written += 1
        return written, skipped, missing

    written = skipped = 0
    missing: list[str] = []
    starts = list(range(0, n, PAGE))
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (w, s, m) in enumerate(pool.map(fetch, starts), 1):
            written += w
            skipped += s
            missing += m
            if i % 100 == 0 or i == len(starts):
                print(f"  [{min(i * PAGE, n)}/{n}] written {written}, already there {skipped}", flush=True)

    fail_path = Path(args.failures)
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path.write_text("".join(f"{k}\n" for k in sorted(missing)), encoding="utf-8")
    size = sum(p.stat().st_size for p in outdir.glob("*.jpg"))
    print(f"\ndone: {written} written, {skipped} already present, "
          f"{len(list(outdir.glob('*.jpg')))} JPEGs in {outdir}/ ({size / 1e9:.2f} GB)")
    if missing:
        print(f"{len(missing)} genes had no image in the response -> {fail_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
