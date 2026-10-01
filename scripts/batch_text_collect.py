#!/usr/bin/env python3
"""Turn text batch results (BigQuery) into <ID>_summary.md / <ID>_summary.json files.

Runs every section's raw model text through summarize_gene.postprocess + write_summary,
so the files are identical in format to a local summarize_gene.py run.

A gene is written only if ALL its expected sections came back successfully; otherwise
its failed keys go to data/batch/text_failures.txt (one key per line), which
batch_text_prepare.py --only-keys can turn into a retry batch. Several result tables
can be given; later ones override earlier ones (e.g. text_out then text_retry_out).

Usage
  python scripts/batch_text_collect.py                                   # text_out -> output/
  python scripts/batch_text_collect.py --table text_out --table text_retry_out
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bq  # noqa: E402
from batch_text_prepare import section_slug  # noqa: E402
from summarize_gene import (  # noqa: E402
    DEFAULT_MODEL, DEFAULT_THINKING, SECTIONS, _first_info, has_real_data, postprocess, write_summary,
)

# text = all non-thought parts of the first candidate, concatenated (same as resp.text)
RESULTS_SQL = """
SELECT
  key,
  status,
  (SELECT STRING_AGG(JSON_VALUE(p, '$.text'), '')
     FROM UNNEST(JSON_QUERY_ARRAY(response, '$.candidates[0].content.parts')) AS p
     WHERE COALESCE(LAX_BOOL(p.thought), FALSE) = FALSE) AS text,
  JSON_VALUE(response, '$.candidates[0].finishReason') AS finish,
  LAX_INT64(response.usageMetadata.promptTokenCount) AS prompt,
  LAX_INT64(response.usageMetadata.thoughtsTokenCount) AS thoughts,
  LAX_INT64(response.usageMetadata.candidatesTokenCount) AS output
FROM `{table}`
"""


def fetch_results(tables: list[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for t in tables:
        n = 0
        for r in bq.query(RESULTS_SQL.format(table=f"{bq.PROJECT}.{bq.DATASET}.{t}")):
            rows[r["key"]] = r
            n += 1
        print(f"read {n} rows from {t}")
    return rows


def failure_reason(r: dict | None) -> str | None:
    if r is None:
        return "missing"
    if r["status"]:
        return f"status: {r['status'][:120]}"
    if not r["text"]:
        return f"empty response (finishReason={r['finish']})"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", action="append", help="result table(s); default text_out")
    ap.add_argument("--clean-dir", default="output")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--failures", default="data/batch/text_failures.txt")
    args = ap.parse_args(argv)

    results = fetch_results(args.table or ["text_out"])
    outdir = Path(args.outdir)
    failures: list[tuple[str, str]] = []
    written = 0
    tokens = collections.Counter()
    notes_total = 0

    for clean_path in sorted(Path(args.clean_dir).glob("ENSG*_hpa_clean.json")):
        clean = json.loads(clean_path.read_text(encoding="utf-8"))
        ensembl_id = clean_path.name.split("_")[0]
        info = _first_info(clean)
        gene = (info.get("Gene name") or ensembl_id).split(" ")[0]
        protein_name = info.get("Protein") or ""

        expected = [t for t, builder in SECTIONS if has_real_data(builder(clean)[0])]
        keys = {t: f"{ensembl_id}__{section_slug(t)}" for t in expected}
        bad = [(keys[t], why) for t in expected if (why := failure_reason(results.get(keys[t])))]
        if bad:
            failures.extend(bad)
            continue

        sections: list[tuple[str, list[str]]] = []
        notes: dict[str, list[str]] = {}
        usage: dict[str, dict] = {}
        for t in expected:
            r = results[keys[t]]
            usage[t] = {k: int(r[k] or 0) for k in ("prompt", "thoughts", "output")}
            tokens.update(usage[t])
            bullets, n = postprocess(r["text"], section=t, gene=gene, protein_name=protein_name)
            if bullets == ["No data available"]:
                continue
            sections.append((t, bullets))
            if n:
                notes[t] = n
                notes_total += len(n)
        write_summary(outdir, ensembl_id, gene, protein_name, sections, notes,
                      {"model": DEFAULT_MODEL, "thinking": DEFAULT_THINKING, "usage": usage})
        written += 1

    fail_path = Path(args.failures)
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path.write_text("".join(f"{k}\n" for k, _ in failures), encoding="utf-8")
    print(f"\nwrote {written} summaries to {outdir}/  ({notes_total} formatting fixes/warnings)")
    print(f"tokens: {dict(tokens)}")
    if failures:
        print(f"{len(failures)} failed sections in "
              f"{len({k.split('__')[0] for k, _ in failures})} genes -> {fail_path}")
        for why, n in collections.Counter(w.split(':')[0] for _, w in failures).most_common():
            print(f"  {n:6}  {why}")
        print("retry: python scripts/batch_text_prepare.py --only-keys "
              f"{fail_path} --out data/batch/text_retry.jsonl --load --table text_retry_in")
    return 0


if __name__ == "__main__":
    sys.exit(main())
