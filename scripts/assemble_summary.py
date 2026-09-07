#!/usr/bin/env python3
"""Assemble a gene summary .md from raw section outputs (marker-delimited text).

Feeds each section's raw bullet text through summarize_gene.postprocess (so the
same formatting contract / validator applies no matter which model produced it),
then renders the card.

Input file format (one per gene):
    ===IDENTITY & FUNCTION===
    - ...
    ===EXPRESSION===
    - ...
    ===PROTEIN LOCALIZATION===
    - ...
    ===DISEASE & CLINICAL RELEVANCE===
    - ...

Usage:
    python scripts/assemble_summary.py raw/ENSG00000146648.txt \
        --clean output/ENSG00000146648_hpa_clean.json --tag sonnet --outdir output
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_gene import _first_info, postprocess, render  # noqa: E402

SECTION_ORDER = [
    "IDENTITY & FUNCTION",
    "EXPRESSION",
    "PROTEIN LOCALIZATION",
    "DISEASE & CLINICAL RELEVANCE",
    "PROTEIN & MOLECULAR CONTEXT",
]


def split_sections(text: str) -> dict[str, str]:
    parts = re.split(r"^===\s*(.+?)\s*===\s*$", text, flags=re.M)
    out: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        out[parts[i].strip().upper()] = parts[i + 1].strip()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", help="marker-delimited section text file")
    ap.add_argument("--clean", required=True, help="the matching <ID>_hpa_clean.json")
    ap.add_argument("--tag", default="alt", help="output suffix: <ID>_summary_<tag>.md")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args(argv)

    clean = json.loads(Path(args.clean).read_text(encoding="utf-8"))
    info = _first_info(clean)
    ensembl_id = Path(args.clean).name.split("_")[0]
    gene = (info.get("Gene name") or ensembl_id).split(" ")[0]
    protein_name = info.get("Protein") or ""

    sections_raw = split_sections(Path(args.raw).read_text(encoding="utf-8"))
    rendered: list[tuple[str, list[str]]] = []
    all_notes: dict[str, list[str]] = {}
    for title in SECTION_ORDER:
        if title not in sections_raw:
            continue
        bullets, notes = postprocess(
            sections_raw[title], section=title, gene=gene, protein_name=protein_name
        )
        rendered.append((title, bullets))
        if notes:
            all_notes[title] = notes

    md = render(gene, ensembl_id, rendered)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{ensembl_id}_summary_{args.tag}.md").write_text(md, encoding="utf-8")
    (outdir / f"{ensembl_id}_summary_{args.tag}.json").write_text(
        json.dumps(
            {"ensembl_id": ensembl_id, "gene": gene, "tag": args.tag,
             "sections": [{"title": t, "bullets": b} for t, b in rendered],
             "format_notes": all_notes},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {outdir / f'{ensembl_id}_summary_{args.tag}.md'}")
    if all_notes:
        for t, ns in all_notes.items():
            print(f"[{t}]")
            for n in ns:
                print(f"  - {n}")
    else:
        print("formatting: clean")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
