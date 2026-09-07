#!/usr/bin/env python3
"""Clean a raw <ID>_hpa.json (produced by fetch_hpa.py) into <ID>_hpa_clean.json.

Keeps only `summary_page`, drops noise keys, and folds a few numbers from
`structured_json` back into the human-readable strings.

Rules
  1. Output is the `summary_page` sections only (plus extras merged inline).
  2. Drop any key containing the word "cluster".
  3. Drop "Immune cell specificity" (and "Immune cell expression cluster", via 2).
  4. Drop "Prognostic summary".
  5. Drop "Detected in blood by immunoassay" / "... by mass spectrometry" /
     "Proximity extension assay" / "SomaScan".
  6. Drop "Brain specificity" when its value is "Low human brain regional specificity".
  7. Fold per-entity expression values into "Tissue specificity" (nTPM, from
     "RNA tissue specific nTPM") and "Single cell type specificity" (nCPM, from
     "RNA single cell type specific nCPM"):
         "Tissue enhanced (Placenta)" -> "Tissue enhanced (Placenta: 61.8 nTPM)"
  8. "Subcellular location": cut everything from "In addition ..." onward and
     append " (Reliability: <label>)", where the IF reliability score is mapped
     Supported/Enhanced -> "Highly consistent", Approved -> "Moderately
     consistent", Uncertain -> "Uncertain".
  9. "Gene name": drop the trailing "(synonym, synonym)" list.

Usage:
    python scripts/clean_hpa.py ENSG00000146648
    python scripts/clean_hpa.py path/to/ENSG00000146648_hpa.json --outdir data
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

DROP_EXACT = {
    "Immune cell specificity",
    "Prognostic summary",
    "Detected in blood by immunoassay",
    "Detected in blood by mass spectrometry",
    "Proximity extension assay",
    "SomaScan",
}

# key in summary_page -> (structured_json field with the per-entity values, unit label)
ENTITY_VALUES = {
    "Tissue specificity": ("RNA tissue specific nTPM", "nTPM"),
    "Single cell type specificity": ("RNA single cell type specific nCPM", "nCPM"),
}


def annotate_entities(value: str, values: dict | None, unit: str) -> str:
    """`Category (A, B)` + {a: 1.0} -> `Category (A: 1.0 unit, B)`."""
    if not values:
        return value
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", value)
    if not m:
        return value
    prefix, inside = m.group(1).strip(), m.group(2)
    lut = {k.lower(): v for k, v in values.items()}
    annotated = []
    for entity in (p.strip() for p in inside.split(",")):
        num = lut.get(entity.lower())
        annotated.append(f"{entity}: {num} {unit}" if num is not None else entity)
    return f"{prefix} ({', '.join(annotated)})"


# HPA IF reliability score -> plain-language consistency label
RELIABILITY_LABEL = {
    "supported": "Highly consistent",
    "enhanced": "Highly consistent",
    "approved": "Moderately consistent",
    "uncertain": "Uncertain",
}


def clean_subcellular(value: str, reliability: str | None) -> str:
    trimmed = re.split(r"\s+In addition\b", value, maxsplit=1)[0].strip().rstrip(",")
    if reliability:
        label = RELIABILITY_LABEL.get(reliability.strip().lower(), reliability)
        trimmed = f"{trimmed} (Reliability: {label})"
    return trimmed


def clean(payload: dict) -> "OrderedDict":
    summary = payload.get("summary_page", {})
    sj = payload.get("structured_json", {}) or {}
    reliability_if = sj.get("Reliability (IF)")

    out: "OrderedDict[str, OrderedDict[str, str]]" = OrderedDict()
    for section, entries in summary.items():
        kept: "OrderedDict[str, str]" = OrderedDict()
        for key, value in entries.items():
            if "cluster" in key.lower():
                continue
            if key in DROP_EXACT:
                continue
            if key == "Brain specificity" and value == "Low human brain regional specificity":
                continue

            if key == "Gene name":
                value = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()  # drop synonyms
            elif key in ENTITY_VALUES:
                field, unit = ENTITY_VALUES[key]
                value = annotate_entities(value, sj.get(field), unit)
            elif key == "Subcellular location":
                value = clean_subcellular(value, reliability_if)

            kept[key] = value
        out[section] = kept
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help="ENSG id, or path to a <ID>_hpa.json file")
    ap.add_argument("--indir", default=".", help="where to look for <ID>_hpa.json when given an id")
    ap.add_argument("--outdir", default=None, help="output dir (default: next to the input)")
    args = ap.parse_args(argv)

    if args.target.endswith(".json"):
        inpath = Path(args.target)
    else:
        ensembl_id = args.target.strip().split("-")[0].upper()
        inpath = Path(args.indir) / f"{ensembl_id}_hpa.json"

    payload = json.loads(inpath.read_text(encoding="utf-8"))
    cleaned = clean(payload)

    outdir = Path(args.outdir) if args.outdir else inpath.parent
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / inpath.name.replace("_hpa.json", "_hpa_clean.json")
    outpath.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {outpath}  ({outpath.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
