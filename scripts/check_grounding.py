#!/usr/bin/env python3
"""Verify a generated summary introduces NO named entity absent from its source data.

The model may paraphrase and draw logical links, but every disease, tissue, cell
type, organ, pathway, ligand, gene or number in the summary must trace back to the
clean JSON it was given.

What is checked (targeted, to keep noise low):
  * proper-noun phrases   - Capitalised multi-word sequences (disease names,
                            "Adrenal gland", "Beta cells", "Li-Fraumeni", ...)
  * biomedical single words - tokens ending in -oma/-itis/-emia/-aemia/-cyte/
                            -blast/-phil/-genic/-cyte plus a small organ list
  * bare integers          - must appear in the data (transcript counts etc.)
Each candidate must appear in the haystack (all clean-JSON string values,
lowercased) as a >=4-char prefix match, which absorbs plural / -is / -ic
paraphrase. Misses are printed with their bullet for manual review.

Usage:
  python scripts/check_grounding.py output/ENSG00000146648_summary_sonnet.json \
      --clean output/ENSG00000146648_hpa_clean.json
  python scripts/check_grounding.py output/*_summary_sonnet.json --clean-dir output
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROPER = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:[ /-][A-Z0-9][A-Za-z0-9]*)+)\b")
BIO_SUFFIX = re.compile(
    r"(oma|omas|itis|emia|aemia|cyte|cytes|blast|blasts|phil|phils|genic|troph|trophs"
    r"|penia|plasia|pathy|ния)$",
    re.I,
)
ORGANS = {
    "placenta", "pancreas", "pancreatic", "brain", "cerebellum", "cerebral", "cortex",
    "liver", "kidney", "spleen", "testis", "testes", "prostate", "adrenal", "thymus",
    "lymphoid", "marrow", "bone", "ovary", "ovarian", "breast", "stomach", "lung",
    "colon", "skin", "muscle", "heart", "blood", "islet", "islets", "neuropil", "pons",
    "endoplasmic", "reticulum", "golgi", "nucleoplasm", "keratinocyte", "keratinocytes",
    "astrocyte", "astrocytes", "hepatocyte", "hepatocytes", "trophoblast",
}
# structural / evaluative words that are never "named entities"
STOP = set(
    """
    has have had having contains contain consists comprises
    the a an and or of in to for with is are was were be been being as at on by from that
    this these those it its into onto than but so per within across between over under above
    below both only also not no non yet however while whereas rather instead
    high higher highest low lower lowest strong strongly weak weakly barely broad broadly
    moderate moderately selective selectively marked marks rank ranks rate rates rated
    score scored scoring shows shown showing seen indicate indicates indicating suggest
    suggests point points pointing confirm confirms confirmed reflects agree agrees
    agreement disagree disagrees disagreeing consistent consistency inconsistent
    detect detected detection undetected present presence absent absence express expressed
    expression expressing level levels data evidence profile profiling notes noting note
    analysis analyses classification category specificity specific enrichment enriched
    enhanced distribution abundance abundant found finds looking looks appear appears
    appearing measured measurement returned experimental produce produced producing
    tissue tissues cell cells celltype type types single subtype lineage
    ms dvp cns covid il tcga hpa
    rna dna protein proteins transcript transcripts isoform isoforms gene genes coding
    encodes encode encoding receptor kinase enzyme hormone marker ligand cofactor
    binding binds bound cytokine surface transmembrane
    mutation mutations mutant variant variants allele alleles cause causes causing result
    results resulting constitutive
    associated association associations linked links relevance clinical disease diseases
    disorder cancer cancers tumor tumour tumors tumours carcinoma prognostic
    localized localised localization localisation location locations compartment
    subcellular secreted secretome membrane membrane-bound bound intracellular
    extracellular predicted prediction reliability
    highly consistent moderately uncertain group model models machine learning classifier
    feature features may meaningful biologically independent which appears
    upregulated regulation upregulation cellular downstream signaling signalling cascade
    cascades pathway pathways process processes function functional multifunctional role
    activation activating activate activates response responses stress human other others
    approved fda drug target targets druggable candidate potential known several many
    fraction fractions weak nuclear cytoplasmic membranous soma endfeet
    named include includes including form severe disease-associated related
    broad targeted screen screens screened converging converge converges matching match
    rests component reported label labels calls call marks marked picture focused islet-focused
    non-specific restricted diverse upon thereby role vital regulation raises lowers increases
    accelerates removal precursor overlaps read-through
    hematopoietic haematopoiesis hematopoiesis differentiation proliferation survival
    progenitor progenitors dendritic mature glial development developmental splicing
    alternative distinct multiple variants same overall
    """.split()
)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower())


def in_hay(term: str, hay: str) -> bool:
    t = norm(term).strip()
    if not t:
        return True
    for w in t.split():
        w = w.rstrip("s") if len(w) > 4 else w
        stem = w[:4] if len(w) > 5 else w
        if stem not in hay:
            return False
    return True


def check_summary(summary_json: Path, clean_json: Path) -> list[str]:
    clean = json.loads(clean_json.read_text(encoding="utf-8"))
    hay = norm(json.dumps(clean, ensure_ascii=False))
    summ = json.loads(summary_json.read_text(encoding="utf-8"))

    flags: list[str] = []
    for sec in summ.get("sections", []):
        for b in sec["bullets"]:
            miss: set[str] = set()
            for m in PROPER.finditer(b):
                phrase = m.group(1)
                parts = [p for p in re.split(r"[ /-]", phrase) if p]
                if all(p.lower() in STOP or p.isdigit() for p in parts):
                    continue
                if not in_hay(phrase, hay):
                    miss.add(phrase)
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", b):
                low = tok.lower()
                if low in STOP or tok[0].isupper():
                    continue
                if BIO_SUFFIX.search(low) or low in ORGANS:
                    if not in_hay(tok, hay):
                        miss.add(tok)
            for num in re.findall(r"\b\d{1,4}\b", b):
                if num not in hay:
                    miss.add(num)
            if miss:
                flags.append(f"[{sec['title']}] {b}\n      NOT IN DATA: {', '.join(sorted(miss))}")
    return flags


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--clean")
    ap.add_argument("--clean-dir", default="output")
    args = ap.parse_args(argv)

    total = 0
    for s in args.summaries:
        sp = Path(s)
        ensembl_id = sp.name.split("_")[0]
        cp = Path(args.clean) if args.clean else Path(args.clean_dir) / f"{ensembl_id}_hpa_clean.json"
        flags = check_summary(sp, cp)
        mark = "OK" if not flags else f"{len(flags)} FLAG(S)"
        print(f"\n===== {sp.name}  [{mark}] =====")
        for f in flags:
            print("  - " + f)
        total += len(flags)
    print(f"\nTOTAL FLAGS: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
