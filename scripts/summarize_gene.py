#!/usr/bin/env python3
"""Turn a cleaned HPA gene JSON (<ID>_hpa_clean.json) into a structured bullet summary.

Uses a local Ollama model (default: gpt-oss:20b). One LLM call per section, each
given only the slice of the JSON it needs plus tailored instructions. Section
inclusion and the RNA-vs-protein split are decided by rules here, not the model.

Sections
  1. IDENTITY & FUNCTION          (always)
  2. EXPRESSION                   (always; protein view only if MS/DVP data exists)
  3. PROTEIN LOCALIZATION         (always)
  4. DISEASE & CLINICAL RELEVANCE (always)
  5. PROTEIN & MOLECULAR CONTEXT  (only when no subcellular AND no secretome data)

Usage
  python scripts/summarize_gene.py output/ENSG00000146648_hpa_clean.json
  python scripts/summarize_gene.py output/ENSG00000146648_hpa_clean.json --model gpt-oss:20b
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gpt-oss:20b"

# ----------------------------------------------------------------------------- #
# formatting contract (also enforced programmatically in postprocess/validate)  #
# ----------------------------------------------------------------------------- #
FORMAT_RULES = """FORMAT (obey exactly):
- Output ONLY a flat bullet list: each line "- ", ONE idea per bullet (never join two facts
  with a dash or semicolon), no header, no preamble.
- No trailing punctuation. Capital first letter. Under 22 words per bullet.
- Plain prose. Never start a bullet with a field name and colon. Never use bold or nesting.
- Keep category labels verbatim (e.g. "Tissue enhanced", "Evidence at protein level"); balance parentheses.
- When you list several items from a comma-separated input, keep the commas.
- State only what the data says; do not infer a mechanism, role or consequence that is not given.
- If the data is empty or uninformative, output exactly: "- No data available"."""

SYSTEM_MSG = (
    "You are a molecular biology curator writing terse, factual gene summary cards "
    "for scientists who scan them in seconds.\n"
    "CRITICAL: Use ONLY the JSON in SECTION DATA. Do NOT use outside knowledge. Every "
    "disease, gene, pathway, tissue, chromosome or number you mention MUST appear "
    "literally in that JSON. The card is about ONE gene, named in GENE: — never mention "
    "any other gene. If a field is missing or empty, write nothing about it.\n" + FORMAT_RULES
)

# ----------------------------------------------------------------------------- #
# rules                                                                         #
# ----------------------------------------------------------------------------- #
NA = {"", "not available", "na", "n/a", "none", "not applicable"}


def _real(v) -> bool:
    return isinstance(v, str) and v.strip().lower() not in NA


def has_protein_expression(loc: dict) -> bool:
    return _real(loc.get("Tissue specificity (MS)")) or _real(loc.get("Cell type specificity (DVP)"))


def has_localization(loc: dict, blood: dict) -> bool:
    return _real(loc.get("Subcellular location")) or _real(loc.get("Extracellular location")) or _real(
        blood.get("Secretome annotation")
    )


# ----------------------------------------------------------------------------- #
# section specs: (title, builder -> (data_dict, extra_instructions) or None)     #
# ----------------------------------------------------------------------------- #
def sec_identity(c: dict) -> tuple[dict, str]:
    info = c.get("EGFR INFORMATION") or _first_info(c)
    fn = c.get("PROTEIN FUNCTION", {})
    data = {
        "name": {k: info.get(k) for k in ("Protein", "Gene name")},
        "Protein class": info.get("Protein class"),
        "Number of transcripts": info.get("Number of transcripts"),
        "Protein function (UniProt)": fn.get("Protein function (UniProt)"),
        "Gene summary (Entrez)": fn.get("Gene summary (Entrez)"),
        "Molecular function (UniProt)": fn.get("Molecular function (UniProt)"),
        "Biological process (UniProt)": fn.get("Biological process (UniProt)"),
        "Ligand (UniProt)": fn.get("Ligand (UniProt)"),
    }
    instr = (
        "Answer 'What is this gene/protein?' in 4-5 short bullets: "
        "(1) protein full name followed by the gene symbol in parentheses, no synonyms "
        "(e.g. 'Insulin (INS)'); "
        "(2) one phrase generalising 'Protein class' into a category (e.g. 'druggable "
        "receptor tyrosine kinase, cancer- and disease-associated'); do not list every class; "
        "(3) one compact functional description from 'Protein function (UniProt)' + "
        "'Gene summary (Entrez)'; "
        "(4) the 'Molecular function (UniProt)' keywords, verbatim as a comma-separated list, "
        "nothing else in this bullet; "
        "(5) a SEPARATE bullet with the 'Biological process (UniProt)' keywords, only if that "
        "field is non-empty; "
        "(6) a SEPARATE short bullet: ligand/cofactor phrased as 'Binds X' plus the transcript "
        "count, only if 'Ligand (UniProt)' or 'Number of transcripts' is present. "
        "Never merge bullets 4, 5 and 6."
    )
    return data, instr


def sec_expression(c: dict) -> tuple[dict, str]:
    loc = c.get("PROTEIN EXPRESSION AND LOCALIZATION", {})
    tissue = c.get("TISSUE RNA EXPRESSION", {})
    celltype = c.get("CELL TYPE RNA EXPRESSION", {})
    info = _first_info(c)
    protein_view = has_protein_expression(loc)
    data = {
        "RNA tissue specificity": tissue.get("Tissue specificity"),
        "RNA single-cell type specificity": celltype.get("Single cell type specificity"),
        "Tissue cell type classification": celltype.get("Tissue cell type classification"),
        "Brain specificity": tissue.get("Brain specificity"),
        "Protein level (MS / DVP)": (
            {
                "Tissue profile": loc.get("Tissue profile"),
                "Tissue specificity (MS)": loc.get("Tissue specificity (MS)"),
                "Cell type specificity (DVP)": loc.get("Cell type specificity (DVP)"),
            }
            if protein_view
            else None
        ),
        "Protein evidence": info.get("Protein evidence"),
    }
    instr = (
        "Answer 'Where is it expressed?' in AT MOST 4 synthesis bullets, then the fixed "
        "bullets below. Do not just repeat tissue or cell-type names as a list, and do not "
        "add meta-commentary about which dataset was used. Combine the evidence: when RNA "
        "tissue and single-cell data point to the same biology, say so in one bullet; "
        + (
            "when protein (MS/DVP) and RNA agree, emphasise the agreement; when they "
            "disagree, say which looks more specific. "
            if protein_view
            else "there is no protein-level MS/DVP data, so judge only whether the RNA-based "
            "classifications are consistent, without mentioning protein expression. "
        )
        + "Numbers were given ONLY so you can tell strong from weak expression — describe it "
        "as 'high', 'low' or 'barely detected' and NEVER write a number, unit, nTPM or nCPM. "
        "'Tissue cell type classification' is the same single-cell data scored per tissue, "
        "not globally — frame it that way. "
        "Next bullet, ONLY if 'Brain specificity' is present: reproduce it closely. "
        "Last bullet: copy the 'Protein evidence' value verbatim — this is the curation "
        "evidence tier, not an expression statement, and must not contradict the bullets above."
    )
    return data, instr


def sec_localization(c: dict) -> tuple[dict, str]:
    loc = c.get("PROTEIN EXPRESSION AND LOCALIZATION", {})
    blood = c.get("PROTEINS IN BLOOD", {})
    sub = loc.get("Subcellular location")
    has_sub = _real(sub)
    data = {
        "Subcellular location": sub if has_sub else None,
        "Predicted location": loc.get("Predicted location"),
        "Extracellular location": loc.get("Extracellular location"),
        "Secretome annotation": blood.get("Secretome annotation"),
    }
    instr = (
        "Answer 'Where is the protein?' in exactly two bullets, no bullet per field: "
        + (
            "(1) main subcellular location(s) from 'Subcellular location', keeping the "
            "'(Reliability: ...)' suffix once; "
            if has_sub
            else "(1) state that there is no experimental subcellular localization data, "
            "then give the predicted compartment from 'Predicted location'; "
        )
        + "(2) whether it is secreted, membrane-bound and/or intracellular, merging "
        "'Predicted location', 'Extracellular location' and 'Secretome annotation' without "
        "repeating any phrase."
    )
    return data, instr


def _terms(s: str | None) -> set[str]:
    """Lowercased disease names from a comma-separated list, parentheticals dropped."""
    if not _real(s):
        return set()
    out = set()
    for part in re.split(r"\s*,\s*", s):
        name = re.sub(r"\(.*?\)", "", part).strip().lower().rstrip(".")
        if name and name != "no":
            out.add(name)
    return out


def blood_signal_assessment(upreg: str | None, models: str | None) -> str | None:
    up = _terms(upreg)
    # model string separates entries with " , "; each is "Name (tags)"
    mods = _terms(re.sub(r"\([^)]*\)", "", models or "")) if _real(models) else set()
    overlap = {u for u in up for m in mods if u in m or m in u}
    if up and overlap:
        names = ", ".join(sorted(overlap))
        return (
            f"high confidence for {names}, which appears in both the measured "
            "upregulated-in-disease data and an independent prediction model"
        )
    if up:
        return "a solid result — measured protein upregulation, not just a prediction"
    if mods:
        return (
            "low confidence — only a prediction model flags this, meaning one classifier "
            "used the gene as a feature, which may not be biologically meaningful"
        )
    return None


def sec_disease(c: dict) -> tuple[dict, str]:
    info = _first_info(c)
    fn = c.get("PROTEIN FUNCTION", {})
    cancer = c.get("CANCER & CELL LINES", {})
    blood = c.get("PROTEINS IN BLOOD", {})
    pclass = info.get("Protein class") or ""
    data = {
        "Named diseases / phenotypes (Entrez)": fn.get("Gene summary (Entrez)"),
        "Cancer specificity": cancer.get("Cancer specificity"),
        "Cell line specificity": cancer.get("Cell line specificity"),
        "Upregulated in disease": blood.get("Upregulated in disease"),
        "Disease prediction models": blood.get("Disease prediction model"),
        "blood_signal_assessment": blood_signal_assessment(
            blood.get("Upregulated in disease"), blood.get("Disease prediction model")
        ),
        "is_fda_drug_target": "FDA approved drug target" in pclass,
    }
    instr = (
        "Answer 'Why should I care?' in 3-4 bullets, clinical relevance only. Never restate "
        "the molecular mechanism, never quote protein-class labels, never begin a bullet with "
        "the gene or protein name, never join two facts with a dash or semicolon. "
        "Bullets: (1) named diseases and phenotypes taken ONLY from the 'Named diseases / "
        "phenotypes (Entrez)' text — do not add diseases from your own knowledge; "
        "(2) cancer relevance from 'Cancer specificity' and 'Cell line specificity'; "
        "(3) ONE bullet for the blood disease signal — if 'Upregulated in disease' names "
        "diseases, list them and end the bullet with the phrase from 'blood_signal_assessment'; "
        "if only 'Disease prediction models' exist, give just the 'blood_signal_assessment' "
        "caveat; skip this bullet entirely if 'blood_signal_assessment' is null. Do not write "
        "any other bullet about prediction models. "
        "Report only stated associations — no speculation ('linking it to tumor growth', "
        "'suggesting a role'), no 'this gene' / 'this protein'. "
        "If 'is_fda_drug_target' is true, the LAST bullet must read exactly 'FDA-approved drug target'."
    )
    return data, instr


def sec_context(c: dict) -> tuple[dict, str]:
    info = _first_info(c)
    fn = c.get("PROTEIN FUNCTION", {})
    loc = c.get("PROTEIN EXPRESSION AND LOCALIZATION", {})
    data = {
        "Number of transcripts": info.get("Number of transcripts"),
        "Protein interactions": info.get("Protein interactions"),
        "Predicted location": loc.get("Predicted location"),
        "Protein function (UniProt)": fn.get("Protein function (UniProt)"),
        "Molecular function (UniProt)": fn.get("Molecular function (UniProt)"),
        "Ligand (UniProt)": fn.get("Ligand (UniProt)"),
    }
    instr = (
        "Answer 'How does it fit into biology?'. Bullets: isoform/transcript count and any "
        "structural note; number of protein interactions; the main pathways or signalling "
        "cascades named in 'Protein function (UniProt)'; ligand/cofactor information. Show "
        "only the most informative items; skip anything absent."
    )
    return data, instr


SECTIONS = [
    ("IDENTITY & FUNCTION", sec_identity, "always"),
    ("EXPRESSION", sec_expression, "always"),
    ("PROTEIN LOCALIZATION", sec_localization, "always"),
    ("DISEASE & CLINICAL RELEVANCE", sec_disease, "always"),
    ("PROTEIN & MOLECULAR CONTEXT", sec_context, "if_no_localization"),
]


def _first_info(c: dict) -> dict:
    """The '<SYMBOL> INFORMATION' block, whatever the symbol is."""
    for k, v in c.items():
        if k.endswith("INFORMATION") and isinstance(v, dict):
            return v
    return {}


# ----------------------------------------------------------------------------- #
# llm + post-processing                                                         #
# ----------------------------------------------------------------------------- #
def call_ollama(model: str, instr: str, data: dict, anchor: str = "", timeout: int = 300) -> str:
    user = (
        f"GENE: {anchor}\n\n"
        f"SECTION INSTRUCTIONS:\n{instr}\n\n"
        f"SECTION DATA (JSON):\n{json.dumps(data, indent=2, ensure_ascii=False)}"
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0, "top_p": 0.9, "seed": 7},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["message"]["content"]


BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")


NUM_UNIT_RE = re.compile(r"\s*:?\s*~?\d[\d,.]*\s*(?:nTPM|nCPM|pTPM|TPM)\b", re.I)

# field names the model sometimes echoes as a "Label: value" prefix — stripped, not just flagged
KNOWN_PREFIXES = (
    "Brain specificity", "Protein evidence", "Predicted location", "Subcellular location",
    "Tissue profile", "Cancer specificity", "Cell line specificity", "Molecular function",
    "Biological process", "Protein function", "Gene summary", "Protein class",
    "Tissue specificity", "Single cell type specificity", "Tissue cell type classification",
    "Extracellular location", "Secretome annotation", "Protein level",
)


def postprocess(
    raw: str, section: str = "", gene: str = "", protein_name: str = ""
) -> tuple[list[str], list[str]]:
    """Extract bullets and normalise them to the formatting contract.

    Returns (bullets, notes) where notes records every fix applied.
    """
    notes: list[str] = []
    bullets: list[str] = []
    for line in raw.splitlines():
        m = BULLET_RE.match(line)
        if not m:
            if line.strip():
                notes.append(f"dropped non-bullet line: {line.strip()[:60]!r}")
            continue
        text = re.sub(r"\s+", " ", m.group(1)).strip()
        text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)  # strip bold/italic
        text = re.sub(r"^[\*_`]+\s*", "", text)
        pm = re.match(r"^([A-Z][A-Za-z /()]{2,40}):\s+(.+)", text)
        if pm and pm.group(1) in KNOWN_PREFIXES:
            text = pm.group(2)
            text = text[0].upper() + text[1:]
            notes.append(f"stripped field-name prefix {pm.group(1)!r}")
        if section == "EXPRESSION":
            new = NUM_UNIT_RE.sub("", text)
            new = re.sub(r":\s*~?\d[\d,.]*\s*(?=[,)])", "", new)  # "(Placenta: 61.8)" -> "(Placenta)"
            new = re.sub(r"\(\s*([,;]\s*)+", "(", new).replace("( ", "(").replace(" )", ")")
            if new != text:
                notes.append("stripped expression numbers/units")
            text = new.strip()
        stripped = text.rstrip(".;,: ")
        if stripped != text:
            notes.append(f"trimmed trailing punctuation: {text[-1]!r}")
        text = stripped
        if text and text[0].islower():
            notes.append(f"capitalised: {text[:40]!r}")
            text = text[0].upper() + text[1:]
        wc = len(text.split())
        if wc > 30:
            notes.append(f"long bullet ({wc} words): {text[:50]!r}")
        if text.count("(") != text.count(")"):
            notes.append(f"unbalanced parentheses: {text[:60]!r}")
        if re.search(r"\s[–—]\s|; ", text):
            notes.append(f"two facts joined in one bullet: {text[:60]!r}")
        head = text.split(":", 1)[0]
        if re.match(r"^[A-Z][A-Za-z ]{1,30}$", head) and "(" not in head and ":" in text:
            notes.append(f"bullet starts with a field-name prefix: {text[:50]!r}")
        if section == "DISEASE & CLINICAL RELEVANCE":
            leads = [n for n in (gene, protein_name) if n]
            if any(re.match(rf"^{re.escape(n)}(?=[\s'’,.]|$)", text, re.I) for n in leads):
                notes.append(f"disease bullet starts with the gene/protein name: {text[:50]!r}")
        if text:
            bullets.append(text)

    # de-dup, preserving order
    seen: set[str] = set()
    deduped = []
    for b in bullets:
        key = b.lower()
        if key in seen:
            notes.append(f"dropped duplicate bullet: {b[:50]!r}")
            continue
        seen.add(key)
        deduped.append(b)

    if not deduped:
        deduped = ["No data available"]
    if len(deduped) > 7:
        notes.append(f"section has {len(deduped)} bullets (>7)")
    return deduped, notes


def render(gene: str, ensembl_id: str, sections: list[tuple[str, list[str]]]) -> str:
    out = [f"# {gene} — gene summary", f"_Ensembl {ensembl_id} · source: Human Protein Atlas_", ""]
    for title, bullets in sections:
        out.append(f"## {title.upper()}")
        out.extend(f"- {b}" for b in bullets)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# ----------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clean_json", help="path to <ID>_hpa_clean.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    path = Path(args.clean_json)
    clean = json.loads(path.read_text(encoding="utf-8"))
    ensembl_id = path.name.split("_")[0]
    info = _first_info(clean)
    gene = (info.get("Gene name") or ensembl_id).split(" ")[0]
    protein_name = info.get("Protein") or ""

    loc = clean.get("PROTEIN EXPRESSION AND LOCALIZATION", {})
    blood = clean.get("PROTEINS IN BLOOD", {})
    localization_exists = has_localization(loc, blood)

    rendered_sections: list[tuple[str, list[str]]] = []
    all_notes: dict[str, list[str]] = {}
    for title, builder, gate in SECTIONS:
        if gate == "if_no_localization" and localization_exists:
            print(f"skip  {title}  (localization data present)")
            continue
        data, instr = builder(clean)
        print(f"call  {title} ...", flush=True)
        raw = call_ollama(args.model, instr, data, anchor=f"{gene} ({protein_name})")
        bullets, notes = postprocess(raw, section=title, gene=gene, protein_name=protein_name)
        rendered_sections.append((title, bullets))
        if notes:
            all_notes[title] = notes

    md = render(gene, ensembl_id, rendered_sections)
    outdir = Path(args.outdir) if args.outdir else path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / f"{ensembl_id}_summary.md"
    md_path.write_text(md, encoding="utf-8")
    json_path = outdir / f"{ensembl_id}_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "ensembl_id": ensembl_id,
                "gene": gene,
                "model": args.model,
                "sections": [{"title": t, "bullets": b} for t, b in rendered_sections],
                "format_notes": all_notes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nwrote {md_path}")
    print(f"wrote {json_path}")
    if all_notes:
        print("\n--- formatting fixes / warnings ---")
        for title, notes in all_notes.items():
            print(f"[{title}]")
            for n in notes:
                print(f"  - {n}")
    else:
        print("\nformatting: clean, no fixes needed")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
