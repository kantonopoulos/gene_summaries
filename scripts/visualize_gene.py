#!/usr/bin/env python3
"""Turn a gene's Markdown summary (<ID>_summary.md) into an infographic image.

Uses Gemini 3 Pro Image ("Nano Banana Pro") via Vertex AI (see
scripts/docs/setup_gemini_vertex_ai.md). The full summary Markdown is handed to the
model together with a fixed style-guide prompt; the model draws a 16:9 infographic
from it directly (no layout logic here — that's the model's job). One shot, no
validation or retry — whatever the model returns is kept.

Usage
  python scripts/visualize_gene.py output/ENSG00000254647_summary.md
  python scripts/visualize_gene.py output/ENSG00000254647_summary.md --outdir output
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from google import genai
from google.genai import types

VERTEX_PROJECT = "scilifelab-hpa-proj-1"
VERTEX_LOCATION = "global"  # regional locations 404 on these models
DEFAULT_MODEL = "gemini-3-pro-image"
ASPECT_RATIO = "16:9"

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    return _client


STYLE_GUIDE = """STYLE GUIDE (obey exactly):
- Header: Render the TITLE text as a bold, left-aligned title (plain designed typography —
  never a literal '#' character or any markdown punctuation) with an engaging narrative
  subtitle based on its role.
- Fluid Layout & Composition: Avoid rigid column grids or stiff boxes. Arrange content
  organically around a prominent central biological focal point tailored to the protein
  (e.g., a detailed organ or cell membrane). Connect dynamic text nodes using fluid
  biological visual streams (such as blood vessels, signaling paths, or cellular channels)
  that naturally guide the viewer's eyes.
- Organic Scaling: Integrate multi-scale illustrations directly into the artwork—seamlessly
  zooming from organ scale to cell type down to subcellular scale (e.g. organelles, vesicles,
  receptors as simple stylized icons — not literal molecular diagrams).
- Adaptive Color Palette: Match colors to the gene's primary biological context (e.g., deep
  navy, cyan, and slate gray for intracellular/ribosomal proteins; crimson/amber for
  blood/liver; soft blues/teal for endocrine). Use a clean off-white background (hex F8F9FA)
  with subtle glows and drop shadows.
- Typography & Details: High-contrast sans-serif font. Keep text brief and integrated
  naturally alongside vector illustrations using clean badge tags and callout lines rather
  than blocky text boxes."""

NEGATIVE_CONSTRAINTS = """HARD CONSTRAINTS (do not violate; the first two are the most important):
- No duplicated information: each fact and section label appears exactly once in the image —
  never repeat the same fact, callout, or icon in two places.
- No misspelled or garbled text: every rendered word must be spelled correctly.
- No chemical structure diagrams of any kind (no skeletal/hexagon sugar-ring formulas, no
  amino acid chains, no protein ribbon diagrams or crystal-structure renderings).
- No charts, graphs, bar plots, line plots, or any data-visualization widgets.
- No literal markdown or typographic marker characters anywhere in the image: never draw
  '#', '##', '-', '*', or similar symbols as glyphs — text is rendered as clean typography
  only."""


def strip_markdown(summary_md: str) -> str:
    """Convert '# Title' / '## SECTION' / '- bullet' into plain structural labels,
    so the model never sees literal '#'/'-' characters to (mis)render as glyphs."""
    lines = []
    for line in summary_md.splitlines():
        if line.startswith("## "):
            lines.append(f"SECTION LABEL: {line[3:].strip()}")
        elif line.startswith("# "):
            lines.append(f"TITLE: {line[2:].strip()}")
        elif line.startswith("- "):
            lines.append(f"FACT: {line[2:].strip()}")
        elif line.strip():
            lines.append(line.strip())
    return "\n".join(lines)


def build_prompt(summary_md: str) -> str:
    structured = strip_markdown(summary_md)
    return (
        f"Create a single infographic image ({ASPECT_RATIO} aspect ratio) summarizing this gene, "
        "using ONLY the facts below. Do not invent facts not present here. The labels "
        "below (TITLE, SECTION LABEL, FACT) describe each line's ROLE only — render its "
        "content as clean designed typography, never the label word itself. Give every "
        "SECTION LABEL its own clearly labelled zone in the artwork — do not skip or "
        "merge sections.\n\n"
        f"{STYLE_GUIDE}\n\n"
        f"{NEGATIVE_CONSTRAINTS}\n\n"
        f"GENE SUMMARY:\n{structured}"
    )


def call_gemini_image(model: str, prompt: str) -> bytes:
    resp = get_client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(aspect_ratio=ASPECT_RATIO),
        ),
    )
    for part in resp.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data
    raise RuntimeError("model returned no image")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary_md", help="path to <ID>_summary.md")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    path = Path(args.summary_md)
    summary_md = path.read_text(encoding="utf-8")
    ensembl_id = path.name.split("_")[0]

    prompt = build_prompt(summary_md)
    print("call  gemini-3-pro-image ...", flush=True)
    image_bytes = call_gemini_image(args.model, prompt)

    outdir = Path(args.outdir) if args.outdir else path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{ensembl_id}_visual.png"
    out_path.write_bytes(image_bytes)
    print(f"wrote {out_path}  ({out_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
