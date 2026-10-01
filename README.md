# HPA gene summaries

Generates a short, fact-grounded summary card and an infographic for every gene in the
[Human Protein Atlas](https://www.proteinatlas.org). It uses Gemini on Vertex AI.

For each gene:

1. **Fetch** its HPA page and JSON.
2. **Clean** the data down to the fields the summaries use.
3. **Summarize** it with `gemini-3.8-flash`, one call per section:
   - Identity & function
   - Expression
   - Protein localization
   - Disease & clinical relevance

   The model may only state what is in the HPA data.
4. **Visualize** the summary as a 16:9 JPEG with `gemini-3-pro-image`.

## Setup

- Python 3 with `google-genai`, `google-auth`, `requests`, `beautifulsoup4`, `Pillow`
- Vertex AI credentials for project `scilifelab-hpa-proj-1`; see
  [docs/setup_gemini_vertex_ai.md](docs/setup_gemini_vertex_ai.md)

## One gene (local)

```bash
python3 scripts/fetch_hpa.py ENSG00000146648 --outdir data
python3 scripts/clean_hpa.py data/ENSG00000146648_hpa.json --outdir output
python3 scripts/summarize_gene.py output/ENSG00000146648_hpa_clean.json
python3 scripts/visualize_gene.py output/ENSG00000146648_summary.md
```

## All genes (batch)

The full run of about 20k genes uses two Vertex AI batch jobs (text, then images) that run on
Google's side. Step-by-step commands, progress checks, retries and timings are in
**[docs/batch_pipeline.md](docs/batch_pipeline.md)**.

## Scripts

| Script | Purpose |
|---|---|
| `fetch_hpa.py` / `clean_hpa.py` | Fetch / clean one gene |
| `fetch_all.py` | Fetch + clean every gene in `data/proteinatlas.tsv` (resumable) |
| `summarize_gene.py` | Summary prompts and formatting rules; summarizes one gene |
| `visualize_gene.py` | Infographic prompt and style guide; draws one gene |
| `check_grounding.py` | Flags summary terms that are not in the source data |
| `batch_text_prepare.py` → `batch_submit.py` → `batch_text_collect.py` | Text batch: build input, submit, collect summaries |
| `batch_image_prepare.py` → `batch_submit.py` → `batch_image_export.py` | Image batch: build input, submit, download JPEGs |
| `bq.py` | BigQuery helpers for the batch scripts |

Prompts live in `summarize_gene.py` and `visualize_gene.py`. Local runs and batches both use
them, so a change there applies to both.

## Outputs

`data/` and `output/` are gitignored.

| Path | Content |
|---|---|
| `data/<ID>_hpa.json` | Raw HPA page + JSON |
| `output/<ID>_hpa_clean.json` | Cleaned input to the summarizer |
| `output/<ID>_summary.md` / `.json` | Summary card (Markdown) / structured version with token usage |
| `output/images/<ID>_<SYMBOL>.jpg` | Infographic (batch run) |
| `output/<ID>_visual.jpg` | Infographic (local `visualize_gene.py` run) |
