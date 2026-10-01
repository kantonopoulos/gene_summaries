# Running the full pipeline with Vertex AI batches

Generates a text summary and an infographic JPEG for every gene in the Human Protein
Atlas, using two Vertex AI batch jobs (text, then images). The jobs run on Google's side,
so the laptop can be closed while they run. Run all commands from the repo root.

```
proteinatlas.tsv ──fetch_all──▶ data/*_hpa.json + output/*_hpa_clean.json
                 ──batch 1 (text)──▶ output/*_summary.md / .json
                 ──batch 2 (images)──▶ output/images/*.jpg
```

## Prerequisites

- Vertex AI access is already set up (see [setup_gemini_vertex_ai.md](setup_gemini_vertex_ai.md)).
  Everything runs under those credentials and is billed to `scilifelab-hpa-proj-1`.
- Batch input/output lives in BigQuery: dataset `gene_summaries_batch` (location US) in
  that project. It is created automatically on first use.
- Python packages: `google-genai`, `google-auth`, `requests`, `beautifulsoup4`, `Pillow`.

## Reference timings (full run, 20,162 genes, Sept/Oct 2026)

| Step | Where | Time |
|---|---|---|
| 1. Fetch + clean | laptop | ~4.5 h |
| 2. Build text input | laptop | ~1 min |
| 3. Text batch (80,648 requests) | Google | a few hours (max 24 h) |
| 4. Collect summaries | laptop | ~2 min |
| 5. Build image input | laptop | ~1 min |
| 6. Image batch (20,162 requests) | Google | a few hours (max 24 h) |
| 7. Download JPEGs (~4.3 GB) | laptop | ~45 min |

---

## 1. Get the gene list and fetch + clean the HPA data

Download `proteinatlas.tsv` from the HPA *Downloads* page (`proteinatlas.tsv.zip`), unzip it,
and put it at `data/proteinatlas.tsv`. Then:

```bash
python3 scripts/fetch_all.py
```

- For each gene it fetches the HPA page + JSON to `data/<ID>_hpa.json`, then runs the clean
  step to write `output/<ID>_hpa_clean.json`.
- It is rate-limited (3 requests/s to HPA) and **resumable**. If it is interrupted, rerun the
  same command and genes already on disk are skipped. Failed genes go to
  `output/fetch_failures.tsv`, and a rerun retries them.
- To keep the Mac awake during the run: `caffeinate -i python3 scripts/fetch_all.py`
- Smoke test first: `python3 scripts/fetch_all.py --limit 20`

**Progress:** a status line every 100 genes with an ETA. To log to a file and follow it:
```bash
python3 -u scripts/fetch_all.py > data/fetch_all.log 2>&1 &
tail -f data/fetch_all.log
```

## 2. Build the text batch input (batch 1)

```bash
python3 scripts/batch_text_prepare.py --load
```

- Builds one request per gene × section (4 per gene). It uses the same prompts and settings
  as `summarize_gene.py`.
- Writes `data/batch/text_requests.jsonl` and loads it into BigQuery table `text_in`,
  replacing that table if it already exists.
- Without `--load`, it only builds the local file.

## 3. Run the text batch

```bash
python3 scripts/batch_submit.py text
```

Submits `text_in` → `text_out` with `gemini-3.8-flash`. The job is recorded in
`data/batch/jobs.jsonl`.

> **New run?** Use a fresh output table so old results are not mixed in, e.g.
> `--dest text_out_2027`, and pass the same name to the collect step (`--table text_out_2027`).

**Progress:**
```bash
python3 scripts/batch_submit.py --status
```
`JOB_STATE_PENDING` → `RUNNING` → `SUCCEEDED`, with ok/failed counts.

## 4. Collect the text summaries

```bash
python3 scripts/batch_text_collect.py
```

- Reads `text_out` and writes `output/<ID>_summary.md` + `output/<ID>_summary.json` per gene.
  The output is identical in format to a local `summarize_gene.py` run.
- A gene is only written if **all** its sections succeeded. Failed request keys go to
  `data/batch/text_failures.txt`, and the script prints the retry command.

**Retry failures** (usually 0–few; transient Google errors):
```bash
python3 scripts/batch_text_prepare.py --only-keys data/batch/text_failures.txt \
    --out data/batch/text_retry.jsonl --load --table text_retry_in
python3 scripts/batch_submit.py text --src text_retry_in --dest text_retry_out
python3 scripts/batch_text_collect.py --table text_out --table text_retry_out
```
For a handful of genes it is faster to run them locally instead:
`python3 scripts/summarize_gene.py output/<ID>_hpa_clean.json --outdir output`

## 4b. Fact check the summaries (grounding)

Optional, but recommended before generating images. It checks that every named entity in a
summary (disease, tissue, cell type, organ, gene, number) appears in that gene's HPA data,
i.e. the model did not invent facts.

```bash
python3 scripts/check_grounding.py output/ENSG*_summary.json --clean-dir output \
    > data/batch/grounding_report.txt
tail -1 data/batch/grounding_report.txt          # TOTAL FLAGS: …
```

To list only the flags that need a human look, filter out the known wording false positives:
```bash
grep "NOT IN DATA" data/batch/grounding_report.txt \
  | grep -vE "NOT IN DATA: (Bulk RNA.*|(Protein|Human) IHC|High Brain RNA|(Displays|Exhibits|Demonstrates|Shows|Additional|Produces|Elevated|Concordant) [A-Za-z0-9]+)$"
```

- The report lists each flagged bullet with the term that was not found
  (`NOT IN DATA: …`).
- **Most flags are false positives from wording.** The checker treats a capitalised word at
  the start of a bullet followed by an HPA label as an unknown name, e.g. "Displays Low cancer
  specificity", "Exhibits Cancer enriched …". "Bulk RNA" is also flagged often. In the
  2026 run, 11,554 raw flags came down to 25 after this filter, and all 25 were valid
  paraphrases or groupings of the data (e.g. "kidney" for a list of kidney cell types).
- If a flag is a real invented fact, fix that gene by rerunning it locally:
  `python3 scripts/summarize_gene.py output/<ID>_hpa_clean.json --outdir output`.
  Then rebuild its image request in step 5.

## 5. Build the image batch input (batch 2)

```bash
python3 scripts/batch_image_prepare.py --load
```

- Builds one request per gene from `output/<ID>_summary.md`. It uses the same prompt and
  settings as `visualize_gene.py`: 16:9, JPEG at quality 90.
- Writes `data/batch/image_requests.jsonl` and loads it into BigQuery table `image_in`.
- If the style guide in `visualize_gene.py` changes, rerun this step before submitting.

## 6. Run the image batch

```bash
python3 scripts/batch_submit.py image
```

Submits `image_in` → `image_out` with `gemini-3-pro-image`. For a new run, use a fresh
`--dest` here too, as in step 3.

**Progress:** same command as for text: `python3 scripts/batch_submit.py --status`

## 7. Download the JPEGs

```bash
caffeinate -i python3 scripts/batch_image_export.py
```

- Extracts only the image bytes from `image_out` and saves
  `output/images/<ENSG>_<SYMBOL>.jpg` (or `<ENSG>.jpg` for genes without a symbol).
- It is **resumable**: genes that already have a file are not downloaded again, so rerun the
  same command after any interruption. Dropped connections are retried automatically.
- Progress is printed every 2,000 images.
- Genes whose response had no image go to `data/batch/image_failures.txt`. Expect ~0.1% of
  genes: the model ran out of tokens or rejected its own image because of misspelled text.

**Regenerate missing images** through a small retry batch:
```bash
python3 scripts/batch_image_prepare.py --only-keys data/batch/image_failures.txt \
    --out data/batch/image_retry.jsonl --load --table image_retry_in
python3 scripts/batch_submit.py image --src image_retry_in --dest image_retry_out
python3 scripts/batch_image_export.py --table image_retry_out
```
For a handful of genes, local runs are quicker:
`python3 scripts/visualize_gene.py output/<ID>_summary.md --outdir output`. This writes
`output/<ID>_visual.jpg`; rename it to `<ENSG>_<SYMBOL>.jpg` and move it into `output/images/`.
It can take 2–3 attempts per gene.

## 8. Package for sharing

```bash
cd output
zip -r -9 gene_summaries_json.zip *_summary.json      # ~19 MB
zip -r -0 gene_images_jpg.zip images                  # ~4.6 GB; JPEGs don't compress further
```

## Where things live

| What | Location |
|---|---|
| Raw / clean HPA data | `data/<ID>_hpa.json`, `output/<ID>_hpa_clean.json` |
| Batch inputs (local copies) | `data/batch/*_requests.jsonl` |
| Submitted jobs log | `data/batch/jobs.jsonl` |
| Batch tables | BigQuery `scilifelab-hpa-proj-1.gene_summaries_batch`: `text_in/out`, `image_in/out`, `image_export` |
| Summaries | `output/<ID>_summary.md`, `output/<ID>_summary.json` |
| Images | `output/images/*.jpg` |
| Failures to retry | `data/batch/text_failures.txt`, `data/batch/image_failures.txt` |
