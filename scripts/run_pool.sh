#!/usr/bin/env bash
# Fetch -> clean -> summarize a pool of genes. Raw JSON in data/, clean + summary in output/.
set -euo pipefail
cd "$(dirname "$0")/.."

GENES=(
  ENSG00000146648   # EGFR  - receptor tyrosine kinase
  ENSG00000122025   # FLT3  - RTK, leukemia
  ENSG00000141510   # TP53  - tumour suppressor, ubiquitous
  ENSG00000131095   # GFAP  - astrocyte marker, brain
  ENSG00000254647   # INS   - insulin, secreted hormone
)

mkdir -p data output
for g in "${GENES[@]}"; do
  echo "############################ $g"
  python3 scripts/fetch_hpa.py "$g" --outdir data
  python3 scripts/clean_hpa.py "data/${g}_hpa.json" --outdir output
  python3 scripts/summarize_gene.py "output/${g}_hpa_clean.json" --outdir output
done
echo "############################ POOL DONE"
