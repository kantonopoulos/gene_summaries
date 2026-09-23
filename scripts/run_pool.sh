#!/usr/bin/env bash
# Fetch -> clean -> summarize a pool of genes. Raw JSON in data/, clean + summary in output/.
set -euo pipefail
cd "$(dirname "$0")/.."

# Pilot gene set.
GENES=(
  ENSG00000251184   # (no symbol)
  ENSG00000111144   # LTA4H
  ENSG00000254647   # INS
  ENSG00000131095   # GFAP
  ENSG00000141510   # TP53
  ENSG00000146648   # EGFR
  ENSG00000122025   # FLT3
  ENSG00000118137   # APOA1
  ENSG00000204655   # MOG
  ENSG00000170782   # OR10A4
  ENSG00000132975   # GPR12
  ENSG00000228253   # MT-ATP8
  ENSG00000086504   # MRPL28
  ENSG00000167034   # NKX3-1
  ENSG00000232216   # IGHV3-43
  ENSG00000211895   # IGHA1
  ENSG00000255569   # TRAV1-1
  ENSG00000000003   # TSPAN6
  ENSG00000118434   # SPACA1
  ENSG00000083845   # RPS5
)

mkdir -p data output
for g in "${GENES[@]}"; do
  echo "############################ $g"
  python3 scripts/fetch_hpa.py "$g" --outdir data
  python3 scripts/clean_hpa.py "data/${g}_hpa.json" --outdir output
  python3 scripts/summarize_gene.py "output/${g}_hpa_clean.json" --outdir output
done
echo "############################ POOL DONE"
