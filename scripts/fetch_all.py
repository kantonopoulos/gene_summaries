#!/usr/bin/env python3
"""Fetch + clean every gene listed in the HPA bulk TSV (data/proteinatlas.tsv).

Resumable: genes whose <ID>_hpa.json already exists are not re-fetched (only
re-cleaned if the clean file is missing). Requests to proteinatlas.org are
globally rate-limited across worker threads; transient errors (timeouts, 429,
5xx) are retried with backoff. Genes that still fail are listed in
<outdir>/fetch_failures.tsv. Rerun the same command to pick them up.

Output layout (what the batch scripts expect):
    data/<ID>_hpa.json          raw  (fetch_hpa.fetch_gene)
    output/<ID>_hpa_clean.json  clean (clean_hpa.clean)

Usage
  python scripts/fetch_all.py                      # all genes in the TSV
  python scripts/fetch_all.py --limit 20           # first 20 (smoke test)
  python scripts/fetch_all.py --workers 4 --rps 4  # tune politeness
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_hpa  # noqa: E402
from clean_hpa import clean  # noqa: E402


class RateLimiter:
    """At most `rps` request starts per second, shared across threads."""

    def __init__(self, rps: float):
        self.interval = 1.0 / rps
        self.lock = threading.Lock()
        self.next_t = time.monotonic()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        time.sleep(max(0.0, t - now))


def install_rate_limit(limiter: RateLimiter) -> None:
    """Route fetch_hpa's requests.get through the limiter (it makes 2 calls per gene)."""
    real_get = requests.get

    def limited_get(*a, **kw):
        limiter.wait()
        return real_get(*a, **kw)

    fetch_hpa.requests.get = limited_get


def is_transient(exc: Exception) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # atomic: an interrupted run never leaves a half-written file


def process(gene_id: str, rawdir: Path, cleandir: Path, retries: int) -> str:
    raw_path = rawdir / f"{gene_id}_hpa.json"
    clean_path = cleandir / f"{gene_id}_hpa_clean.json"
    status = "cached"
    if not raw_path.exists():
        for attempt in range(retries + 1):
            try:
                payload = fetch_hpa.fetch_gene(gene_id)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == retries or not is_transient(exc):
                    raise
                time.sleep(min(60, 5 * 2**attempt))
        write_json(raw_path, payload)
        status = "fetched"
    else:
        payload = None
    if not clean_path.exists():
        if payload is None:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        write_json(clean_path, clean(payload))
    return status


def read_gene_ids(tsv: Path) -> list[str]:
    with tsv.open(encoding="utf-8", newline="") as f:
        return [row["Ensembl"].strip() for row in csv.DictReader(f, delimiter="\t")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", default="data/proteinatlas.tsv")
    ap.add_argument("--rawdir", default="data")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rps", type=float, default=3.0, help="max HTTP requests/second to HPA (2 per gene)")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N genes")
    args = ap.parse_args(argv)

    rawdir, cleandir = Path(args.rawdir), Path(args.outdir)
    rawdir.mkdir(parents=True, exist_ok=True)
    cleandir.mkdir(parents=True, exist_ok=True)
    ids = read_gene_ids(Path(args.tsv))[: args.limit]
    install_rate_limit(RateLimiter(args.rps))

    counts = {"fetched": 0, "cached": 0, "failed": 0}
    failures: list[tuple[str, str]] = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, g, rawdir, cleandir, args.retries): g for g in ids}
        for i, fut in enumerate(as_completed(futs), 1):
            g = futs[fut]
            try:
                counts[fut.result()] += 1
            except Exception as exc:  # noqa: BLE001
                counts["failed"] += 1
                failures.append((g, f"{type(exc).__name__}: {exc}"))
            if i % 100 == 0 or i == len(ids):
                el = time.monotonic() - t0
                rate = counts["fetched"] / el if el else 0
                left = len(ids) - i
                eta = f"{left / rate / 3600:.1f}h" if rate else "?"
                print(f"[{i}/{len(ids)}] {counts}  {rate:.2f} genes/s  eta {eta}", flush=True)

    fail_path = cleandir / "fetch_failures.tsv"
    if failures:
        fail_path.write_text("".join(f"{g}\t{e}\n" for g, e in sorted(failures)), encoding="utf-8")
        print(f"{len(failures)} failures -> {fail_path} (rerun to retry)")
    elif fail_path.exists():
        fail_path.unlink()
    print(f"done: {counts}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
