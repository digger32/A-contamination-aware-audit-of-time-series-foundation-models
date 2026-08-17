#!/usr/bin/env python3
"""Download and verify the datasets and checkpoints.

  --check      HEAD-verify every Zenodo record and HF repo WITHOUT downloading
               (dataset-availability guardrail: run this BEFORE committing compute).
  (default)    download Monash .tsf zips -> data/raw/monash/, GIFT-Eval via `hf`,
               Lag-Llama checkpoint via `hf` -> data/checkpoints/.
               Skip-if-exists by default; --force to re-download (frozen-artifact rule).

Records sha256 of every downloaded file into data/DOWNLOADS.sha256.
"""
import argparse
import hashlib
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402

RAW = ROOT / "data" / "raw" / "monash"
CKPT = ROOT / "data" / "checkpoints"
GIFT = ROOT / "data" / "raw" / "gifteval"


def zenodo_url(rec: int, fname: str) -> str:
    return f"https://zenodo.org/record/{rec}/files/{fname.replace('.tsf', '.zip')}"


def head_ok(url: str) -> bool:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status in (200, 302)
    except Exception as e:
        print(f"    !! {e}")
        return False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-gifteval", action="store_true")
    a = ap.parse_args()

    mon = yaml.safe_load((ROOT / "configs" / "datasets_monash.yaml").read_text())
    gif = yaml.safe_load((ROOT / "configs" / "datasets_gifteval.yaml").read_text())
    failures = []

    print("== Monash (Zenodo) ==")
    for name, cfg in mon["datasets"].items():
        url = zenodo_url(cfg["zenodo"], cfg["file"])
        if a.check:
            ok = head_ok(url)
            print(f"  [{'ok' if ok else 'FAIL'}] {name}  {url}")
            if not ok:
                failures.append(name)
            continue
        dest = RAW / f"{name}.zip"
        if dest.exists() and not a.force:
            print(f"  [skip] {name} (exists)")
        else:
            RAW.mkdir(parents=True, exist_ok=True)
            print(f"  [dl] {name} <- {url}")
            urllib.request.urlretrieve(url, dest)
        with zipfile.ZipFile(dest) as z:
            z.extractall(RAW)
        with (ROOT / "data" / "DOWNLOADS.sha256").open("a") as fh:
            fh.write(f"{sha256(dest)}  {dest.relative_to(ROOT)}\n")

    print("== GIFT-Eval (HF: Salesforce/GiftEval) ==")
    if a.check:
        ok = head_ok(f"https://huggingface.co/datasets/{gif['hf_repo']}")
        print(f"  [{'ok' if ok else 'FAIL'}] {gif['hf_repo']}")
        if not ok:
            failures.append("gifteval")
    elif not a.skip_gifteval:
        GIFT.mkdir(parents=True, exist_ok=True)
        subprocess.run(["hf", "download", gif["hf_repo"], "--repo-type=dataset",
                        "--local-dir", str(GIFT)], check=True)

    print("== Lag-Llama checkpoint ==")
    if a.check:
        ok = head_ok("https://huggingface.co/time-series-foundation-models/Lag-Llama")
        print(f"  [{'ok' if ok else 'FAIL'}] time-series-foundation-models/Lag-Llama")
        if not ok:
            failures.append("lag-llama")
    else:
        CKPT.mkdir(parents=True, exist_ok=True)
        if not (CKPT / "lag-llama.ckpt").exists() or a.force:
            subprocess.run(["hf", "download", "time-series-foundation-models/Lag-Llama",
                            "lag-llama.ckpt", "--local-dir", str(CKPT)], check=True)
        else:
            print("  [skip] lag-llama.ckpt (exists)")

    if a.check:
        print("\nCHECK RESULT:", "ALL REACHABLE" if not failures
              else f"FAILURES: {failures} — fix record ids / access before any run")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
