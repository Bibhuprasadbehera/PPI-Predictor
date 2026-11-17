"""Download mmCIF files for table PDB ids that have no local coordinates.

Existing .pdb files under DEEPINTERACT/Positive_data/pdb are kept and are not
downloaded again. A file that already exists and parses as gzipped mmCIF is
skipped, so the run can be resumed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEEP = Path("/home/bibhu/Documents/DeepInteract/DEEPINTERACT")
PAIRS = ROOT / "model_data" / "pairs.tsv"
PDB_DIR = DEEP / "Positive_data" / "pdb"
OUT = ROOT / "model_data" / "structures"
LOG = ROOT / "model_data" / "structure_download.log"
MANIFEST = ROOT / "model_data" / "structure_download.tsv"
WORKERS = 8
TIMEOUT_S = 180
RETRIES = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(event: str, **fields) -> None:
    parts = [f"INFO  {event}"]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    line = " ".join(parts)
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def script_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def local_pdb(pdb_id: str) -> Path | None:
    for name in (f"{pdb_id}.pdb", f"{pdb_id.upper()}.pdb", f"{pdb_id.lower()}.pdb"):
        path = PDB_DIR / name
        if path.exists() and path.stat().st_size > 1000:
            return path
    return None


def valid_cif_gz(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 64:
        return False
    try:
        with gzip.open(path, "rb") as handle:
            head = handle.read(200)
    except (OSError, EOFError, gzip.BadGzipFile):
        return False
    return b"data_" in head or b"loop_" in head


def download_one(pdb_id: str) -> dict:
    dest = OUT / f"{pdb_id}.cif.gz"
    if valid_cif_gz(dest):
        return {"pdb_id": pdb_id, "status": "present", "bytes": dest.stat().st_size}
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.cif.gz"
    tmp = dest.with_suffix(".cif.gz.part")
    last_error = ""
    for attempt in range(1, RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "DeepInteract/structure-rebuild"})
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                status = getattr(response, "status", 200)
                payload = response.read()
            if status != 200 or len(payload) < 64 or payload[:2] != b"\x1f\x8b":
                last_error = f"bad_response status={status} bytes={len(payload)}"
            else:
                tmp.write_bytes(payload)
                if valid_cif_gz(tmp):
                    tmp.replace(dest)
                    return {"pdb_id": pdb_id, "status": "downloaded", "bytes": dest.stat().st_size}
                last_error = "gzip_parse_failed"
                tmp.unlink(missing_ok=True)
        except urllib.error.HTTPError as exc:
            last_error = f"http_{exc.code}"
            if exc.code in (404, 410):
                break
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"
        time.sleep(min(2 ** attempt, 20))
    tmp.unlink(missing_ok=True)
    return {"pdb_id": pdb_id, "status": "failed", "bytes": 0, "error": last_error}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(PAIRS, sep="\t")
    pdb_ids = sorted(set(pairs.pdb_id.astype(str)))
    have_pdb = []
    need = []
    for pdb_id in pdb_ids:
        if local_pdb(pdb_id) is not None:
            have_pdb.append(pdb_id)
        else:
            need.append(pdb_id)
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    log(
        "run started",
        run_id=run_id,
        pipeline="structure_download",
        utc_start=utc_now(),
        script_sha=script_sha(),
        pairs=len(pairs),
        pdb_ids=len(pdb_ids),
        local_pdb=len(have_pdb),
        to_download=len(need),
        workers=WORKERS,
        timeout_s=TIMEOUT_S,
    )
    rows = [{"pdb_id": pdb_id, "status": "local_pdb", "bytes": local_pdb(pdb_id).stat().st_size, "error": ""} for pdb_id in have_pdb]
    done = 0
    failed = 0
    downloaded = 0
    present = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, pdb_id): pdb_id for pdb_id in need}
        for future in as_completed(futures):
            row = future.result()
            row.setdefault("error", "")
            rows.append(row)
            done += 1
            if row["status"] == "downloaded":
                downloaded += 1
            elif row["status"] == "present":
                present += 1
            elif row["status"] == "failed":
                failed += 1
                log("download failed", pdb_id=row["pdb_id"], error=row["error"])
            if done % 100 == 0 or done == len(need):
                log(
                    "download progress",
                    done=done,
                    total=len(need),
                    downloaded=downloaded,
                    present=present,
                    failed=failed,
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(MANIFEST, sep="\t", index=False)
    counts = frame.status.value_counts().to_dict()
    log(
        "run finished",
        status="ok" if failed == 0 else "partial",
        utc_end=utc_now(),
        manifest=str(MANIFEST),
        counts=json.dumps(counts, sort_keys=True),
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("run failed", error=traceback.format_exc().replace("\n", " | "))
        raise
