"""Resumable MEME then FIMO for each chain. Positives first.

The MEME binary was built with prefix /home/bioinfo/meme. A patched copy
reads /home/bibhu/meme58, which points at the MEME etc/ data files.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "ppi" / "bin"
MEME = BIN / "meme"
FIMO = BIN / "fimo"
PLIB = Path("/home/bibhu/meme58/prior30.plib")
HSEQ = ROOT / "model_data" / "hseq"
CHAINS = ROOT / "model_data" / "chains"
MEME_OUT = ROOT / "model_data" / "meme"
FIMO_OUT = ROOT / "model_data" / "fimo"
LOG = ROOT / "model_data" / "motif_progress.log"


def ensure_tools():
    old = b"/home/bioinfo/meme/share/meme-5.5.8"
    new = b"/home/bibhu/meme58"
    pad = new + b"\0" * (len(old) - len(new))
    BIN.mkdir(parents=True, exist_ok=True)
    src = Path("/home/bibhu/Documents/DeepInteract/meme-5.5.8")
    for name in ("meme", "fimo"):
        blob = (src / "src" / name).read_bytes().replace(old, pad, 1)
        dest = BIN / name
        if not dest.exists() or dest.read_bytes() != blob:
            dest.write_bytes(blob)
            dest.chmod(0o755)
    share = Path("/home/bibhu/meme58")
    if not share.exists():
        share.symlink_to(src / "etc")


def clean_fasta(src: Path, dest: Path):
    header = None
    chunks = []
    records = []

    def flush():
        if header is None:
            return
        seq = "".join("X" if c in "UOBZJ" else c for c in "".join(chunks))
        if len(seq) >= 8:
            records.append(f">{header}\n{seq}\n")

    for line in src.read_text().splitlines():
        if line.startswith(">"):
            flush()
            header = line[1:]
            chunks = []
        else:
            chunks.append(line.strip())
    flush()
    if not records:
        raise RuntimeError("no sequence longer than 7")
    dest.write_text("".join(records))
    return len(records)


def have_meme(chain_id: str) -> bool:
    path = MEME_OUT / f"{chain_id}_meme.txt"
    return path.exists() and b"MOTIF" in path.read_bytes()


def run_meme(chain_id: str, fasta: Path, work: Path) -> bool:
    if have_meme(chain_id):
        return True
    clean = work / "clean.fasta"
    n_kept = clean_fasta(fasta, clean)
    if n_kept < 2:
        return False
    passes = (
        ["-mod", "oops", "-nmotifs", "3", "-minw", "6", "-maxw", "50"],
        ["-mod", "zoops", "-nmotifs", "10", "-minw", "5", "-maxw", "50"],
    )
    for params in passes:
        out = work / "meme_out"
        if out.exists():
            shutil.rmtree(out)
        subprocess.run(
            [str(MEME), str(clean), "-protein", "-oc", str(out), "-nostatus",
             "-time", "14400", "-maxsize", "20000000", "-objfun", "classic",
             "-markov_order", "0", "-plib", str(PLIB), *params],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        produced = out / "meme.txt"
        n = produced.read_bytes().count(b"\nMOTIF") + produced.read_bytes().count(b"MOTIF ") if produced.exists() else 0
        if produced.exists() and b"MOTIF" in produced.read_bytes():
            MEME_OUT.mkdir(parents=True, exist_ok=True)
            shutil.copy(produced, MEME_OUT / f"{chain_id}_meme.txt")
            if n >= 2 or params[1] == "zoops":
                return True
    return have_meme(chain_id)


def run_fimo(chain_id: str, work: Path) -> str:
    dest = FIMO_OUT / f"{chain_id}_fimo.tsv"
    if dest.exists() and dest.stat().st_size > 0:
        return "fimo-skip"
    meme = MEME_OUT / f"{chain_id}_meme.txt"
    query = CHAINS / f"{chain_id}.fasta"
    if not meme.exists() or not query.exists():
        return "fimo-missing"
    out = work / "fimo_out"
    if out.exists():
        shutil.rmtree(out)
    subprocess.run(
        [str(FIMO), "--oc", str(out), "--verbosity", "1", "--thresh", "1.0E-4", str(meme), str(query)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    produced = out / "fimo.tsv"
    if not produced.exists():
        return "fimo-empty"
    FIMO_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(produced, dest)
    return "fimo-ok"


def one(chain_id: str) -> str:
    fasta = HSEQ / f"{chain_id}.fasta"
    if not fasta.exists():
        fasta = CHAINS / f"{chain_id}.fasta"
    if not fasta.exists():
        return f"{chain_id} no-hseq"
    work = Path(f"/tmp/motif_{chain_id}")
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        ok = run_meme(chain_id, fasta, work)
        fimo = run_fimo(chain_id, work) if ok else "no-meme"
        return f"{chain_id} meme={'ok' if ok else 'fail'} {fimo}"
    except Exception as exc:
        return f"{chain_id} error {type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def ordered_chains():
    df = pd.read_csv(ROOT / "model_data" / "pairs.tsv", sep="\t")
    pos = set(df.loc[df.label == 1, "chain_id_1"]) | set(df.loc[df.label == 1, "chain_id_2"])
    rest = sorted(p.stem for p in HSEQ.glob("*.fasta"))
    need = set(df.chain_id_1) | set(df.chain_id_2)
    rest = sorted(set(rest) | need)
    first = sorted(c for c in rest if c in pos)
    second = [c for c in rest if c not in pos]
    return first + second


def nseq(chain_id: str) -> int:
    path = HSEQ / f"{chain_id}.fasta"
    if not path.exists():
        return 1
    n = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=120)
    args = parser.parse_args()
    ensure_tools()
    todo = [c for c in ordered_chains() if not (FIMO_OUT / f"{c}_fimo.tsv").exists()]
    small, large = [], []
    for cid in todo:
        (large if nseq(cid) > 250 else small).append(cid)
    # Small alignments finish in seconds. Keep most cores on those so the
    # queue does not stall behind a few multi-thousand-sequence MEME jobs.
    large_workers = min(24, len(large)) or 1
    small_workers = max(1, min(len(small), args.workers - large_workers))
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as log:
        log.write(
            f"START remaining {len(todo)} small {len(small)}/{small_workers} "
            f"large {len(large)}/{large_workers}\n"
        )
        log.flush()
        done = 0
        pools = []
        if small:
            pools.append(ProcessPoolExecutor(max_workers=small_workers))
        if large:
            pools.append(ProcessPoolExecutor(max_workers=large_workers))
        try:
            futs = {}
            if small:
                futs.update({pools[0].submit(one, cid): cid for cid in small})
            if large:
                futs.update({pools[-1].submit(one, cid): cid for cid in large})
            for fut in as_completed(futs):
                line = fut.result()
                done += 1
                log.write(f"{done}/{len(todo)} {line}\n")
                log.flush()
                if done % 50 == 0:
                    print(f"{done}/{len(todo)}", flush=True)
        finally:
            for pool in pools:
                pool.shutdown(wait=True, cancel_futures=False)
        log.write("DONE\n")
    print("DONE", len(todo), flush=True)


if __name__ == "__main__":
    main()
