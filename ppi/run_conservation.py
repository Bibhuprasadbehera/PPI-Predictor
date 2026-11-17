"""Index-aligned conservation profiles.

Each output row is one query residue. The query is aligned to its own HMM
together with the homologs, and columns where the query is a gap are dropped.
That keeps inserts in the query, so the profile length equals the chain length.
Entropy is stored as a fraction of log2(21), so every conservation column is in
[0, 1]. Chains with no homolog get a one-hot profile of the same length.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from Bio import AlignIO, SeqIO

ROOT = Path(__file__).resolve().parents[1]
DEEP = Path("/home/bibhu/Documents/DeepInteract/DEEPINTERACT")
HMMER = Path("/home/bibhu/Documents/DeepInteract/hmmer-3.4/bin")
HMMBUILD = str(HMMER / "hmmbuild")
HMMALIGN = str(HMMER / "hmmalign")
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}
ENTROPY_SCALE = math.log2(21)
VERSION = 2


def read_seq(path: Path) -> str:
    return "".join(
        line.strip() for line in path.read_text().splitlines() if line and not line.startswith(">")
    )


def find_hseq(chain_id: str) -> Path | None:
    local = ROOT / "model_data" / "hseq" / f"{chain_id}.fasta"
    if local.exists() and local.stat().st_size > 0:
        return local
    for folder in ("hseq_1", "hseq_2"):
        path = DEEP / "Positive_data" / folder / f"{chain_id}.fasta"
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def one_hot(seq: str) -> np.ndarray:
    mat = np.zeros((len(seq), 22), dtype=np.float32)
    for i, aa in enumerate(seq.upper()):
        mat[i, AA_INDEX.get(aa, 20)] = 1.0
    return mat


def query_row(alignment, query_id: str):
    # The query is written as q|<chain id> so a homolog with the same chain id
    # cannot be chosen in its place.
    wanted = f"q|{query_id}"
    for rec in alignment:
        bare = rec.id.split("|", 1)[-1]
        if rec.id == wanted or bare == wanted:
            return rec
    return None


def profile_from_stockholm(sto: Path, query_id: str, length: int) -> np.ndarray:
    alignment = AlignIO.read(sto, "stockholm")
    query = query_row(alignment, query_id)
    if query is None:
        raise RuntimeError(f"query row missing for {query_id}")
    qseq = str(query.seq)
    cols = [i for i, c in enumerate(qseq) if c not in "-."]
    if len(cols) != length:
        raise RuntimeError(f"{query_id} profile {len(cols)} != chain {length}")
    mat = np.zeros((length, 22), dtype=np.float32)
    seqs = [str(rec.seq) for rec in alignment]
    for out_i, col in enumerate(cols):
        counts = np.zeros(21, dtype=np.float64)
        for seq in seqs:
            aa = seq[col].upper()
            counts[AA_INDEX.get(aa, 20)] += 1
        total = counts.sum()
        if total == 0:
            continue
        freq = counts / total
        ent = float(-(freq[freq > 0] * np.log2(freq[freq > 0])).sum())
        mat[out_i, :21] = freq
        mat[out_i, 21] = ent / ENTROPY_SCALE
    return mat


def already_good(path: Path, length: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as blob:
            source = str(blob["source"])
            return (
                int(blob["v"]) == VERSION
                and int(blob["length"]) == length
                and len(blob["profile"]) == length
                and not source.startswith("onehot-fallback")
            )
    except Exception:
        return False


def save_profile(path: Path, mat: np.ndarray, length: int, source: str) -> None:
    if len(mat) != length:
        raise RuntimeError(f"refusing to write {path.name}: {len(mat)} != {length}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        profile=mat.astype(np.float16),
        length=np.int32(length),
        v=np.int32(VERSION),
        source=np.array(source),
    )


def one(chain_id: str) -> str:
    out = ROOT / "model_data" / "conservation" / f"{chain_id}.npz"
    query_path = ROOT / "model_data" / "chains" / f"{chain_id}.fasta"
    if not query_path.exists():
        return "no-query"
    seq = read_seq(query_path)
    length = len(seq)
    if already_good(out, length):
        return "skip"
    hseq = find_hseq(chain_id)
    try:
        if hseq is None:
            save_profile(out, one_hot(seq), length, "onehot")
            return "onehot"
        records = list(SeqIO.parse(hseq, "fasta"))
        homologs = [rec for rec in records if str(rec.seq).replace("-", "").upper() != seq]
        if not homologs:
            save_profile(out, one_hot(seq), length, "onehot")
            return "onehot"
        with tempfile.TemporaryDirectory(prefix="cons_") as td:
            combined = Path(td) / "all.fasta"
            with combined.open("w") as handle:
                handle.write(f">q|{chain_id}\n{seq}\n")
                SeqIO.write(homologs, handle, "fasta")
            hmm = Path(td) / "q.hmm"
            sto = Path(td) / "aln.sto"
            subprocess.run(
                [HMMBUILD, "--amino", str(hmm), str(query_path)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [HMMALIGN, "-o", str(sto), str(hmm), str(combined)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            mat = profile_from_stockholm(sto, chain_id, length)
        save_profile(out, mat, length, "msa")
        return "ok"
    except Exception:
        save_profile(out, one_hot(seq), length, "onehot-fallback")
        err = ROOT / "model_data" / "conservation_fallback.txt"
        with err.open("a") as handle:
            handle.write(f"{chain_id}\t{traceback.format_exc().strip().splitlines()[-1]}\n")
        return "onehot-fallback"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", nargs="*", default=[])
    args = parser.parse_args()
    if args.ids:
        chains = args.ids
    else:
        chains = sorted(p.stem for p in (ROOT / "model_data" / "chains").glob("*.fasta"))
    if args.limit:
        chains = chains[: args.limit]
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, cid): cid for cid in chains}
        done = 0
        for fut in as_completed(futures):
            status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if done % 100 == 0 or done == len(chains):
                print(f"{done}/{len(chains)} {counts}", flush=True)
    print("FINAL", counts, flush=True)


if __name__ == "__main__":
    main()
