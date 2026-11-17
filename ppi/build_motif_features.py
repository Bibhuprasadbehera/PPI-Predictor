"""Per-residue motif channels from FIMO hits and the MEME PSSM."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "model_data"
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_I = {a: i for i, a in enumerate(AA)}


def parse_pssms(path: Path):
    text = path.read_text(errors="replace")
    motifs = []
    for block in re.split(r"(?=^MOTIF\s+)", text, flags=re.MULTILINE):
        if not block.startswith("MOTIF"):
            continue
        match = re.search(r"log-odds matrix:.*?w=\s*(\d+).*?\n(.*?)\n-{10,}", block, re.DOTALL)
        if not match:
            continue
        width = int(match.group(1))
        rows = []
        for line in match.group(2).strip().splitlines():
            if len(rows) >= width:
                break
            vals = line.split()
            if len(vals) < 20:
                continue
            rows.append([float(x) for x in vals[:20]])
        if len(rows) == width:
            motifs.append(np.asarray(rows, dtype=np.float32))
    return motifs


def pssm_track(seq: str, motifs) -> np.ndarray:
    L = len(seq)
    track = np.zeros(L, dtype=np.float32)
    if not motifs:
        return track
    idx = np.array([AA_I.get(c, 0) for c in seq.upper()], dtype=np.int32)
    known = np.array([c in AA_I for c in seq.upper()])
    for pssm in motifs:
        w = int(pssm.shape[0])
        if w > L:
            continue
        scores = np.zeros(L - w + 1, dtype=np.float32)
        ok = np.ones(L - w + 1, dtype=bool)
        for k in range(w):
            cols = idx[k : k + L - w + 1]
            scores += pssm[k, cols]
            ok &= known[k : k + L - w + 1]
        scores = np.where(ok, scores, -1e9)
        for k in range(w):
            track[k : k + scores.size] = np.maximum(track[k : k + scores.size], scores)
    return track


def fimo_tracks(seq: str, path: Path):
    L = len(seq)
    binary = np.zeros(L, dtype=np.float32)
    neglog = np.zeros(L, dtype=np.float32)
    if not path.exists() or path.stat().st_size == 0:
        return binary, neglog
    try:
        df = pd.read_csv(path, sep="\t", comment="#")
    except pd.errors.EmptyDataError:
        return binary, neglog
    if df.empty or "start" not in df.columns:
        return binary, neglog
    pcol = "p-value" if "p-value" in df.columns else "p_value"
    for start, stop, p in zip(df["start"], df["stop"], df[pcol]):
        try:
            start, stop, p = int(start), int(stop), float(p)
        except (TypeError, ValueError):
            continue
        if not (1 <= start <= stop <= L):
            continue
        binary[start - 1 : stop] = 1
        if p > 0:
            neglog[start - 1 : stop] = np.maximum(neglog[start - 1 : stop], -math.log10(p))
    return binary, neglog


def main():
    out = DATA / "motifs"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for fasta in sorted((DATA / "chains").glob("*.fasta")):
        dest = out / f"{fasta.stem}.npz"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        seq = "".join(ln.strip() for ln in fasta.read_text().splitlines() if not ln.startswith(">"))
        meme = DATA / "meme" / f"{fasta.stem}_meme.txt"
        if not meme.exists():
            continue
        motifs = parse_pssms(meme)
        binary, neglog = fimo_tracks(seq, DATA / "fimo" / f"{fasta.stem}_fimo.tsv")
        pssm = pssm_track(seq, motifs)
        feat = np.stack([binary, neglog, pssm], axis=1).astype(np.float16)
        np.savez_compressed(dest, motif=feat)
        n += 1
        if n % 500 == 0:
            print(n, flush=True)
    print("DONE", n, flush=True)


if __name__ == "__main__":
    main()
