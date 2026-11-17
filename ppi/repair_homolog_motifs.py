"""Rebuild homolog FASTA, then MEME, then FIMO for chains that had no homolog set.

MEME is not run on a single chain. If PHMMER still finds fewer than two
sequences, the old single-sequence motif files are removed and the motif
channels are set to zero.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

from build_motif_features import fimo_tracks, parse_pssms, pssm_track
from run_motifs import MEME, PLIB, clean_fasta

ROOT = Path(__file__).resolve().parents[1]
DEEP = Path("/home/bibhu/Documents/DeepInteract/DEEPINTERACT")
DATA = ROOT / "model_data"
PHMMER = "/home/bibhu/Documents/DeepInteract/hmmer-3.4/bin/phmmer"
DB = DEEP / "db" / "uniprot_sprot.fasta"


def needs_repair(chain_id: str) -> bool:
    path = DATA / "hseq" / f"{chain_id}.fasta"
    if not path.exists():
        return True
    n = sum(1 for line in path.read_text().splitlines() if line.startswith(">"))
    return n < 2


def phmmer_hseq(chain_id: str, index) -> int:
    query = DATA / "chains" / f"{chain_id}.fasta"
    out = DATA / "hseq" / f"{chain_id}.fasta"
    with tempfile.TemporaryDirectory(prefix="ph_") as td:
        tbl = Path(td) / "hits.txt"
        subprocess.run(
            [PHMMER, "--cpu", "1", "--tblout", str(tbl), "--noali", "-E", "0.01", str(query), str(DB)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        targets = []
        lines = [ln for ln in tbl.read_text().splitlines() if ln and not ln.startswith("#")]
        for ln in lines:
            targets.append(ln.split()[0])
        targets = list(dict.fromkeys(targets))
    query_rec = next(SeqIO.parse(query, "fasta"))
    records = [query_rec]
    for target in targets:
        if target in index and target != query_rec.id:
            records.append(index[target])
    SeqIO.write(records, out, "fasta")
    return len(records)


def meme_then_fimo(chain_id: str) -> str:
    hseq = DATA / "hseq" / f"{chain_id}.fasta"
    meme_path = DATA / "meme" / f"{chain_id}_meme.txt"
    fimo_path = DATA / "fimo" / f"{chain_id}_fimo.tsv"
    motif_path = DATA / "motifs" / f"{chain_id}.npz"
    n = sum(1 for line in hseq.read_text().splitlines() if line.startswith(">"))
    if n < 2:
        meme_path.unlink(missing_ok=True)
        fimo_path.unlink(missing_ok=True)
        seq = "".join(ln.strip() for ln in (DATA / "chains" / f"{chain_id}.fasta").read_text().splitlines() if not ln.startswith(">"))
        feat = np.zeros((len(seq), 3), dtype=np.float16)
        np.savez_compressed(motif_path, motif=feat)
        return f"{chain_id} no-homolog n={n}"
    meme_path.unlink(missing_ok=True)
    fimo_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="meme_") as td:
        work = Path(td)
        clean = work / "clean.fasta"
        kept = clean_fasta(hseq, clean)
        if kept < 2:
            seq = "".join(ln.strip() for ln in (DATA / "chains" / f"{chain_id}.fasta").read_text().splitlines() if not ln.startswith(">"))
            np.savez_compressed(motif_path, motif=np.zeros((len(seq), 3), dtype=np.float16))
            return f"{chain_id} homologs-too-short n={n}"
        produced = None
        for params in (
            ["-mod", "oops", "-nmotifs", "3", "-minw", "6", "-maxw", "50"],
            ["-mod", "zoops", "-nmotifs", "10", "-minw", "5", "-maxw", "50"],
        ):
            out = work / "meme_out"
            if out.exists():
                shutil.rmtree(out)
            subprocess.run(
                [str(MEME), str(clean), "-protein", "-oc", str(out), "-nostatus", "-time", "14400",
                 "-maxsize", "20000000", "-objfun", "classic", "-markov_order", "0", "-plib", str(PLIB), *params],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            candidate = out / "meme.txt"
            if candidate.exists() and b"MOTIF" in candidate.read_bytes():
                shutil.copy(candidate, meme_path)
                produced = candidate
                if candidate.read_bytes().count(b"MOTIF") >= 2 or params[1] == "zoops":
                    break
        if produced is None:
            return f"{chain_id} meme-fail n={n}"
        fimo_dir = work / "fimo_out"
        subprocess.run(
            [str(ROOT / "ppi" / "bin" / "fimo"), "--oc", str(fimo_dir), "--verbosity", "1",
             "--thresh", "1.0E-4", str(meme_path), str(DATA / "chains" / f"{chain_id}.fasta")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if (fimo_dir / "fimo.tsv").exists():
            shutil.copy(fimo_dir / "fimo.tsv", fimo_path)
    seq = "".join(ln.strip() for ln in (DATA / "chains" / f"{chain_id}.fasta").read_text().splitlines() if not ln.startswith(">"))
    binary, neglog = fimo_tracks(seq, fimo_path)
    pssm = pssm_track(seq, parse_pssms(meme_path))
    np.savez_compressed(motif_path, motif=np.stack([binary, neglog, pssm], axis=1).astype(np.float16))
    return f"{chain_id} rebuilt n={n}"


def meme_then_fimo_short(chain_id: str) -> str:
    return f"{chain_id} still-short"


def main():
    chains = []
    for path in (DATA / "chains").glob("*.fasta"):
        if needs_repair(path.stem):
            chains.append(path.stem)
    print(f"repair {len(chains)}", flush=True)
    index = SeqIO.index(str(DB), "fasta")
    found = {}
    for i, cid in enumerate(chains, 1):
        found[cid] = phmmer_hseq(cid, index)
        if i % 25 == 0:
            print(f"phmmer {i}/{len(chains)}", flush=True)
    index.close()
    print("phmmer done", flush=True)
    with ProcessPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(meme_then_fimo, cid) for cid in chains]
        for i, fut in enumerate(as_completed(futs), 1):
            print(f"{i}/{len(chains)} {fut.result()}", flush=True)


if __name__ == "__main__":
    main()
