"""Frozen ESM-2 650M, layer 33, one chain at a time. Resumable."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import esm

ROOT = Path(__file__).resolve().parents[1]
CHAINS = ROOT / "model_data" / "chains"
OUT = ROOT / "model_data" / "embeddings"
WINDOW = 2048
OVERLAP = 512


def windows(length: int):
    if length <= WINDOW:
        yield 0, length
        return
    step = WINDOW - OVERLAP
    start = 0
    while start < length:
        end = min(start + WINDOW, length)
        yield start, end
        if end == length:
            break
        start += step


def embed_chain(model, alphabet, batch, seq: str, device):
    L = len(seq)
    acc = torch.zeros(L, 1280, device=device)
    weight = torch.zeros(L, device=device)
    for start, end in windows(L):
        piece = seq[start:end]
        _, _, tokens = batch([("c", piece)])
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[33], return_contacts=False)
        rep = out["representations"][33][0, 1 : end - start + 1].float()
        core0, core1 = 0, end - start
        if start > 0:
            core0 = OVERLAP // 2
        if end < L:
            core1 = (end - start) - OVERLAP // 2
        acc[start + core0 : start + core1] += rep[core0:core1]
        weight[start + core0 : start + core1] += 1
    weight = weight.clamp(min=1).unsqueeze(1)
    return (acc / weight).to(torch.float16).cpu().numpy()


def read_seq(path: Path) -> str:
    return "".join(ln.strip() for ln in path.read_text().splitlines() if not ln.startswith(">"))


def embed_batch(model, converter, items, device):
    """items: list of (path, seq) with len(seq) <= WINDOW."""
    _, _, tokens = converter([(p.stem, seq) for p, seq in items])
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[33], return_contacts=False)
    reps = out["representations"][33].float()
    for i, (path, seq) in enumerate(items):
        arr = reps[i, 1 : len(seq) + 1].to(torch.float16).cpu().numpy()
        np.save(OUT / f"{path.stem}.npy", arr)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval().to(device=device, dtype=torch.bfloat16)
    converter = alphabet.get_batch_converter()
    chains = sorted(CHAINS.glob("*.fasta"))
    pending = []
    done = 0
    for path in chains:
        dest = OUT / f"{path.stem}.npy"
        if dest.exists() and dest.stat().st_size > 0:
            done += 1
            continue
        pending.append(path)
    short, long = [], []
    for path in pending:
        seq = read_seq(path)
        (short if len(seq) <= WINDOW else long).append((path, seq))
    short.sort(key=lambda item: len(item[1]))
    budget = 24576
    bucket, tokens = [], 0
    total = len(chains)
    for item in short:
        need = len(item[1]) + 2
        if bucket and tokens + need > budget:
            embed_batch(model, converter, bucket, device)
            done += len(bucket)
            bucket, tokens = [], 0
            if done % 200 == 0:
                print(f"{done}/{total}", flush=True)
        bucket.append(item)
        tokens += need
    if bucket:
        embed_batch(model, converter, bucket, device)
        done += len(bucket)
        print(f"{done}/{total}", flush=True)
    for path, seq in long:
        arr = embed_chain(model, alphabet, converter, seq, device)
        np.save(OUT / f"{path.stem}.npy", arr)
        done += 1
        print(f"{done}/{total} long {path.stem} {len(seq)}", flush=True)
    print(f"DONE {done}/{total}", flush=True)


if __name__ == "__main__":
    main()
