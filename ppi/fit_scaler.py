"""Fit per-dimension embedding stats and the PSSM log1p stats on train chains."""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import DATA, EMB, MOTIF, P_CAP, SCALER_PATH


def main():
    df = pd.read_csv(DATA / "pairs.tsv", sep="\t")
    train = df[df.split == "train"]
    if "seq_ambiguous" in train.columns:
        train = train[~train.seq_ambiguous.astype(bool)]
    chains = sorted(set(train.chain_id_1) | set(train.chain_id_2))
    esm_sum = np.zeros(1280, dtype=np.float64)
    esm_sq = np.zeros(1280, dtype=np.float64)
    pssm_sum = 0.0
    pssm_sq = 0.0
    n = 0
    for i, cid in enumerate(chains, start=1):
        h = np.load(EMB / f"{cid}.npy")
        m = np.load(MOTIF / f"{cid}.npz")["motif"]
        x = h.astype(np.float64)
        esm_sum += x.sum(axis=0)
        esm_sq += np.square(x).sum(axis=0)
        pssm = np.log1p(np.clip(m[:, 2].astype(np.float64), 0, None))
        pssm_sum += float(pssm.sum())
        pssm_sq += float(np.square(pssm).sum())
        n += len(h)
        if i % 500 == 0 or i == len(chains):
            print(f"{i}/{len(chains)} residues {n}", flush=True)
    if n == 0:
        raise SystemExit("no train residues")
    mean = esm_sum / n
    var = esm_sq / n - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    p_mean = pssm_sum / n
    p_var = pssm_sq / n - p_mean * p_mean
    p_std = float(np.sqrt(max(p_var, 1e-12)))
    np.savez_compressed(
        SCALER_PATH,
        esm_mean=mean.astype(np.float32),
        esm_std=std.astype(np.float32),
        pssm_mean=np.float32(p_mean),
        pssm_std=np.float32(p_std),
        p_cap=np.float32(P_CAP),
        n_residues=np.int64(n),
        n_chains=np.int32(len(chains)),
    )
    print(
        f"wrote {SCALER_PATH.name} chains {len(chains)} residues {n} "
        f"esm_std median {np.median(std):.4f} pssm mean {p_mean:.4f} std {p_std:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
