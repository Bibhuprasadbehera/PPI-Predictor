"""Load index-aligned chain features and apply a train-only scaler.

Embeddings are z-scored per dimension. Conservation stays in [0, 1].
The motif p-value channel is capped, and the PSSM channel is log1p then z-scored.
Statistics come from train chains only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "model_data"
EMB = DATA / "embeddings"
CONS = DATA / "conservation"
MOTIF = DATA / "motifs"
SCALER_PATH = DATA / "feature_scaler.npz"
P_CAP = 10.0


def load_scaler(path: Path = SCALER_PATH) -> dict:
    with np.load(path) as blob:
        return {
            "esm_mean": blob["esm_mean"].astype(np.float32),
            "esm_std": blob["esm_std"].astype(np.float32),
            "pssm_mean": float(blob["pssm_mean"]),
            "pssm_std": float(blob["pssm_std"]),
            "p_cap": float(blob["p_cap"]),
        }


def load_raw(chain_id: str):
    emb_path = EMB / f"{chain_id}.npy"
    cons_path = CONS / f"{chain_id}.npz"
    motif_path = MOTIF / f"{chain_id}.npz"
    h = np.load(emb_path)
    with np.load(cons_path) as blob:
        c = blob["profile"]
    with np.load(motif_path) as blob:
        m = blob["motif"]
    if not (len(h) == len(c) == len(m)):
        raise RuntimeError(
            f"{chain_id} feature lengths differ: embedding {len(h)}, "
            f"conservation {len(c)}, motif {len(m)}"
        )
    return h, c, m


def transform(h: np.ndarray, c: np.ndarray, m: np.ndarray, scaler: dict) -> np.ndarray:
    emb = (h.astype(np.float32) - scaler["esm_mean"]) / scaler["esm_std"]
    cons = c.astype(np.float32)
    motif = m.astype(np.float32).copy()
    cap = scaler["p_cap"]
    motif[:, 1] = np.clip(motif[:, 1], 0, cap) / cap
    pssm = np.log1p(np.clip(motif[:, 2], 0, None))
    motif[:, 2] = (pssm - scaler["pssm_mean"]) / scaler["pssm_std"]
    return np.concatenate([emb, cons, motif], axis=1)
