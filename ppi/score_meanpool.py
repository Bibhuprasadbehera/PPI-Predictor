"""Score the scaled mean-pool checkpoint on the current test split.

Writes one row per test pair. The plot reads this file.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_baseline import DATA, Pairs, collate
from train_meanpool import MeanPair

OUT = Path(__file__).resolve().parents[1] / "figures" / "audit" / "meanpool_scaled_test.tsv"


def main():
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA / "pairs.tsv", sep="\t")
    ids = set((DATA / "splits" / "test.txt").read_text().split())
    test = df[df.pair_id.isin(ids)].reset_index(drop=True)
    model = MeanPair().to(device)
    model.load_state_dict(torch.load(DATA / "meanpool_scaled.pt", map_location=device))
    model.eval()
    loader = DataLoader(
        Pairs(test), batch_size=16, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True
    )
    logits = []
    labels = []
    with torch.no_grad():
        for x1, x2, m1, m2, y in loader:
            logit = model(x1.to(device), x2.to(device), m1.to(device), m2.to(device))
            logits.append(logit.float().cpu())
            labels.append(y)
    logit = torch.cat(logits).numpy()
    label = torch.cat(labels).numpy()
    if len(logit) != len(test) or not (label == test.label.to_numpy()).all():
        raise SystemExit(f"score rows {len(logit)} do not match the test table {len(test)}")
    out = test[["pair_id", "label", "seq_ambiguous"]].copy()
    out["logit"] = logit
    out["prob"] = 1 / (1 + torch.tensor(logit).mul(-1).exp().numpy())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep="\t", index=False)
    print(f"wrote {OUT} n={len(out)} device={device} duration_s={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
