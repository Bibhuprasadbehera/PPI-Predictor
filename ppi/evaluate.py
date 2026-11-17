"""Score a saved linker: AUROC, AUPRC, calibration, UniProt-cluster intervals."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from train_baseline import Linker, Pairs, collate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "model_data"


def find(parent, x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent, a, b):
    ra, rb = find(parent, a), find(parent, b)
    if ra != rb:
        parent[rb] = ra


def components(frame: pd.DataFrame):
    parent = {}
    for row in frame.itertuples(index=False):
        union(parent, row.uniprot_id_1, row.uniprot_id_2)
    for pdb, group in frame.groupby("pdb_id"):
        ids = list(set(group.uniprot_id_1) | set(group.uniprot_id_2))
        for other in ids[1:]:
            union(parent, ids[0], other)
    return np.array([find(parent, u) for u in frame.uniprot_id_1])


def predict(model, frame, device):
    loader = DataLoader(
        Pairs(frame), batch_size=8, shuffle=False, num_workers=8, collate_fn=collate, pin_memory=True
    )
    logits = []
    model.eval()
    with torch.no_grad():
        for x1, x2, m1, m2, _y in loader:
            logit = model(x1.to(device), x2.to(device), m1.to(device), m2.to(device))
            logits.append(logit.float().cpu())
    return torch.cat(logits).numpy()


def sigmoid(logits, temperature=1.0):
    return 1 / (1 + np.exp(-logits / temperature))


def ece(prob, labels, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        if i < bins - 1:
            mask = (prob >= edges[i]) & (prob < edges[i + 1])
        else:
            mask = (prob >= edges[i]) & (prob <= edges[i + 1])
        if mask.sum() == 0:
            continue
        total += mask.mean() * abs(labels[mask].mean() - prob[mask].mean())
    return float(total)


def fit_temperature(logits, labels):
    best_t, best_nll = 1.0, None
    for temp in np.linspace(0.5, 5.0, 46):
        prob = np.clip(sigmoid(logits, temp), 1e-6, 1 - 1e-6)
        nll = -(labels * np.log(prob) + (1 - labels) * np.log(1 - prob)).mean()
        if best_nll is None or nll < best_nll:
            best_t, best_nll = float(temp), float(nll)
    return best_t


def bootstrap(prob, labels, groups, draws=1000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    scores = []
    for _ in range(draws):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in chosen])
        y = labels[idx]
        if y.min() == y.max():
            continue
        scores.append(roc_auc_score(y, prob[idx]))
    arr = np.asarray(scores)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), len(arr)


def metrics(name, logits, labels, temperature=1.0):
    prob = sigmoid(logits, temperature)
    return (
        f"{name} n={len(labels)} pos={int(labels.sum())} "
        f"AUROC {roc_auc_score(labels, prob):.4f} "
        f"AUPRC {average_precision_score(labels, prob):.4f} "
        f"ECE {ece(prob, labels):.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(DATA / "motif_linker_best.pt"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA / "pairs.tsv", sep="\t")
    groups_all = components(df)
    df = df.assign(component=groups_all)
    model = Linker().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    lines = [f"checkpoint {args.ckpt}", "features scaled; unambiguous pairs are the primary metric"]

    def primary(frame):
        if "seq_ambiguous" not in frame.columns:
            return frame.reset_index(drop=True)
        return frame[~frame.seq_ambiguous.astype(bool)].reset_index(drop=True)

    val = primary(df[df.split == "val"])
    test = primary(df[df.split == "test"])
    test_all = df[df.split == "test"].reset_index(drop=True)
    val_logits = predict(model, val, device)
    test_logits = predict(model, test, device)
    temp = fit_temperature(val_logits, val.label.to_numpy())
    lines.append(f"temperature_fit_on_val {temp:.2f}")
    for split_name, frame, logits in (
        ("val", val, val_logits),
        ("test", test, test_logits),
    ):
        y = frame.label.to_numpy()
        lines.append(metrics(f"{split_name} raw", logits, y))
        lines.append(metrics(f"{split_name} calibrated", logits, y, temp))
        if split_name == "test":
            lo, hi, n = bootstrap(sigmoid(logits, temp), y, frame.component.to_numpy())
            lines.append(f"test calibrated AUROC 95% cluster interval {lo:.4f} {hi:.4f} draws {n}")
            ident = frame.seq_identical_to_a_positive.to_numpy().astype(bool)
            if ident.any():
                lines.append(
                    f"test identical-sequence negatives n={int(ident.sum())} "
                    f"mean_prob {sigmoid(logits, temp)[ident].mean():.4f}"
                )
            other = ~ident & (y == 0)
            if other.any():
                lines.append(
                    f"test other negatives n={int(other.sum())} "
                    f"mean_prob {sigmoid(logits, temp)[other].mean():.4f}"
                )
    full_logits = predict(model, test_all, device)
    lines.append(metrics("test_full raw", full_logits, test_all.label.to_numpy()))
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    out = DATA / "motif_metrics.txt"
    previous = out.read_text() if out.exists() else ""
    out.write_text(previous + "\n" + text)


if __name__ == "__main__":
    main()
