"""Motif-free baseline: frozen ESM-2 embeddings plus conservation, trained on GPU.

This is iteration 1.0. Motif channels are absent. The linker is a symmetrized
bilinear score with a top-32 log-sum-exp readout.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from features import load_raw, load_scaler, transform

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "model_data"


class Pairs(Dataset):
    def __init__(self, frame: pd.DataFrame, scaler=None):
        self.rows = frame.reset_index(drop=True)
        self.scaler = load_scaler() if scaler is None else scaler

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        x1 = transform(*load_raw(row.chain_id_1), self.scaler)
        x2 = transform(*load_raw(row.chain_id_2), self.scaler)
        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.tensor(float(row.label), dtype=torch.float32),
        )


def collate(batch):
    x1, x2, y = zip(*batch)
    n1 = max(t.size(0) for t in x1)
    n2 = max(t.size(0) for t in x2)
    d = x1[0].size(1)
    b = len(batch)
    a = torch.zeros(b, n1, d)
    c = torch.zeros(b, n2, d)
    m1 = torch.zeros(b, n1, dtype=torch.bool)
    m2 = torch.zeros(b, n2, dtype=torch.bool)
    for i, (u, v) in enumerate(zip(x1, x2)):
        a[i, : u.size(0)] = u.float()
        c[i, : v.size(0)] = v.float()
        m1[i, : u.size(0)] = True
        m2[i, : v.size(0)] = True
    return a, c, m1, m2, torch.stack(y)


class CrossBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)

    def forward(self, query, context, context_pad):
        # PyTorch treats True as "ignore this key".
        mix, _ = self.attn(query, context, context, key_padding_mask=~context_pad)
        hidden = self.n1(query + mix)
        return self.n2(hidden + self.ff(hidden))


class Linker(nn.Module):
    def __init__(self, indim=1280 + 22 + 3, hidden=512, heads=8, layers=3, rank=64, k=32):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(indim), nn.Linear(indim, hidden), nn.GELU())
        self.blocks = nn.ModuleList(CrossBlock(hidden, heads) for _ in range(layers))
        self.w1 = nn.Linear(hidden, rank, bias=False)
        self.w2 = nn.Linear(hidden, rank, bias=False)
        self.k = k

    def forward(self, x1, x2, m1, m2):
        z1, z2 = self.proj(x1), self.proj(x2)
        for block in self.blocks:
            nxt1 = block(z1, z2, m2)
            nxt2 = block(z2, z1, m1)
            z1, z2 = nxt1, nxt2
        a, b = self.w1(z1), self.w2(z2)
        a2, b2 = self.w2(z1), self.w1(z2)
        scores = 0.5 * (
            torch.einsum("bir,bjr->bij", a, b) + torch.einsum("bir,bjr->bij", a2, b2)
        )
        valid = m1.unsqueeze(2) & m2.unsqueeze(1)
        scores = scores.masked_fill(~valid, -1e4)
        flat = scores.flatten(1)
        k = min(self.k, flat.size(1))
        top = torch.topk(flat, k, dim=1).values
        return torch.logsumexp(top, dim=1) - math.log(k)


def run_epoch(model, loader, opt, loss_fn, device, train):
    model.train(train)
    total, n = 0.0, 0
    for x1, x2, m1, m2, y in loader:
        x1, x2 = x1.to(device), x2.to(device)
        m1, m2, y = m1.to(device), m2.to(device), y.to(device)
        if train:
            opt.zero_grad(set_to_none=True)
        logit = model(x1, x2, m1, m2)
        loss = loss_fn(logit, y)
        if train:
            loss.backward()
            opt.step()
        total += float(loss) * len(y)
        n += len(y)
    return total / max(n, 1)


def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x1, x2, m1, m2, y in loader:
            logit = model(x1.to(device), x2.to(device), m1.to(device), m2.to(device))
            logits.append(logit.float().cpu())
            labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def report(name, logits, labels):
    prob = 1 / (1 + np.exp(-logits))
    auroc = roc_auc_score(labels, prob)
    auprc = average_precision_score(labels, prob)
    line = f"{name} n={len(labels)} pos={int(labels.sum())} AUROC {auroc:.4f} AUPRC {auprc:.4f}"
    print(line, flush=True)
    return line


def main():
    device = torch.device("cuda")
    df = pd.read_csv(DATA / "pairs.tsv", sep="\t")
    splits = {}
    for name in ("train", "val"):
        ids = set((DATA / "splits" / f"{name}.txt").read_text().split())
        splits[name] = df[df.pair_id.isin(ids)]
    def clear(frame):
        if "seq_ambiguous" not in frame.columns:
            return frame
        return frame[~frame.seq_ambiguous.astype(bool)].copy()

    splits = {name: clear(frame) for name, frame in splits.items()}
    train_ds = Pairs(splits["train"])
    val_ds = Pairs(splits["val"])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=12, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate, pin_memory=True)
    model = Linker().to(device)
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    if args.eval:
        model.load_state_dict(torch.load(DATA / "motif_linker.pt", map_location=device))
        test_ids = set((DATA / "splits" / "test.txt").read_text().split())
        test_ds = Pairs(clear(df[df.pair_id.isin(test_ids)]))
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate, pin_memory=True)
        lines = [
            report("val", *predict(model, val_loader, device)),
            report("test", *predict(model, test_loader, device)),
        ]
        (DATA / "motif_metrics.txt").write_text("\n".join(lines) + "\n")
        return
    n_pos = max(int((splits["train"].label == 1).sum()), 1)
    n_neg = int((splits["train"].label == 0).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    log = DATA / "train_motif.log"
    best_score, best_epoch, wait = -1.0, 0, 0
    with log.open("a") as handle:
        handle.write(f"train {len(train_ds)} val {len(val_ds)} pos_weight {n_neg / n_pos:.3f}\n")
        for epoch in range(1, 9):
            tr = run_epoch(model, train_loader, opt, loss_fn, device, True)
            logits, labels = predict(model, val_loader, device)
            score = average_precision_score(labels, 1 / (1 + np.exp(-logits)))
            line = f"epoch {epoch} train_loss {tr:.4f} val_auprc {score:.4f}\n"
            handle.write(line)
            handle.flush()
            print(line, flush=True)
            if score > best_score:
                best_score, best_epoch, wait = score, epoch, 0
                torch.save(model.state_dict(), DATA / "motif_linker_best.pt")
            else:
                wait += 1
                if wait >= 2:
                    handle.write(f"stop at epoch {epoch} best {best_epoch} val_auprc {best_score:.4f}\n")
                    break
        handle.write("saved motif_linker_best.pt\n")


if __name__ == "__main__":
    main()
