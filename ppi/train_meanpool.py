"""One-change follow-up: mean-pool the same features, then a small pair MLP.

The cross-attention linker scored at chance. This asks whether the frozen
embeddings carry any pair signal at all.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from train_baseline import DATA, Pairs, collate


class MeanPair(nn.Module):
    def __init__(self, indim=1280 + 22 + 3, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(indim * 3),
            nn.Linear(indim * 3, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x1, x2, m1, m2):
        def pool(x, mask):
            weight = mask.unsqueeze(-1).float()
            return (x * weight).sum(1) / weight.sum(1).clamp(min=1)
        a, b = pool(x1, m1), pool(x2, m2)
        pair = torch.cat([a + b, (a - b).abs(), a * b], dim=1)
        return self.net(pair).squeeze(1)


def epoch(model, loader, opt, loss_fn, device, train):
    model.train(train)
    total, n = 0.0, 0
    for x1, x2, m1, m2, y in loader:
        x1, x2 = x1.to(device), x2.to(device)
        m1, m2, y = m1.to(device), m2.to(device), y.to(device)
        if train:
            opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x1, x2, m1, m2), y)
        if train:
            loss.backward()
            opt.step()
        total += float(loss) * len(y)
        n += len(y)
    return total / max(n, 1)


def scores(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x1, x2, m1, m2, y in loader:
            logit = model(x1.to(device), x2.to(device), m1.to(device), m2.to(device))
            logits.append(logit.float().cpu())
            labels.append(y)
    logit = torch.cat(logits).numpy()
    label = torch.cat(labels).numpy()
    prob = 1 / (1 + np.exp(-logit))
    return roc_auc_score(label, prob), average_precision_score(label, prob)


def subset_of(df, name):
    ids = set((DATA / "splits" / f"{name}.txt").read_text().split())
    return df[df.pair_id.isin(ids)].copy()


def clear_pairs(frame):
    if "seq_ambiguous" not in frame.columns:
        return frame
    return frame[~frame.seq_ambiguous.astype(bool)].copy()


def main():
    device = torch.device("cuda")
    import pandas as pd
    df = pd.read_csv(DATA / "pairs.tsv", sep="\t")
    train_df = clear_pairs(subset_of(df, "train"))
    val_df = clear_pairs(subset_of(df, "val"))
    test_all = subset_of(df, "test")
    test_df = clear_pairs(test_all)
    test_amb = test_all[test_all.seq_ambiguous.astype(bool)] if "seq_ambiguous" in test_all.columns else test_all.iloc[0:0]
    train_loader = DataLoader(Pairs(train_df), batch_size=16, shuffle=True, num_workers=8, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(Pairs(val_df), batch_size=16, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(Pairs(test_df), batch_size=16, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)
    model = MeanPair().to(device)
    n_pos = max(int((train_df.label == 1).sum()), 1)
    n_neg = int((train_df.label == 0).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    log = DATA / "train_meanpool_scaled.log"
    best, wait = -1.0, 0
    with log.open("w") as handle:
        handle.write(
            f"unambiguous train {len(train_df)} pos_rate {train_df.label.mean():.4f} "
            f"val {len(val_df)} pos_rate {val_df.label.mean():.4f} "
            f"test {len(test_df)} pos_rate {test_df.label.mean():.4f} "
            f"pos_weight {n_neg / n_pos:.3f}\n"
        )
        for epoch_i in range(1, 9):
            tr = epoch(model, train_loader, opt, loss_fn, device, True)
            auroc, auprc = scores(model, val_loader, device)
            line = f"epoch {epoch_i} train_loss {tr:.4f} val_auroc {auroc:.4f} val_auprc {auprc:.4f}\n"
            handle.write(line)
            handle.flush()
            print(line, flush=True)
            if auprc > best:
                best, wait = auprc, 0
                torch.save(model.state_dict(), DATA / "meanpool_scaled.pt")
            else:
                wait += 1
                if wait >= 2:
                    handle.write(f"stop best_val_auprc {best:.4f}\n")
                    break
        model.load_state_dict(torch.load(DATA / "meanpool_scaled.pt", map_location=device))
        auroc, auprc = scores(model, test_loader, device)
        line = f"test_unambiguous AUROC {auroc:.4f} AUPRC {auprc:.4f} n={len(test_df)}\n"
        handle.write(line)
        print(line, flush=True)
        full_loader = DataLoader(Pairs(test_all), batch_size=16, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)
        auroc, auprc = scores(model, full_loader, device)
        line = f"test_full AUROC {auroc:.4f} AUPRC {auprc:.4f} n={len(test_all)}\n"
        handle.write(line)
        print(line, flush=True)
        if len(test_amb) and test_amb.label.nunique() > 1:
            amb_loader = DataLoader(Pairs(test_amb), batch_size=16, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)
            auroc, auprc = scores(model, amb_loader, device)
            line = f"test_ambiguous AUROC {auroc:.4f} AUPRC {auprc:.4f} n={len(test_amb)}\n"
            handle.write(line)
            print(line, flush=True)


if __name__ == "__main__":
    main()
