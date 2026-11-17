"""Redraw the data-stat figure and the baseline-result figure from measured files."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from figure_style import INK, MM, NEGATIVE, NEUTRAL, POSITIVE, WARN, apply_style, panel_letter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "model_data"
AUDIT = ROOT / "figures" / "audit"
FIG = ROOT / "figures"
OLD_DATA = ROOT / "plots" / "data_stats"
OLD_BASE = ROOT / "plots" / "baseline_results"


def save(fig, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for dest in (folder / name, FIG / name):
        fig.savefig(dest.with_suffix(".pdf"))
        fig.savefig(dest.with_suffix(".png"))
    plt.close(fig)


def metric_row(metrics, stage, item):
    hit = metrics[(metrics.stage == stage) & (metrics.item == item)]
    if hit.empty:
        raise KeyError(f"{stage}/{item}")
    return hit.iloc[0]


def data_stats(pairs: pd.DataFrame, metrics: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplot_mosaic(
        [["A", "A", "A", "A"],
         ["B", "B", "C", "C"],
         ["D", "D", "E", "E"]],
        figsize=(183 * MM, 180 * MM),
        layout="constrained",
    )
    # A. How complete the inputs are. Chain-level rates, one denominator each.
    checks = [
        ("Cα FASTA", metric_row(metrics, "feature_presence", "fasta")),
        ("Embedding, length matched", metric_row(metrics, "length_match", "embedding")),
        ("Motif array, length matched", metric_row(metrics, "length_match", "motif")),
        ("Conservation, length matched", metric_row(metrics, "length_match", "conservation")),
        ("Conservation from an MSA", metric_row(metrics, "conservation_source", "msa|v2")),
        ("At least one FIMO hit", metric_row(metrics, "motif_coverage", "chains_with_any_fimo_hit")),
    ]
    amb = metric_row(metrics, "label_conflict", "pairs_with_both_labels_on_same_sequences")
    both_u = metric_row(metrics, "label_conflict", "pairs_whose_uniprot_pair_has_both_labels")
    names = [name for name, _row in checks] + [
        "One label for the sequences",
        "One label for the UniProt pair",
    ]
    rates = [row.rate for _name, row in checks] + [1 - amb.rate, 1 - both_u.rate]
    totals = [int(row.n_total) for _name, row in checks] + [int(amb.n_total), int(both_u.n_total)]
    counts = [int(row.n) for _name, row in checks] + [int(amb.n_total - amb.n), int(both_u.n_total - both_u.n)]
    y = np.arange(len(names))
    axes["A"].barh(y, rates, color=[POSITIVE] * 4 + [NEUTRAL] * 2 + [WARN] * 2)
    axes["A"].set_yticks(y, names)
    axes["A"].set_xlim(0, 1.25)
    axes["A"].set_xlabel("Fraction complete")
    axes["A"].invert_yaxis()
    for i, (rate, n, total) in enumerate(zip(rates, counts, totals)):
        axes["A"].text(rate + 0.02, i, f"{n}/{total}", va="center", fontsize=7)
    axes["A"].set_title("Blue bars are files. Green bars are annotation. Orange bars are a single label.")
    panel_letter(axes["A"], "A")

    pos = int((pairs.label == 1).sum())
    neg = pairs[pairs.label == 0]
    zero = int((neg.no_of_contacts == 0).sum())
    mid = int(((neg.no_of_contacts > 0) & (neg.no_of_contacts < 20)).sum())
    axes["B"].bar(
        ["Positive\n(≥20 contacts)", "Negative\n(1–19 contacts)", "Negative\n(0 contacts)"],
        [pos, mid, zero],
        color=[POSITIVE, "#56B4E9", NEGATIVE],
    )
    for x, n in enumerate((pos, mid, zero)):
        axes["B"].text(x, n, str(n), ha="center", va="bottom", fontsize=7)
    axes["B"].set_ylabel("Pairs")
    axes["B"].set_ylim(0, max(pos, mid, zero) * 1.15)
    axes["B"].set_title(f"n = {len(pairs)}")
    panel_letter(axes["B"], "B")

    for label, color, name in ((1, POSITIVE, "Positive"), (0, NEGATIVE, "Negative")):
        vals = pairs.loc[pairs.label == label, "no_of_contacts"].to_numpy()
        axes["C"].hist(np.log10(vals + 1), bins=30, histtype="step", linewidth=1.2, color=color, label=name)
    axes["C"].axvline(np.log10(21), color=INK, linestyle="--", linewidth=0.8, label="20 contacts")
    axes["C"].set_xlabel("log10(contacts + 1)")
    axes["C"].set_ylabel("Pairs")
    axes["C"].legend(frameon=False)
    axes["C"].set_title(f"{zero} negatives sit at 0")
    panel_letter(axes["C"], "C")

    lengths = {}
    for row in pairs.itertuples(index=False):
        lengths[row.chain_id_1] = int(row.len_1)
        lengths[row.chain_id_2] = int(row.len_2)
    lens = np.asarray(list(lengths.values()))
    axes["D"].hist(lens, bins=40, color="#4D4D4D", linewidth=0)
    axes["D"].axvline(1022, color=WARN, linestyle="--", linewidth=0.8)
    over = int((lens > 1022).sum())
    axes["D"].set_xlabel("Chain length (Cα residues)")
    axes["D"].set_ylabel("Chains")
    axes["D"].set_title(f"n = {len(lens)}; {over} longer than 1,022")
    panel_letter(axes["D"], "D")

    order = ["train", "val", "test"]
    pos_n, neg_n = [], []
    for name in order:
        sub = pairs[pairs.split == name]
        pos_n.append(int((sub.label == 1).sum()))
        neg_n.append(int((sub.label == 0).sum()))
    axes["E"].bar(order, pos_n, color=POSITIVE, label="Positive")
    axes["E"].bar(order, neg_n, bottom=pos_n, color=NEGATIVE, label="Negative")
    for x, (a, b) in enumerate(zip(pos_n, neg_n)):
        axes["E"].text(x, a + b, str(a + b), ha="center", va="bottom", fontsize=7)
    axes["E"].set_ylabel("Pairs")
    axes["E"].legend(frameon=False)
    axes["E"].set_ylim(0, max(a + b for a, b in zip(pos_n, neg_n)) * 1.15)
    axes["E"].set_title("Current split")
    panel_letter(axes["E"], "E")
    save(fig, OLD_DATA, "data_stats")


def baseline(scores: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplot_mosaic(
        [["A", "B"], ["C", "D"]],
        figsize=(183 * MM, 140 * MM),
        layout="constrained",
    )
    clear = scores[~scores.seq_ambiguous.astype(bool)]
    y = clear.label.to_numpy()
    prob = clear.prob.to_numpy()
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    axes["A"].plot(fpr, tpr, color=POSITIVE, label=f"Mean pool {auc:.3f}")
    axes["A"].plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=0.8, label="Chance 0.500")
    axes["A"].set_xlim(0, 1)
    axes["A"].set_ylim(0, 1)
    axes["A"].set_xlabel("False positive rate")
    axes["A"].set_ylabel("True positive rate")
    axes["A"].legend(frameon=False, loc="lower right")
    axes["A"].set_title(f"Unambiguous test, n = {len(clear)}")
    panel_letter(axes["A"], "A")

    precision, recall, _ = precision_recall_curve(y, prob)
    ap = average_precision_score(y, prob)
    base = float(y.mean())
    axes["B"].plot(recall, precision, color=POSITIVE, label=f"Mean pool {ap:.3f}")
    axes["B"].axhline(base, color=INK, linestyle="--", linewidth=0.8, label=f"Positive rate {base:.3f}")
    axes["B"].set_xlim(0, 1)
    axes["B"].set_ylim(0, 1)
    axes["B"].set_xlabel("Recall")
    axes["B"].set_ylabel("Precision")
    axes["B"].legend(frameon=False, loc="upper right")
    axes["B"].set_title(f"Unambiguous test, n = {len(clear)}")
    panel_letter(axes["B"], "B")

    log = (DATA / "train_meanpool_scaled.log").read_text().splitlines()
    epochs, loss, auroc, auprc = [], [], [], []
    for line in log:
        if not line.startswith("epoch"):
            continue
        parts = dict(zip(line.split()[::2], line.split()[1::2]))
        epochs.append(int(parts["epoch"]))
        loss.append(float(parts["train_loss"]))
        auroc.append(float(parts["val_auroc"]))
        auprc.append(float(parts["val_auprc"]))
    axes["C"].plot(epochs, auroc, color=POSITIVE, marker="o", label="Val AUROC")
    axes["C"].plot(epochs, auprc, color=WARN, marker="o", label="Val AUPRC")
    axes["C"].set_xlabel("Epoch")
    axes["C"].set_ylabel("Validation score")
    axes["C"].set_ylim(0.2, 0.7)
    axes["C"].legend(frameon=False)
    axes["C"].set_title(f"Train loss {loss[0]:.2f} to {loss[-1]:.2f}")
    panel_letter(axes["C"], "C")

    slices = []
    for name, frame in (
        ("Unambiguous", clear),
        ("Full test", scores),
        ("Both labels", scores[scores.seq_ambiguous.astype(bool)]),
    ):
        yy = frame.label.to_numpy()
        pp = frame.prob.to_numpy()
        slices.append((name, len(frame), roc_auc_score(yy, pp), average_precision_score(yy, pp), float(yy.mean())))
    x = np.arange(len(slices))
    axes["D"].bar(x - 0.15, [row[2] for row in slices], width=0.3, color=POSITIVE, label="AUROC")
    axes["D"].bar(x + 0.15, [row[3] for row in slices], width=0.3, color=WARN, label="AUPRC")
    for i, row in enumerate(slices):
        axes["D"].plot([i - 0.35, i + 0.35], [row[4], row[4]], color=INK, linewidth=0.8)
    axes["D"].axhline(0.5, color=INK, linestyle="--", linewidth=0.6)
    axes["D"].set_xticks(x, [f"{row[0]}\nn={row[1]}" for row in slices])
    axes["D"].set_ylim(0, 1)
    axes["D"].set_ylabel("Score")
    axes["D"].legend(frameon=False)
    axes["D"].set_title("Black ticks: positive rate in that slice")
    panel_letter(axes["D"], "D")
    save(fig, OLD_BASE, "baseline_results")


def main():
    pairs = pd.read_csv(DATA / "pairs.tsv", sep="\t")
    metrics = pd.read_csv(AUDIT / "coverage_metrics.csv")
    scores = pd.read_csv(AUDIT / "meanpool_scaled_test.tsv", sep="\t")
    data_stats(pairs, metrics)
    baseline(scores)
    print("wrote data_stats and baseline_results", flush=True)


if __name__ == "__main__":
    main()
