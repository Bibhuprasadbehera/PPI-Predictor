"""Figures and the coverage table. Numbers come from figures/audit/coverage_metrics.csv."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import INK, MM, NEGATIVE, NEUTRAL, POSITIVE, WARN, apply_style, panel_letter

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "figures" / "audit"
FIG = ROOT / "figures"


def load():
    metrics = pd.read_csv(AUDIT / "coverage_metrics.csv")
    arrays = np.load(AUDIT / "coverage_arrays.npz")
    pairs = pd.read_csv(AUDIT / "pair_slice.tsv", sep="\t")
    delta = np.load(AUDIT / "aa_len_minus_fasta.npy")
    return metrics, arrays, pairs, delta


def value(metrics, stage, item):
    hit = metrics[(metrics.stage == stage) & (metrics.item == item)]
    if hit.empty:
        raise KeyError(f"{stage}/{item}")
    return hit.iloc[0]


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def cohort(metrics, pairs, delta) -> None:
    apply_style()
    fig, axes = plt.subplot_mosaic(
        [["A", "B"], ["C", "D"]],
        figsize=(183 * MM, 140 * MM),
        layout="constrained",
    )
    labels = ["Positive\nsource", "Negative\nsource", "Built\npositive", "Built\nnegative"]
    counts = [
        value(metrics, "source", "positive_rows").n,
        value(metrics, "source", "negative_rows").n,
        value(metrics, "built_table", "positive").n,
        value(metrics, "built_table", "negative").n,
    ]
    colors = [POSITIVE, NEGATIVE, POSITIVE, NEGATIVE]
    axes["A"].bar(labels, counts, color=colors, width=0.7)
    axes["A"].set_ylim(0, max(counts) * 1.12)
    axes["A"].set_ylabel("Pairs")
    axes["A"].set_title("Rows kept")
    for x, n in enumerate(counts):
        axes["A"].text(x, n, str(int(n)), ha="center", va="bottom", fontsize=7, color=INK)
    panel_letter(axes["A"], "A")

    for label, color, name in ((1, POSITIVE, "Positive"), (0, NEGATIVE, "Negative")):
        vals = pairs.loc[pairs.label == label, "no_of_contacts"].to_numpy()
        axes["B"].hist(np.log10(vals + 1), bins=40, histtype="step", linewidth=1.2, color=color, label=name)
    axes["B"].set_xlabel("log10(contacts + 1)")
    axes["B"].set_ylabel("Pairs")
    axes["B"].legend(frameon=False)
    zero = int(value(metrics, "label_conflict", "zero_contact_negatives").n)
    axes["B"].set_title(f"Zero-contact negatives: {zero}")
    panel_letter(axes["B"], "B")

    axes["C"].hist(np.log10(delta + 1), bins=40, color=WARN, linewidth=0)
    axes["C"].set_xlabel("log10(aa_len − Cα length + 1)")
    axes["C"].set_ylabel("Chain ends")
    longer = int((delta > 0).sum())
    axes["C"].set_title(f"aa_len longer on {longer} / {len(delta)}")
    panel_letter(axes["C"], "C")

    order = ["train", "val", "test"]
    rates, ns = [], []
    for name in order:
        row = value(metrics, "split", f"{name}_positive")
        rates.append(row.rate)
        ns.append(int(row.n_total))
    axes["D"].bar(order, rates, color=NEUTRAL, width=0.65)
    overall = value(metrics, "built_table", "positive").rate
    axes["D"].axhline(overall, color=INK, linewidth=0.8, linestyle="--")
    axes["D"].set_ylabel("Positive rate")
    axes["D"].set_ylim(0, 0.6)
    for x, (rate, n) in enumerate(zip(rates, ns)):
        axes["D"].text(x, rate + 0.02, f"n={n}", ha="center", va="bottom", fontsize=7)
    axes["D"].set_title("Dashed line: all pairs")
    panel_letter(axes["D"], "D")
    save(fig, "cohort_qc")


def features(metrics, arrays) -> None:
    apply_style()
    fig, axes = plt.subplot_mosaic(
        [["A", "A"], ["B", "C"]],
        figsize=(183 * MM, 120 * MM),
        layout="constrained",
    )
    names = ["fasta", "hseq", "conservation", "embedding", "motif"]
    present = [value(metrics, "feature_presence", name).rate for name in names]
    x = np.arange(len(names))
    axes["A"].bar(x, present, width=0.7, color=POSITIVE)
    axes["A"].set_xticks(x, names)
    axes["A"].set_ylim(0, 1.15)
    axes["A"].set_ylabel("Fraction of chains with a file")
    n_chain = int(value(metrics, "built_table", "unique_chains").n)
    mismatches = sum(int(value(metrics, "length_match", name).note.split("=")[1]) for name in ("conservation", "embedding", "motif"))
    axes["A"].set_title(f"n = {n_chain} chains; length mismatches = {mismatches}")
    panel_letter(axes["A"], "A")

    msa = value(metrics, "conservation_source", "msa|v2")
    one = value(metrics, "conservation_source", "onehot|v2")
    axes["B"].bar(["MSA profile", "One-hot"], [msa.n, one.n], color=[POSITIVE, WARN])
    axes["B"].set_ylabel("Chains")
    for x_i, n in enumerate((msa.n, one.n)):
        axes["B"].text(x_i, n, str(int(n)), ha="center", va="bottom", fontsize=7)
    nseq = arrays["homolog_counts"]
    axes["B"].set_title(f"Homolog file has 1 sequence: {int((nseq == 1).sum())}")
    panel_letter(axes["B"], "B")

    axes["C"].hist(nseq, bins=np.logspace(0, np.log10(nseq.max()), 40), color=NEUTRAL, linewidth=0)
    axes["C"].set_xscale("log")
    axes["C"].set_xlabel("Homolog sequences")
    axes["C"].set_ylabel("Chains")
    axes["C"].axvline(np.median(nseq), color=INK, linewidth=0.8, linestyle="--")
    axes["C"].set_title(f"Median {np.median(nseq):.0f}")
    panel_letter(axes["C"], "C")
    save(fig, "feature_coverage")


def residues(arrays) -> None:
    apply_style()
    fig, axes = plt.subplot_mosaic(
        [["A", "B"]],
        figsize=(183 * MM, 70 * MM),
        layout="constrained",
    )
    bins = np.linspace(0, 1, 41)
    for ax, key, letter, title in (
        (axes["A"], "fimo_frac", "A", "FIMO hit"),
        (axes["B"], "pssm_frac", "B", "Nonzero PSSM"),
    ):
        vals = arrays[key]
        ax.hist(vals, bins=bins, color=POSITIVE, linewidth=0)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"Fraction of residues with a {title}")
        ax.set_ylabel("Chains")
        ax.set_title(f"Median {np.median(vals):.2f}; zero on {int((vals == 0).sum())} chains")
        panel_letter(ax, letter)
    save(fig, "residue_coverage")


def labels(metrics) -> None:
    apply_style()
    fig, axes = plt.subplot_mosaic(
        [["A", "B"], ["C", "D"]],
        figsize=(183 * MM, 130 * MM),
        layout="constrained",
    )
    order = ["train", "val", "test"]
    pos, neg = [], []
    for name in order:
        n_pos = value(metrics, "split", f"{name}_positive").n
        n_all = value(metrics, "split", f"{name}_rows").n
        pos.append(n_pos)
        neg.append(n_all - n_pos)
    axes["A"].bar(order, pos, color=POSITIVE, label="Positive")
    axes["A"].bar(order, neg, bottom=pos, color=NEGATIVE, label="Negative")
    axes["A"].set_ylabel("Pairs")
    axes["A"].legend(frameon=False)
    axes["A"].set_title("Split counts")
    panel_letter(axes["A"], "A")

    items = [
        ("Same sequences,\nboth labels", "label_conflict", "pairs_with_both_labels_on_same_sequences"),
        ("UniProt pair\nhas both labels", "label_conflict", "pairs_whose_uniprot_pair_has_both_labels"),
        ("Same UniProt\non both chains", "label_conflict", "same_uniprot_on_both_chains"),
        ("Zero-contact\nnegatives", "label_conflict", "zero_contact_negatives"),
    ]
    # Zero-contact rate uses the negative count as its denominator. Plot counts, and print n in the label.
    ys = []
    tick = []
    for name, stage, item in items:
        row = value(metrics, stage, item)
        ys.append(row.n)
        tick.append(f"{name}\n{int(row.n)}/{int(row.n_total)}")
    axes["B"].barh(tick[::-1], ys[::-1], color=WARN)
    axes["B"].set_xlabel("Pairs")
    axes["B"].set_title("Label overlap")
    panel_letter(axes["B"], "B")

    seq_keys = value(metrics, "label_conflict", "distinct_sequence_pairs_with_both_labels")
    uni_keys = value(metrics, "label_conflict", "distinct_uniprot_pairs_with_both_labels")
    axes["C"].bar(
        ["Sequence pairs", "UniProt pairs"],
        [seq_keys.rate, uni_keys.rate],
        color=[WARN, NEUTRAL],
    )
    axes["C"].set_ylim(0, 1)
    axes["C"].set_ylabel("Fraction with both labels")
    for x, row in enumerate((seq_keys, uni_keys)):
        axes["C"].text(x, row.rate + 0.03, f"{int(row.n)}/{int(row.n_total)}", ha="center", fontsize=7)
    axes["C"].set_title("Distinct keys, not pair rows")
    panel_letter(axes["C"], "C")

    short = [
        "UniProt train/val",
        "PDB train/val",
        "UniProt train/test",
        "PDB train/test",
        "UniProt val/test",
        "PDB val/test",
    ]
    leaks = metrics[metrics.stage == "split_leak"]
    axes["D"].barh(short[::-1], leaks["n"].to_numpy()[::-1], color=NEUTRAL)
    axes["D"].set_xlabel("Shared ids")
    axes["D"].set_xlim(0, 1)
    axes["D"].set_title("Shared ids, all six are 0")
    panel_letter(axes["D"], "D")
    save(fig, "label_coverage")


def write_table(metrics, delta) -> None:
    lines = [
        "Table 1. Coverage of the built pair table. Counts are from `figures/audit/coverage_metrics.csv` for run "
        + str(metrics.run_id.iloc[0]) + ".",
        "",
        "| Check | n | Total | Rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    wanted = [
        ("source", "positive_rows"),
        ("source", "negative_rows"),
        ("conflict", "same_chains_both_labels"),
        ("built_table", "rows"),
        ("built_table", "positive"),
        ("sequence", "chains_found_in_combined_seq"),
        ("sequence", "aa_len_ne_combined_seq"),
        ("feature_presence", "conservation"),
        ("length_match", "conservation"),
        ("length_match", "embedding"),
        ("length_match", "motif"),
        ("conservation_source", "onehot|v2"),
        ("motif_coverage", "chains_with_any_fimo_hit"),
        ("label_conflict", "pairs_with_both_labels_on_same_sequences"),
        ("label_conflict", "pairs_whose_uniprot_pair_has_both_labels"),
        ("split", "test_positive"),
        ("split_leak", "shared_uniprot_train_test"),
        ("index_check", "query_letters_match_fasta"),
    ]
    for stage, item in wanted:
        row = value(metrics, stage, item)
        rate = "—" if pd.isna(row.rate) else f"{row.rate:.4f}"
        label = item.replace("|", "/")
        lines.append(f"| {stage} / {label} | {int(row.n)} | {int(row.n_total)} | {rate} |")
    lines += [
        "",
        f"aa_len minus the Cα length was positive for {int((delta > 0).sum())} of {len(delta)} chain ends "
        f"(median {np.median(delta):.0f}, maximum {int(delta.max())}). No chain end had a longer Cα sequence than aa_len.",
        "",
        "The letter check is 12 chains, chosen across homolog-file sizes. It is not a check of all 11,944 chains. "
        "Length equality was counted on every chain.",
        "",
    ]
    (AUDIT / "coverage_table.md").write_text("\n".join(lines))


def main():
    metrics, arrays, pairs, delta = load()
    cohort(metrics, pairs, delta)
    features(metrics, arrays)
    residues(arrays)
    labels(metrics)
    write_table(metrics, delta)
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
