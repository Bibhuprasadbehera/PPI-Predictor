"""Stage 1A: canonical Cα chains, pair table, UniProt-component split, stats."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[1]
DEEP = Path("/home/bibhu/Documents/DeepInteract/DEEPINTERACT")
OUT = ROOT / "model_data"
CHAINS = OUT / "chains"
SPLITS = OUT / "splits"

DROP = {
    ("2PM7", "A", "B"),
    ("2PM7", "C", "D"),
    ("1SFC", "E", "H"),
}


def load_needed(paths, needed):
    found = {}
    for path in paths:
        for rec in SeqIO.parse(path, "fasta"):
            if rec.id in needed and rec.id not in found:
                found[rec.id] = str(rec.seq)
        if len(found) == len(needed):
            break
    return found


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


def main():
    pos = pd.read_csv(
        DEEP / "Positive_data/files/msa_with_contacts_filtered_pfam_combined_reshaped.tsv",
        sep="\t",
    )
    neg = pd.read_csv(DEEP / "Negative_data/files/negative_dataset.tsv", sep="\t")
    pos["label"] = 1
    neg["label"] = 0
    df = pd.concat([pos, neg], ignore_index=True)

    drop_keys = {(pdb, *tuple(sorted((a, b)))) for pdb, a, b in DROP}
    keep = []
    for row in df.itertuples(index=False):
        key = (row.pdb_id, *tuple(sorted((row.chain_1, row.chain_2))))
        bad = row.label == 0 and (key in drop_keys or row.no_of_contacts >= 20)
        bad = bad or (row.label == 1 and row.no_of_contacts < 20)
        keep.append(not bad)
    df = df.loc[keep].copy()

    chain_ids = set(df.pdb_id + "_" + df.chain_1) | set(df.pdb_id + "_" + df.chain_2)
    seqs = load_needed(
        [
            DEEP / "Positive_data/fasta/combined_seq.fasta",
            DEEP / "Negative_data/fasta/combined_seq.fasta",
        ],
        chain_ids,
    )
    missing = sorted(chain_ids - set(seqs))
    if missing:
        raise SystemExit(f"missing {len(missing)} chains, e.g. {missing[:5]}")

    CHAINS.mkdir(parents=True, exist_ok=True)
    for cid, seq in seqs.items():
        path = CHAINS / f"{cid}.fasta"
        if path.exists() and path.stat().st_size > 0:
            continue
        path.write_text(f">{cid}\n{seq}\n")

    df["chain_id_1"] = df.pdb_id + "_" + df.chain_1
    df["chain_id_2"] = df.pdb_id + "_" + df.chain_2
    df["len_1"] = df.chain_id_1.map(lambda c: len(seqs[c]))
    df["len_2"] = df.chain_id_2.map(lambda c: len(seqs[c]))
    df["pair_id"] = df.pdb_id + "_" + df.chain_1 + "_" + df.chain_2

    pos_seq_pairs = set()
    for row in df.itertuples():
        if row.label != 1:
            continue
        a, b = seqs[row.chain_id_1], seqs[row.chain_id_2]
        pos_seq_pairs.add(tuple(sorted((a, b))))
    ident = []
    for row in df.itertuples():
        a, b = seqs[row.chain_id_1], seqs[row.chain_id_2]
        ident.append(row.label == 0 and tuple(sorted((a, b))) in pos_seq_pairs)
    df["seq_identical_to_a_positive"] = ident

    parent = {}
    for row in df.itertuples():
        union(parent, row.uniprot_id_1, row.uniprot_id_2)
    for _pdb, group in df.groupby("pdb_id"):
        ids = list(set(group.uniprot_id_1) | set(group.uniprot_id_2))
        for other in ids[1:]:
            union(parent, ids[0], other)
    comp_pairs = defaultdict(list)
    for i, row in enumerate(df.itertuples()):
        comp_pairs[find(parent, row.uniprot_id_1)].append(i)
    components = sorted(comp_pairs.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
    n = len(df)
    target = {"train": int(round(0.70 * n)), "val": int(round(0.15 * n))}
    assigned = {name: [] for name in ("train", "val", "test")}
    counts = {name: 0 for name in assigned}
    for _root, idxs in components:
        if counts["train"] < target["train"]:
            bucket = "train"
        elif counts["val"] < target["val"]:
            bucket = "val"
        else:
            bucket = "test"
        assigned[bucket].extend(idxs)
        counts[bucket] += len(idxs)
    df["split"] = ""
    for name, idxs in assigned.items():
        df.loc[df.index[idxs], "split"] = name

    keep_cols = [
        "pair_id", "pdb_id", "chain_1", "chain_2", "chain_id_1", "chain_id_2",
        "label", "no_of_contacts", "len_1", "len_2",
        "uniprot_id_1", "uniprot_id_2", "sequence_similarity",
        "pfam_domain", "pfam_domains_1", "pfam_domains_2",
        "seq_identical_to_a_positive", "split",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    df[keep_cols].to_csv(OUT / "pairs.tsv", sep="\t", index=False)
    SPLITS.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        ids = df.loc[df.split == name, "pair_id"]
        (SPLITS / f"{name}.txt").write_text("\n".join(ids) + "\n")

    lengths = list(map(len, seqs.values()))
    pos_df = df[df.label == 1]
    neg_df = df[df.label == 0]
    cells = int((df.len_1 * df.len_2).sum())
    contacts = float(df.no_of_contacts.sum())
    lines = [
        "# Stage 1A stats",
        "",
        f"- Chains written: {len(seqs)}",
        f"- Pairs: {len(df)} (positive {len(pos_df)}, negative {len(neg_df)})",
        f"- Zero-contact negatives: {int((neg_df.no_of_contacts == 0).sum())}",
        f"- Negatives with a sequence pair identical to a positive: {int(df.seq_identical_to_a_positive.sum())}",
        f"- Chain length min/median/max: {min(lengths)} / {sorted(lengths)[len(lengths)//2]} / {max(lengths)}",
        f"- Chains longer than 1022: {sum(L > 1022 for L in lengths)}",
        f"- Chains longer than 2048: {sum(L > 2048 for L in lengths)}",
        f"- Table contact density: {100 * contacts / cells:.5f}% ({int(contacts)} / {cells})",
        "",
        "## Split",
        "",
    ]
    u, p = {}, {}
    for name in ("train", "val", "test"):
        sub = df[df.split == name]
        lines.append(f"- {name}: {len(sub)} pairs, positive {int((sub.label == 1).sum())}")
        u[name] = set(sub.uniprot_id_1) | set(sub.uniprot_id_2)
        p[name] = set(sub.pdb_id)
    lines += [
        f"- Shared UniProt train∩val: {len(u['train'] & u['val'])}",
        f"- Shared UniProt train∩test: {len(u['train'] & u['test'])}",
        f"- Shared UniProt val∩test: {len(u['val'] & u['test'])}",
        f"- Shared PDB train∩val: {len(p['train'] & p['val'])}",
        f"- Shared PDB train∩test: {len(p['train'] & p['test'])}",
        f"- Shared PDB val∩test: {len(p['val'] & p['test'])}",
        "",
        "Conservation profiles are written by `run_conservation.py`.",
        "",
    ]
    (OUT / "stats.md").write_text("\n".join(lines))
    summary = {"chains": len(seqs), "pairs": int(len(df)), "splits": {k: int(v) for k, v in counts.items()}}
    (OUT / "stage1a_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
