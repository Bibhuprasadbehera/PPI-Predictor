"""Rebalance the component split and flag sequence pairs that carry both labels.

Components stay whole, so a UniProt id or a PDB id never crosses a split.
Assignment tracks both the 70/15/15 sizes and the global positive rate.
Identical (seq1, seq2) pairs that occur with both labels are flagged and left
in the table. Training drops them from the loss and reports them separately.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "model_data"
CHAINS = DATA / "chains"


def read_seq(chain_id: str) -> str:
    path = CHAINS / f"{chain_id}.fasta"
    return "".join(
        line.strip() for line in path.read_text().splitlines() if line and not line.startswith(">")
    )


def digest(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()


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


def pack(components, n_pos: int):
    n = sum(comp["n"] for comp in components)
    rate = n_pos / n
    target_n = {
        "train": int(round(0.70 * n)),
        "val": int(round(0.15 * n)),
    }
    target_n["test"] = n - target_n["train"] - target_n["val"]
    target_p = {
        "train": int(round(rate * target_n["train"])),
        "val": int(round(rate * target_n["val"])),
    }
    target_p["test"] = n_pos - target_p["train"] - target_p["val"]
    state = {name: {"idx": [], "n": 0, "pos": 0} for name in ("train", "val", "test")}
    for comp in sorted(components, key=lambda c: (-c["n"], c["root"])):
        def score(name, comp=comp):
            new_n = state[name]["n"] + comp["n"]
            new_p = state[name]["pos"] + comp["pos"]
            overflow = max(0, new_n - target_n[name])
            rate_err = abs(new_p / new_n - rate)
            fill = state[name]["n"] / target_n[name]
            return (overflow, rate_err, fill)

        choice = min(state, key=score)
        state[choice]["idx"].extend(comp["idx"])
        state[choice]["n"] += comp["n"]
        state[choice]["pos"] += comp["pos"]
    return state, target_n, rate


def main():
    df = pd.read_csv(DATA / "pairs.tsv", sep="\t")
    chain_ids = sorted(set(df.chain_id_1) | set(df.chain_id_2))
    seqs = {cid: read_seq(cid) for cid in chain_ids}
    hashes = {cid: digest(seq) for cid, seq in seqs.items()}
    labels = defaultdict(set)
    keys = []
    for row in df.itertuples(index=False):
        key = tuple(sorted((hashes[row.chain_id_1], hashes[row.chain_id_2])))
        keys.append(key)
        labels[key].add(int(row.label))
    ambiguous_keys = {key for key, seen in labels.items() if len(seen) > 1}
    df["seq_ambiguous"] = [key in ambiguous_keys for key in keys]

    parent = {}
    for row in df.itertuples(index=False):
        union(parent, row.uniprot_id_1, row.uniprot_id_2)
    for _pdb, group in df.groupby("pdb_id"):
        ids = list(set(group.uniprot_id_1) | set(group.uniprot_id_2))
        for other in ids[1:]:
            union(parent, ids[0], other)
    grouped = defaultdict(list)
    for i, row in enumerate(df.itertuples(index=False)):
        grouped[find(parent, row.uniprot_id_1)].append(i)
    components = []
    for root, idxs in grouped.items():
        components.append({
            "root": str(root),
            "idx": idxs,
            "n": len(idxs),
            "pos": int(df.iloc[idxs].label.sum()),
        })
    state, target_n, rate = pack(components, int(df.label.sum()))
    df["split"] = ""
    for name, bucket in state.items():
        df.loc[df.index[bucket["idx"]], "split"] = name
    if (df.split == "").any():
        raise SystemExit("unassigned pairs")

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        a, b = df[df.split == left], df[df.split == right]
        shared_u = (set(a.uniprot_id_1) | set(a.uniprot_id_2)) & (set(b.uniprot_id_1) | set(b.uniprot_id_2))
        shared_p = set(a.pdb_id) & set(b.pdb_id)
        if shared_u or shared_p:
            raise SystemExit(f"leakage {left}/{right} uniprot {len(shared_u)} pdb {len(shared_p)}")

    df.to_csv(DATA / "pairs.tsv", sep="\t", index=False)
    split_dir = DATA / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Split after label balancing",
        "",
        f"- Pairs: {len(df)}  positive rate {rate:.4f}",
        f"- Sequence pairs with both labels: {int(df.seq_ambiguous.sum())} "
        f"({100 * df.seq_ambiguous.mean():.1f}%)",
        f"- Distinct contradictory sequence pairs: {len(ambiguous_keys)}",
        "",
    ]
    for name in ("train", "val", "test"):
        sub = df[df.split == name]
        amb = sub[sub.seq_ambiguous]
        clear = sub[~sub.seq_ambiguous]
        ids = sub.pair_id.astype(str)
        (split_dir / f"{name}.txt").write_text("\n".join(ids) + "\n")
        lines.append(
            f"- {name}: {len(sub)} pairs (target {target_n[name]}), "
            f"positive {int(sub.label.sum())} rate {sub.label.mean():.4f}; "
            f"unambiguous {len(clear)} rate {clear.label.mean():.4f}; "
            f"ambiguous {len(amb)}"
        )
    lines += [
        "",
        "Ambiguous pairs stay in the table. The training loss and the primary",
        "test metric use the unambiguous pairs. The full test set is still scored.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (DATA / "split_balance.md").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
