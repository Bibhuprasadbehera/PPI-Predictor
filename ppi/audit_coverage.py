"""Measure pair-table construction and feature coverage from the source files.

Writes a metrics table. It does not train a model and it does not edit features.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import AlignIO, SeqIO

from run_conservation import HMMALIGN, HMMBUILD, find_hseq, query_row, read_seq

ROOT = Path(__file__).resolve().parents[1]
DEEP = Path("/home/bibhu/Documents/DeepInteract/DEEPINTERACT")
DATA = ROOT / "model_data"
OUT = ROOT / "figures" / "audit"
POS_PATH = DEEP / "Positive_data/files/msa_with_contacts_filtered_pfam_combined_reshaped.tsv"
NEG_PATH = DEEP / "Negative_data/files/negative_dataset.tsv"
FASTA = [
    DEEP / "Positive_data/fasta/combined_seq.fasta",
    DEEP / "Negative_data/fasta/combined_seq.fasta",
]
# The written drop list in ppi/build_stage1a.py. Counted here, not assumed.
DROP = {("2PM7", "A", "B"), ("2PM7", "C", "D"), ("1SFC", "E", "H")}
RUN_ID = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def log(event: str, **fields) -> None:
    parts = [f"{key}={value}" for key, value in fields.items()]
    line = f"INFO  {event}  " + " ".join(parts)
    print(line, flush=True)
    with (OUT / "audit.log").open("a") as handle:
        handle.write(line + "\n")


def chain_key(pdb, a, b):
    return (pdb, *tuple(sorted((str(a), str(b)))))


def load_seqs(needed: set[str]) -> dict[str, str]:
    found = {}
    for path in FASTA:
        for rec in SeqIO.parse(path, "fasta"):
            if rec.id in needed and rec.id not in found:
                found[rec.id] = str(rec.seq)
        if len(found) == len(needed):
            break
    return found


def add(rows, stage, item, n, n_total, unit, status="ok", note=""):
    rate = None if n_total in (0, None) else n / n_total
    rows.append({
        "run_id": RUN_ID,
        "stage": stage,
        "item": item,
        "n": int(n),
        "n_total": None if n_total is None else int(n_total),
        "rate": rate,
        "unit": unit,
        "status": status,
        "note": note,
    })


def feature_scan(chain_ids: list[str], rows: list[dict]) -> dict:
    """One pass over every chain named by the pair table."""
    n = len(chain_ids)
    present = {name: 0 for name in ("fasta", "hseq", "conservation", "embedding", "motif")}
    length_ok = {name: 0 for name in ("conservation", "embedding", "motif")}
    length_bad = {name: 0 for name in ("conservation", "embedding", "motif")}
    cons_source = defaultdict(int)
    cons_bad_range = 0
    cons_missing_version = 0
    homolog_counts = []
    fimo_frac = []
    pssm_frac = []
    pssm_max = 0.0
    neglog_max = 0.0
    fimo_any = 0
    pssm_any = 0
    missing = []
    for i, cid in enumerate(chain_ids, start=1):
        fasta = DATA / "chains" / f"{cid}.fasta"
        if fasta.exists() and fasta.stat().st_size > 0:
            present["fasta"] += 1
            seq = read_seq(fasta)
            length = len(seq)
        else:
            missing.append(cid)
            continue
        hseq = DATA / "hseq" / f"{cid}.fasta"
        if hseq.exists() and hseq.stat().st_size > 0:
            present["hseq"] += 1
            homolog_counts.append(sum(1 for line in hseq.open() if line.startswith(">")))
        cons = DATA / "conservation" / f"{cid}.npz"
        if cons.exists() and cons.stat().st_size > 0:
            present["conservation"] += 1
            with np.load(cons) as blob:
                profile = blob["profile"]
                source = str(blob["source"]) if "source" in blob.files else "missing"
                version = int(blob["v"]) if "v" in blob.files else -1
            cons_source[f"{source}|v{version}"] += 1
            if version != 2:
                cons_missing_version += 1
            if len(profile) == length:
                length_ok["conservation"] += 1
            else:
                length_bad["conservation"] += 1
            freq = profile[:, :21].astype(np.float32).sum(axis=1)
            ent = profile[:, 21].astype(np.float32)
            if freq.max() > 1.05 or ent.max() > 1.01 or ent.min() < -0.01:
                cons_bad_range += 1
        emb = DATA / "embeddings" / f"{cid}.npy"
        if emb.exists() and emb.stat().st_size > 0:
            present["embedding"] += 1
            shape = np.load(emb, mmap_mode="r").shape
            if shape[0] == length and len(shape) == 2 and shape[1] == 1280:
                length_ok["embedding"] += 1
            else:
                length_bad["embedding"] += 1
        motif = DATA / "motifs" / f"{cid}.npz"
        if motif.exists() and motif.stat().st_size > 0:
            present["motif"] += 1
            track = np.load(motif)["motif"]
            if len(track) == length and track.shape[1] == 3:
                length_ok["motif"] += 1
                hit = float((track[:, 0] > 0).mean())
                pssm = float((track[:, 2] > 0).mean())
                fimo_frac.append(hit)
                pssm_frac.append(pssm)
                if hit > 0:
                    fimo_any += 1
                if pssm > 0:
                    pssm_any += 1
                pssm_max = max(pssm_max, float(track[:, 2].max()))
                neglog_max = max(neglog_max, float(track[:, 1].max()))
            else:
                length_bad["motif"] += 1
        if i % 2000 == 0 or i == n:
            log("feature scan", done=i, chains=n)
    for name, count in present.items():
        add(rows, "feature_presence", name, count, n, "chain")
    for name in length_ok:
        add(rows, "length_match", name, length_ok[name], present[name], "chain",
            note=f"mismatch={length_bad[name]}")
    for key, count in sorted(cons_source.items()):
        add(rows, "conservation_source", key, count, n, "chain")
    add(rows, "conservation_qc", "version_not_2", cons_missing_version, n, "chain")
    add(rows, "conservation_qc", "freq_or_entropy_out_of_range", cons_bad_range, n, "chain")
    add(rows, "motif_coverage", "chains_with_any_fimo_hit", fimo_any, present["motif"], "chain")
    add(rows, "motif_coverage", "chains_with_any_pssm", pssm_any, present["motif"], "chain")
    np.savez_compressed(
        OUT / "coverage_arrays.npz",
        homolog_counts=np.asarray(homolog_counts, dtype=np.int32),
        fimo_frac=np.asarray(fimo_frac, dtype=np.float32),
        pssm_frac=np.asarray(pssm_frac, dtype=np.float32),
    )
    summary = {
        "pssm_max": pssm_max,
        "neglog_p_max": neglog_max,
        "homolog_n": int(len(homolog_counts)),
    }
    if homolog_counts:
        arr = np.asarray(homolog_counts)
        summary["homolog_min"] = int(arr.min())
        summary["homolog_median"] = float(np.median(arr))
        summary["homolog_p90"] = float(np.quantile(arr, 0.9))
        summary["homolog_max"] = int(arr.max())
        add(rows, "homologs", "nseq_median", int(np.median(arr)), len(arr), "sequence")
    if fimo_frac:
        summary["fimo_frac_median"] = float(np.median(fimo_frac))
        summary["pssm_frac_median"] = float(np.median(pssm_frac))
        summary["fimo_frac_zero"] = int(sum(x == 0 for x in fimo_frac))
    if missing:
        (OUT / "chains_missing_fasta.txt").write_text("\n".join(missing) + "\n")
    return summary


def letter_spotcheck(chain_ids: list[str], rows: list[dict], n_check: int = 12) -> None:
    """Re-align a few chains and compare query letters with the FASTA. Not the full corpus."""
    ranked = []
    for cid in chain_ids:
        path = DATA / "hseq" / f"{cid}.fasta"
        if path.exists():
            ranked.append((path.stat().st_size, cid))
    ranked.sort()
    # Small, middle, and one larger file. The check is a sample, not a census.
    picks = []
    if ranked:
        step = max(len(ranked) // n_check, 1)
        picks = [cid for _size, cid in ranked[::step][:n_check]]
    ok = 0
    fail = []
    for cid in picks:
        seq = read_seq(DATA / "chains" / f"{cid}.fasta").upper()
        hseq = find_hseq(cid)
        try:
            with tempfile.TemporaryDirectory(prefix="audit_") as td:
                combined = Path(td) / "all.fasta"
                recs = list(SeqIO.parse(hseq, "fasta"))
                homologs = [rec for rec in recs if str(rec.seq).replace("-", "").upper() != seq]
                with combined.open("w") as handle:
                    handle.write(f">q|{cid}\n{seq}\n")
                    if homologs:
                        SeqIO.write(homologs, handle, "fasta")
                hmm = Path(td) / "q.hmm"
                sto = Path(td) / "aln.sto"
                subprocess.run(
                    [HMMBUILD, "--amino", str(hmm), str(DATA / "chains" / f"{cid}.fasta")],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    [HMMALIGN, "-o", str(sto), str(hmm), str(combined)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                alignment = AlignIO.read(sto, "stockholm")
                query = query_row(alignment, cid)
                letters = "".join(c for c in str(query.seq) if c not in "-.").upper()
            if letters == seq:
                ok += 1
            else:
                fail.append(cid)
        except Exception as exc:
            fail.append(f"{cid}:{type(exc).__name__}")
        log("letter check", chain=cid, ok_so_far=ok, tested=len(picks))
    add(
        rows, "index_check", "query_letters_match_fasta", ok, len(picks), "chain",
        status="ok" if not fail else "failed",
        note="sample_not_census; fail=" + ",".join(fail),
    )


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.log").write_text("")
    log(
        "run started", run_id=RUN_ID, pipeline="coverage_audit",
        utc_start=datetime.now(timezone.utc).isoformat(),
        pos=str(POS_PATH), neg=str(NEG_PATH),
    )
    rows = []
    pos = pd.read_csv(POS_PATH, sep="\t")
    neg = pd.read_csv(NEG_PATH, sep="\t")
    pos["label"] = 1
    neg["label"] = 0
    pos["source"] = "positive_tsv"
    neg["source"] = "negative_tsv"
    add(rows, "source", "positive_rows", len(pos), len(pos), "pair")
    add(rows, "source", "negative_rows", len(neg), len(neg), "pair")
    raw = pd.concat([pos, neg], ignore_index=True)
    raw["chain_key"] = [
        chain_key(r.pdb_id, r.chain_1, r.chain_2) for r in raw.itertuples(index=False)
    ]
    dup = int(raw.duplicated("chain_key").sum())
    add(rows, "source", "duplicate_chain_keys", dup, len(raw), "pair")

    both = []
    for key, group in raw.groupby("chain_key"):
        if group.label.nunique() > 1:
            both.append(key)
    add(rows, "conflict", "same_chains_both_labels", len(both), len(raw), "pair",
        note=",".join(f"{a}:{b}-{c}" for a, b, c in both))
    contact_pos_low = int(((raw.label == 1) & (raw.no_of_contacts < 20)).sum())
    contact_neg_high = int(((raw.label == 0) & (raw.no_of_contacts >= 20)).sum())
    add(rows, "contact_rule", "positive_under_20", contact_pos_low, len(pos), "pair")
    add(rows, "contact_rule", "negative_at_least_20", contact_neg_high, len(neg), "pair")
    drop_keys = {chain_key(*item) for item in DROP}
    listed = int(raw.chain_key.isin(drop_keys).sum())
    add(rows, "conflict", "rows_on_written_drop_list", listed, len(raw), "pair")

    raw["chain_id_1"] = raw.pdb_id + "_" + raw.chain_1
    raw["chain_id_2"] = raw.pdb_id + "_" + raw.chain_2
    needed = set(raw.chain_id_1) | set(raw.chain_id_2)
    seqs = load_seqs(needed)
    add(rows, "sequence", "chains_in_source_tables", len(needed), len(needed), "chain")
    add(rows, "sequence", "chains_found_in_combined_seq", len(seqs), len(needed), "chain")

    built = pd.read_csv(DATA / "pairs.tsv", sep="\t")
    add(rows, "built_table", "rows", len(built), len(raw), "pair")
    add(rows, "built_table", "positive", int((built.label == 1).sum()), len(built), "pair")
    add(rows, "built_table", "negative", int((built.label == 0).sum()), len(built), "pair")
    built_keys = {chain_key(r.pdb_id, r.chain_1, r.chain_2) for r in built.itertuples(index=False)}
    raw_keys = set(raw.chain_key)
    add(rows, "built_table", "keys_missing_from_source", len(built_keys - raw_keys), len(built), "pair")
    add(rows, "built_table", "source_keys_absent", len(raw_keys - built_keys), len(raw), "pair")

    # Annotation coverage on the built table.
    pfam_missing = int(built.pfam_domain.isna().sum() + (built.pfam_domain.astype(str).str.strip() == "").sum())
    # The isna sum double-counts empty strings that are also na. Recompute cleanly.
    pfam_blank = built.pfam_domain.isna() | (built.pfam_domain.astype(str).str.strip().isin(["", "nan", "None"]))
    add(rows, "annotation", "pfam_domain_blank", int(pfam_blank.sum()), len(built), "pair")
    sim = pd.to_numeric(built.sequence_similarity, errors="coerce")
    add(rows, "annotation", "sequence_similarity_missing", int(sim.isna().sum()), len(built), "pair")
    if sim.notna().any():
        add(rows, "annotation", "sequence_similarity_ge_25", int((sim >= 25).sum()), len(built), "pair",
            note=f"min={sim.min():.4g} median={sim.median():.4g} max={sim.max():.4g}")
        add(rows, "annotation", "sequence_similarity_ge_0.25", int((sim >= 0.25).sum()), len(built), "pair")

    chain_ids = sorted(set(built.chain_id_1) | set(built.chain_id_2))
    add(rows, "built_table", "unique_chains", len(chain_ids), len(chain_ids), "chain")
    lengths = {}
    for cid in chain_ids:
        path = DATA / "chains" / f"{cid}.fasta"
        if path.exists():
            lengths[cid] = len(read_seq(path))
    len_mismatch = 0
    for row in built.itertuples(index=False):
        if lengths.get(row.chain_id_1) != int(row.len_1) or lengths.get(row.chain_id_2) != int(row.len_2):
            len_mismatch += 1
    add(rows, "built_table", "pairs_len_column_ne_fasta", len_mismatch, len(built), "pair")

    aa = pd.to_numeric(raw.aa_len_1, errors="coerce")
    # Compare source aa_len with the canonical FASTA for chains we have.
    aa_diff = 0
    aa_compared = 0
    for row in raw.itertuples(index=False):
        for cid, claimed in ((row.chain_id_1, row.aa_len_1), (row.chain_id_2, row.aa_len_2)):
            if cid not in seqs:
                continue
            aa_compared += 1
            try:
                if int(claimed) != len(seqs[cid]):
                    aa_diff += 1
            except (TypeError, ValueError):
                aa_diff += 1
    add(rows, "sequence", "aa_len_ne_combined_seq", aa_diff, aa_compared, "chain_end")

    # Sequence conflicts, recomputed. Not read from the seq_ambiguous column.
    hashes = {cid: hashlib.sha1(seqs[cid].encode()).hexdigest() for cid in seqs}
    labels_for_seq = defaultdict(set)
    seq_keys = []
    for row in built.itertuples(index=False):
        if row.chain_id_1 not in hashes or row.chain_id_2 not in hashes:
            seq_keys.append(None)
            continue
        key = tuple(sorted((hashes[row.chain_id_1], hashes[row.chain_id_2])))
        seq_keys.append(key)
        labels_for_seq[key].add(int(row.label))
    amb_keys = {key for key, seen in labels_for_seq.items() if len(seen) > 1}
    amb_n = sum(key in amb_keys for key in seq_keys)
    add(rows, "label_conflict", "pairs_with_both_labels_on_same_sequences", amb_n, len(built), "pair")
    add(rows, "label_conflict", "distinct_sequence_pairs_with_both_labels", len(amb_keys), len(labels_for_seq), "sequence_pair")
    if "seq_ambiguous" in built.columns:
        flagged = int(built.seq_ambiguous.astype(bool).sum())
        recomputed = np.array([key in amb_keys for key in seq_keys])
        agree = int((recomputed == built.seq_ambiguous.astype(bool).to_numpy()).sum())
        add(rows, "label_conflict", "seq_ambiguous_flag_agrees", agree, len(built), "pair",
            note=f"flagged={flagged} recomputed={amb_n}")

    upair_labels = defaultdict(set)
    for row in built.itertuples(index=False):
        key = tuple(sorted((str(row.uniprot_id_1), str(row.uniprot_id_2))))
        upair_labels[key].add(int(row.label))
    both_u = {key for key, seen in upair_labels.items() if len(seen) > 1}
    u_pairs = sum(tuple(sorted((str(r.uniprot_id_1), str(r.uniprot_id_2)))) in both_u for r in built.itertuples(index=False))
    add(rows, "label_conflict", "pairs_whose_uniprot_pair_has_both_labels", u_pairs, len(built), "pair")
    add(rows, "label_conflict", "distinct_uniprot_pairs_with_both_labels", len(both_u), len(upair_labels), "uniprot_pair")
    same_u = int((built.uniprot_id_1.astype(str) == built.uniprot_id_2.astype(str)).sum())
    add(rows, "label_conflict", "same_uniprot_on_both_chains", same_u, len(built), "pair")
    zero_neg = int(((built.label == 0) & (built.no_of_contacts == 0)).sum())
    add(rows, "label_conflict", "zero_contact_negatives", zero_neg, int((built.label == 0).sum()), "pair")

    for name in ("train", "val", "test"):
        sub = built[built.split == name]
        add(rows, "split", f"{name}_rows", len(sub), len(built), "pair")
        add(rows, "split", f"{name}_positive", int((sub.label == 1).sum()), len(sub), "pair")
        ids = (DATA / "splits" / f"{name}.txt").read_text().split()
        add(rows, "split", f"{name}_file_matches_table", int(set(ids) == set(sub.pair_id.astype(str))), 1, "split")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        a = built[built.split == left]
        b = built[built.split == right]
        shared_u = (set(a.uniprot_id_1) | set(a.uniprot_id_2)) & (set(b.uniprot_id_1) | set(b.uniprot_id_2))
        shared_p = set(a.pdb_id) & set(b.pdb_id)
        add(rows, "split_leak", f"shared_uniprot_{left}_{right}", len(shared_u), 1, "uniprot")
        add(rows, "split_leak", f"shared_pdb_{left}_{right}", len(shared_p), 1, "pdb")

    # Per-split arrays for the figure, saved beside the metrics.
    built[["split", "label", "no_of_contacts", "len_1", "len_2", "seq_ambiguous"]].to_csv(
        OUT / "pair_slice.tsv", sep="\t", index=False
    )
    log("table counted", pairs=len(built), chains=len(chain_ids))
    extra = feature_scan(chain_ids, rows)
    letter_spotcheck(chain_ids, rows)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "coverage_metrics.csv", index=False)
    (OUT / "coverage_summary.json").write_text(json.dumps({"run_id": RUN_ID, **extra}, indent=2) + "\n")
    log(
        "run finished", run_id=RUN_ID, status="ok",
        duration_s=round(time.time() - started, 1),
        metrics=str(OUT / "coverage_metrics.csv"),
    )


if __name__ == "__main__":
    main()
