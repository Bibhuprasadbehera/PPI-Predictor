"""DSSP and Cα contacts on the canonical chain sequences.

Structures are read from model_data/structures/{pdb}.cif.gz or, when that
file was not needed, from the existing Positive_data/pdb/{pdb}.pdb. DSSP is
parsed from mkdssp text. Residue features are written only when every FASTA
residue maps onto a Cα, so a short profile is never padded out to the chain
length. Contact lists use that same index. pairs.tsv is read and not written.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.Data.PDBData import residue_sasa_scales
from Bio.PDB import FastMMCIFParser, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

ROOT = Path(__file__).resolve().parents[1]
DEEP = Path("/home/bibhu/Documents/DeepInteract/DEEPINTERACT")
PAIRS = ROOT / "model_data" / "pairs.tsv"
CHAINS = ROOT / "model_data" / "chains"
PDB_DIR = DEEP / "Positive_data" / "pdb"
STRUCTURES = ROOT / "model_data" / "structures"
DSSP_DIR = ROOT / "model_data" / "dssp"
CONTACT_DIR = ROOT / "model_data" / "contacts"
LOG = ROOT / "model_data" / "structure_features.log"
FAILURES = ROOT / "model_data" / "dssp_failures.tsv"
DISAGREE = ROOT / "model_data" / "contact_disagreement.tsv"
METRICS = ROOT / "model_data" / "structure_metrics.csv"
CONTACT_A = 8.0
SANDER = residue_sasa_scales["Sander"]
SS3 = str.maketrans("HGIEBTS-", "HHHEECCC")

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K", "SEP": "S", "TPO": "T",
    "PTR": "Y", "MLY": "K", "MLZ": "K", "HYP": "P", "DAL": "A",
    "CSO": "C", "CME": "C", "CSD": "C", "OCS": "C", "KCX": "K",
    "LLP": "K", "CSS": "C", "CSX": "C", "CAS": "C", "FME": "M",
    "PCA": "E", "SAC": "S", "UNK": "X",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(event: str, **fields) -> None:
    parts = [f"INFO  {event}"]
    for key, value in fields.items():
        text = str(value).replace("\n", " ")
        parts.append(f"{key}={text}")
    line = " ".join(parts)
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def read_seq(path: Path) -> str:
    return "".join(
        line.strip() for line in path.read_text().splitlines() if line and not line.startswith(">")
    )


def valid_cif_gz(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 64:
        return False
    try:
        with gzip.open(path, "rb") as handle:
            head = handle.read(200)
    except (OSError, EOFError, gzip.BadGzipFile):
        return False
    return b"data_" in head or b"loop_" in head


def local_pdb(pdb_id: str) -> Path | None:
    for name in (f"{pdb_id}.pdb", f"{pdb_id.upper()}.pdb", f"{pdb_id.lower()}.pdb"):
        path = PDB_DIR / name
        if path.exists() and path.stat().st_size > 1000:
            return path
    return None


def structure_file(pdb_id: str) -> Path | None:
    cif = STRUCTURES / f"{pdb_id}.cif.gz"
    if valid_cif_gz(cif):
        return cif
    return local_pdb(pdb_id)


def map_onto(target: str, query: str) -> list[int] | None:
    """Index of each target residue in query. None when a target residue is gapped."""
    if target == query:
        return list(range(len(target)))
    if not target or not query:
        return None
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(target, query)[0]
    mapped = [-1] * len(target)
    blocks = alignment.aligned
    for (start_t, end_t), (start_q, end_q) in zip(blocks[0], blocks[1]):
        span = int(end_t - start_t)
        if int(end_q - start_q) != span:
            return None
        for offset in range(span):
            mapped[int(start_t) + offset] = int(start_q) + offset
    if any(index < 0 for index in mapped):
        return None
    return mapped


def _self_check() -> None:
    assert map_onto("ABC", "ABC") == [0, 1, 2]
    assert map_onto("ABC", "XABC") == [1, 2, 3]
    assert map_onto("ACD", "ABD") == [0, 1, 2]
    assert map_onto("ABC", "AB") is None


MODIFIED = {
    "MSE", "SEC", "PYL", "SEP", "TPO", "PTR", "MLY", "MLZ", "HYP", "DAL",
    "CSO", "CME", "CSD", "OCS", "KCX", "LLP", "CSS", "CSX", "CAS", "FME",
    "PCA", "SAC",
}


def letter_of(resname: str, hetflag: str) -> str | None:
    if resname in AA3 and (hetflag == " " or resname in MODIFIED):
        return AA3[resname]
    if hetflag == " ":
        return "X"
    return None


def select_ca(residue):
    if residue.is_disordered() == 2:
        chosen = None
        for name in residue.disordered_get_id_list():
            alt = residue.disordered_get(name)
            if "CA" not in alt:
                continue
            atom = alt["CA"]
            altloc = atom.get_altloc() if not atom.is_disordered() else " "
            if altloc in (" ", "A", "1"):
                chosen = name
                break
        if chosen is not None:
            residue.disordered_select(chosen)
    if "CA" not in residue:
        return None
    atom = residue["CA"]
    if atom.is_disordered():
        ids = list(atom.disordered_get_id_list())
        picked = None
        for pref in (" ", "A", "1"):
            if pref in ids:
                picked = atom.disordered_get(pref)
                break
        atom = picked if picked is not None else atom.disordered_get(ids[0])
    element = (atom.element or "").upper()
    if element and element != "C":
        return None
    return atom


def extract_ca(structure) -> dict[str, list[dict]]:
    model = structure[0]
    chains = {}
    for chain in model:
        rows = []
        seen = set()
        for residue in chain:
            hetflag, resseq, icode = residue.id
            resname = residue.get_resname().strip()
            if hetflag == "W" or resname in {"HOH", "WAT", "DOD"}:
                continue
            aa = letter_of(resname, hetflag)
            if aa is None:
                continue
            atom = select_ca(residue)
            if atom is None:
                continue
            icode = icode if icode else " "
            key = (int(resseq), icode)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "resseq": key[0],
                    "icode": icode,
                    "aa": aa,
                    "resname": resname,
                    "xyz": np.asarray(atom.get_coord(), dtype=np.float64),
                }
            )
        if rows:
            chains[str(chain.id)] = rows
    return chains


def open_structure(path: Path, pdb_id: str, work: Path) -> tuple[object, Path]:
    if path.name.endswith(".cif.gz"):
        cif_path = work / f"{pdb_id}.cif"
        with gzip.open(path, "rb") as src, cif_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        parser = FastMMCIFParser(auth_chains=True, auth_residues=True, QUIET=True)
        return parser.get_structure(pdb_id, str(cif_path)), cif_path
    parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_id, str(path)), path


def _parse_dssp_line(line: str) -> dict | None:
    if len(line) < 16 or line[9] == " ":
        return None
    shift = 0
    try:
        acc = int(line[34:38])
        phi = float(line[103:109])
        psi = float(line[109:115])
    except ValueError:
        if len(line) <= 34 or line[34] == " ":
            return None
        shift = line[34:].find(" ")
        if shift < 0:
            return None
        try:
            acc = int(line[34 + shift : 38 + shift])
            phi = float(line[103 + shift : 109 + shift])
            psi = float(line[109 + shift : 115 + shift])
        except ValueError:
            return None
    aa = line[13]
    if aa in ("!", "*"):
        return None
    ss = line[16]
    if ss == " ":
        ss = "-"
    try:
        resseq = int(line[5:10])
    except ValueError:
        return None
    return {
        "chain": line[11],
        "resseq": resseq,
        "icode": line[10] if line[10] else " ",
        "aa": "C" if aa.islower() else aa,
        "ss": ss,
        "acc": acc,
        "phi": phi,
        "psi": psi,
    }


def parse_dssp(text: str) -> dict[str, list[dict]]:
    started = False
    by_chain: dict[str, list[dict]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[1] == "RESIDUE":
            started = True
            continue
        if not started:
            continue
        rec = _parse_dssp_line(line)
        if rec is None:
            continue
        by_chain.setdefault(rec["chain"], []).append(rec)
    return by_chain


# ppi/bin/mkdssp is the conda-forge 4.4.11 binary with its data-directory
# string pointed at mmcif_pdbx.dic. components.cif sits in the same directory
# and the process cwd is that directory, because this build otherwise opens
# the dictionary directory as a file.
MKDSSP = ROOT / "ppi" / "bin" / "mkdssp"


def run_mkdssp(path: Path) -> str:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    binary = MKDSSP if MKDSSP.exists() else Path(shutil.which("mkdssp") or "mkdssp")
    proc = subprocess.run(
        [str(binary), "--output-format=dssp", str(path)],
        capture_output=True,
        text=True,
        env=env,
        cwd=binary.parent,
        timeout=900,
        check=False,
    )
    if not any(line.split()[1:2] == ["RESIDUE"] for line in proc.stdout.splitlines()):
        detail = (proc.stderr or proc.stdout or "no output").strip().splitlines()
        raise RuntimeError(detail[-1][:300] if detail else "mkdssp produced no DSSP table")
    return proc.stdout


def attach_dssp(residues: list[dict], by_chain: dict[str, list[dict]], chain_id: str) -> tuple[list, int]:
    """DSSP record or None for each Cα, plus the number of hits."""
    keys = [(row["resseq"], row["icode"]) for row in residues]

    def from_lookup(lookup: dict) -> list:
        return [lookup.get(key) for key in keys]

    own = {(rec["resseq"], rec["icode"]): rec for rec in by_chain.get(chain_id, [])}
    own_hits = sum(record is not None for record in from_lookup(own))
    if residues and own_hits >= 0.8 * len(residues):
        return from_lookup(own), own_hits

    best_lookup = own
    best_hits = own_hits
    for cid, records in by_chain.items():
        lookup = {(rec["resseq"], rec["icode"]): rec for rec in records}
        hits = sum(key in lookup for key in keys)
        if hits > best_hits:
            best_lookup, best_hits = lookup, hits
    if residues and best_hits >= 0.8 * len(residues):
        return from_lookup(best_lookup), best_hits

    ca_seq = "".join(row["aa"] for row in residues)
    best_rows = None
    best_n = -1
    for records in by_chain.values():
        dssp_seq = "".join(rec["aa"] for rec in records)
        mapped = map_onto(ca_seq, dssp_seq)
        if mapped is None:
            continue
        rows = [records[index] for index in mapped]
        if len(rows) > best_n:
            best_rows, best_n = rows, len(rows)
    if best_rows is not None:
        return best_rows, best_n
    return [None] * len(residues), 0


def relative_acc(acc: int, resname: str) -> float:
    maximum = SANDER.get(resname)
    if not maximum:
        return np.nan
    return float(min(1.0, acc / maximum))


def build_chain(chain_id: str, fasta: str, residues: list[dict], by_chain: dict[str, list[dict]]) -> tuple[dict | None, dict]:
    ca_seq = "".join(row["aa"] for row in residues) if residues else ""
    info = {
        "chain_id": chain_id,
        "fasta_len": len(fasta),
        "ca_len": len(residues),
        "n_matched": 0,
        "dssp_hits": 0,
        "reason": "",
    }
    mapped = map_onto(fasta, ca_seq)
    if mapped is None:
        info["reason"] = "fasta_not_covered"
        info["n_matched"] = 0 if not residues else _covered(fasta, ca_seq)
        return None, info
    info["n_matched"] = len(mapped)
    chosen = [residues[index] for index in mapped]
    dssp_rows, hits = attach_dssp(chosen, by_chain, chain_id)
    # chain_id here is the author chain id, not pdb_chain.
    info["dssp_hits"] = hits
    if chosen and hits < 0.8 * len(chosen):
        info["reason"] = "dssp_coverage"
        return None, info
    length = len(fasta)
    ss8 = []
    ss3 = []
    rsa = np.empty(length, dtype=np.float32)
    phi = np.empty(length, dtype=np.float32)
    psi = np.empty(length, dtype=np.float32)
    resseq = np.empty(length, dtype=np.int32)
    icode = []
    ca = np.empty((length, 3), dtype=np.float32)
    for i, (residue, rec) in enumerate(zip(chosen, dssp_rows)):
        resseq[i] = residue["resseq"]
        icode.append(residue["icode"])
        ca[i] = residue["xyz"]
        if rec is None:
            ss8.append("-")
            ss3.append("-")
            rsa[i] = np.nan
            phi[i] = np.nan
            psi[i] = np.nan
        else:
            ss = rec["ss"] if rec["ss"] else "-"
            ss8.append(ss)
            ss3.append(ss.translate(SS3))
            rsa[i] = relative_acc(rec["acc"], residue["resname"])
            phi[i] = rec["phi"]
            psi[i] = rec["psi"]
    if len(ss8) != length:
        info["reason"] = "length_mismatch"
        return None, info
    info["reason"] = "exact" if ca_seq == fasta else "aligned"
    return {
        "ss8": np.array(ss8),
        "ss3": np.array(ss3),
        "rsa": rsa,
        "phi": phi,
        "psi": psi,
        "resseq": resseq,
        "icode": np.array(icode),
        "ca": ca,
        "length": np.int32(length),
        "exact": np.int8(ca_seq == fasta),
    }, info


def _covered(fasta: str, ca_seq: str) -> int:
    mapped = [-1] * len(fasta)
    if not fasta or not ca_seq:
        return 0
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(fasta, ca_seq)[0]
    for (start_t, end_t), _query in zip(alignment.aligned[0], alignment.aligned[1]):
        for index in range(int(start_t), int(end_t)):
            mapped[index] = 1
    return sum(index >= 0 for index in mapped)


def npz_ok(path: Path, length: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                int(data["length"]) == length
                and data["rsa"].shape == (length,)
                and data["ss8"].shape == (length,)
                and data["ca"].shape == (length, 3)
            )
    except Exception:
        return False


def save_npz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def contact_pairs(ca_1: np.ndarray, ca_2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    cols = []
    limit = CONTACT_A * CONTACT_A
    other = np.asarray(ca_2, dtype=np.float64)
    for start in range(0, len(ca_1), 512):
        block = np.asarray(ca_1[start : start + 512], dtype=np.float64)
        dist2 = np.sum((block[:, None, :] - other[None, :, :]) ** 2, axis=2)
        ii, jj = np.where(dist2 <= limit)
        if len(ii):
            rows.append(ii + start)
            cols.append(jj)
    if not rows:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    return np.concatenate(rows).astype(np.int32), np.concatenate(cols).astype(np.int32)


def process_pdb(task: tuple) -> dict:
    pdb_id, chain_names, pair_rows = task
    chain_results = []
    contact_results = []
    path = structure_file(pdb_id)
    needed = []
    sequences = {}
    for name in chain_names:
        chain_key = f"{pdb_id}_{name}"
        fasta_path = CHAINS / f"{chain_key}.fasta"
        if not fasta_path.exists():
            chain_results.append(
                {"chain_id": chain_key, "fasta_len": 0, "ca_len": 0, "n_matched": 0, "dssp_hits": 0, "reason": "fasta_missing"}
            )
            continue
        fasta = read_seq(fasta_path)
        sequences[name] = fasta
        if npz_ok(DSSP_DIR / f"{chain_key}.npz", len(fasta)):
            chain_results.append(
                {"chain_id": chain_key, "fasta_len": len(fasta), "ca_len": len(fasta), "n_matched": len(fasta), "dssp_hits": len(fasta), "reason": "present"}
            )
        else:
            needed.append(name)
    pending_pairs = []
    for row in pair_rows:
        contact_path = CONTACT_DIR / f"{row['pair_id']}.npz"
        if contact_path.exists() and contact_path.stat().st_size > 0:
            contact_results.append({"pair_id": row["pair_id"], "status": "present"})
        else:
            pending_pairs.append(row)
    if path is None:
        for name in needed:
            fasta = sequences[name]
            chain_results.append(
                {
                    "chain_id": f"{pdb_id}_{name}",
                    "fasta_len": len(fasta),
                    "ca_len": 0,
                    "n_matched": 0,
                    "dssp_hits": 0,
                    "reason": "structure_missing",
                }
            )
        for row in pending_pairs:
            contact_results.append({"pair_id": row["pair_id"], "status": "structure_missing"})
        return {"pdb_id": pdb_id, "chains": chain_results, "contacts": contact_results}
    if not needed and not pending_pairs:
        return {"pdb_id": pdb_id, "chains": chain_results, "contacts": contact_results}

    work = Path(tempfile.mkdtemp(prefix=f"dssp_{pdb_id}_"))
    try:
        structure, dssp_input = open_structure(path, pdb_id, work)
        residues = extract_ca(structure)
        raw_path = ROOT / "model_data" / "dssp_raw" / f"{pdb_id}.dssp"
        dssp_text = raw_path.read_text() if raw_path.exists() else ""
        if "RESIDUE" not in dssp_text:
            dssp_text = run_mkdssp(dssp_input)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(dssp_text)
        by_chain = parse_dssp(dssp_text)
        written = {}
        for name in needed:
            chain_key = f"{pdb_id}_{name}"
            payload, info = build_chain(name, sequences[name], residues.get(name, []), by_chain)
            info["chain_id"] = chain_key
            if payload is None:
                chain_results.append(info)
                continue
            save_npz(DSSP_DIR / f"{chain_key}.npz", payload)
            written[name] = payload["ca"]
            chain_results.append(info)
        for row in pending_pairs:
            c1, c2 = row["chain_1"], row["chain_2"]
            ca_1 = written.get(c1)
            ca_2 = written.get(c2)
            if ca_1 is None:
                ca_1 = _load_ca(pdb_id, c1, sequences.get(c1))
            if ca_2 is None:
                ca_2 = _load_ca(pdb_id, c2, sequences.get(c2))
            if ca_1 is None or ca_2 is None:
                contact_results.append({"pair_id": row["pair_id"], "status": "chain_unmapped"})
                continue
            rr, cc = contact_pairs(ca_1, ca_2)
            save_npz(
                CONTACT_DIR / f"{row['pair_id']}.npz",
                {"row": rr, "col": cc, "count": np.int32(len(rr))},
            )
            contact_results.append({"pair_id": row["pair_id"], "status": "written", "count": int(len(rr))})
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"[:300]
        recorded = {row["chain_id"] for row in chain_results}
        for name in needed:
            chain_key = f"{pdb_id}_{name}"
            if chain_key not in recorded:
                chain_results.append(
                    {
                        "chain_id": chain_key,
                        "fasta_len": len(sequences.get(name, "")),
                        "ca_len": 0,
                        "n_matched": 0,
                        "dssp_hits": 0,
                        "reason": reason,
                    }
                )
        recorded_pairs = {item["pair_id"] for item in contact_results}
        for row in pending_pairs:
            if row["pair_id"] not in recorded_pairs:
                contact_results.append({"pair_id": row["pair_id"], "status": reason[:180]})
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"pdb_id": pdb_id, "chains": chain_results, "contacts": contact_results}


def _load_ca(pdb_id: str, chain: str, fasta: str | None) -> np.ndarray | None:
    if fasta is None:
        path = CHAINS / f"{pdb_id}_{chain}.fasta"
        if not path.exists():
            return None
        fasta = read_seq(path)
    npz = DSSP_DIR / f"{pdb_id}_{chain}.npz"
    if not npz_ok(npz, len(fasta)):
        return None
    with np.load(npz, allow_pickle=False) as data:
        return data["ca"]


def git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
        return commit, dirty
    except Exception:
        return "none", True


def write_reports(pairs: pd.DataFrame, chain_rows: list[dict]) -> None:
    fail = pd.DataFrame(chain_rows)
    if fail.empty:
        fail = pd.DataFrame(columns=["chain_id", "fasta_len", "ca_len", "n_matched", "dssp_hits", "reason"])
    fail = fail.drop_duplicates("chain_id", keep="last")
    reasons = dict(zip(fail.chain_id, fail.reason))
    failed = fail[~fail.reason.isin(["present", "exact", "aligned"])]
    failed.to_csv(FAILURES, sep="\t", index=False)

    disagree_rows = []
    equal = 0
    differ = 0
    compared = 0
    flips = 0
    deltas = []
    for row in pairs.itertuples(index=False):
        path = CONTACT_DIR / f"{row.pair_id}.npz"
        table_count = float(row.no_of_contacts)
        if path.exists() and path.stat().st_size > 0:
            with np.load(path, allow_pickle=False) as data:
                recomputed = int(data["count"])
            delta = recomputed - table_count
            status = "compared"
            compared += 1
            deltas.append(abs(delta))
            if recomputed == table_count:
                equal += 1
            else:
                differ += 1
            if (recomputed >= 20) != (int(row.label) == 1):
                flips += 1
        else:
            recomputed = np.nan
            delta = np.nan
            c1 = reasons.get(row.chain_id_1, "missing")
            c2 = reasons.get(row.chain_id_2, "missing")
            status = c1 if c1 == c2 else f"{c1}|{c2}"
        disagree_rows.append(
            {
                "pair_id": row.pair_id,
                "pdb_id": row.pdb_id,
                "chain_1": row.chain_1,
                "chain_2": row.chain_2,
                "label": int(row.label),
                "split": row.split,
                "no_of_contacts": table_count,
                "recomputed_contacts": recomputed,
                "delta": delta,
                "status": status,
            }
        )
    disagree = pd.DataFrame(disagree_rows)
    disagree.to_csv(DISAGREE, sep="\t", index=False)
    chain_ok = int(fail.reason.isin(["present", "exact", "aligned"]).sum()) if len(fail) else 0
    metrics = pd.DataFrame(
        [
            {"item": "chains_written", "count": chain_ok, "total": int(fail.chain_id.nunique()) if len(fail) else 0},
            {"item": "chains_failed", "count": int(len(failed)), "total": int(fail.chain_id.nunique()) if len(fail) else 0},
            {"item": "pairs_compared", "count": compared, "total": len(pairs)},
            {"item": "pairs_count_equal", "count": equal, "total": compared},
            {"item": "pairs_count_differ", "count": differ, "total": compared},
            {"item": "pairs_label_would_flip", "count": flips, "total": compared},
            {
                "item": "median_abs_delta",
                "count": float(np.median(deltas)) if deltas else 0,
                "total": compared,
            },
        ]
    )
    metrics.to_csv(METRICS, index=False)
    log(
        "reports written",
        failures=str(FAILURES),
        disagreement=str(DISAGREE),
        metrics=str(METRICS),
        chains_written=chain_ok,
        chains_failed=int(len(failed)),
        pairs_compared=compared,
        pairs_equal=equal,
        pairs_differ=differ,
        label_would_flip=flips,
        median_abs_delta=f"{(float(np.median(deltas)) if deltas else 0):.4f}",
    )


def main() -> None:
    _self_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pdb", action="append", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    binary = MKDSSP if MKDSSP.exists() else Path(shutil.which("mkdssp") or "")
    if not binary or not binary.exists():
        raise SystemExit("mkdssp is not available")
    if not (binary.parent / "components.cif").exists():
        raise SystemExit(f"components.cif is missing next to {binary}")
    version = subprocess.check_output([str(binary), "--version"], text=True).strip().splitlines()[0]
    pairs = pd.read_csv(PAIRS, sep="\t")
    pairs_sha = hashlib.sha256(PAIRS.read_bytes()).hexdigest()[:12]
    if args.pdb:
        wanted = set(args.pdb)
        pairs = pairs[pairs.pdb_id.isin(wanted)].copy()
    jobs = []
    for pdb_id, group in pairs.groupby("pdb_id", sort=True):
        names = sorted(set(group.chain_1.astype(str)) | set(group.chain_2.astype(str)))
        rows = group[["pair_id", "chain_1", "chain_2", "no_of_contacts"]].to_dict("records")
        jobs.append((str(pdb_id), names, rows))
    if args.limit:
        jobs = jobs[: args.limit]
    commit, dirty = git_state()
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    log(
        "run started",
        run_id=run_id,
        pipeline="structure_features",
        utc_start=utc_now(),
        script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12],
        commit=commit,
        dirty=str(dirty).lower(),
        pairs_sha=pairs_sha,
        python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        mkdssp=version,
        workers=args.workers,
        jobs=len(jobs),
        contact_A=CONTACT_A,
        rsa_scale="Sander",
    )
    started = time_start()
    chain_rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_pdb, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            chain_rows.extend(result["chains"])
            done += 1
            if done % 25 == 0 or done == len(jobs):
                written = sum(row["reason"] in {"exact", "aligned", "present"} for row in chain_rows)
                log("feature progress", done=done, total=len(jobs), chains_ok=written, chains_seen=len(chain_rows))
    if hashlib.sha256(PAIRS.read_bytes()).hexdigest()[:12] != pairs_sha:
        raise SystemExit("pairs.tsv changed while structures were built")
    if args.pdb or args.limit:
        log("run finished", status="partial", utc_end=utc_now(), duration_s=f"{time_start() - started:.1f}", note="subset")
        fail = pd.DataFrame(chain_rows).drop_duplicates("chain_id", keep="last")
        fail.to_csv(FAILURES, sep="\t", index=False)
        return
    # Rebuild chain status from this run plus files already present.
    write_reports(pd.read_csv(PAIRS, sep="\t"), chain_rows)
    log("run finished", status="ok", utc_end=utc_now(), duration_s=f"{time_start() - started:.1f}")


def time_start() -> float:
    return datetime.now(timezone.utc).timestamp()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("run failed", error=traceback.format_exc().replace("\n", " | "))
        raise
