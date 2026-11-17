# Progress

This work lives in the PPI-Predictor repo (`ppi/` for code, `model_data/` for generated files). Source PDB tables and the Swiss-Prot database stay in `DEEPINTERACT/`.

## Done

- Canonical Cα sequences for 11,944 chains in `model_data/chains/`, taken from `combined_seq.fasta`.
- Conservation profiles for all 11,944 chains in `model_data/conservation/`. 127 chains had an empty homolog search and received a query-only profile (`conservation_empty_alignment.txt`).
- ESM-2 650M layer-33 embeddings for all 11,944 chains in `model_data/embeddings/`.
- Pair table `model_data/pairs.tsv`: 8,830 pairs (3,153 positive, 5,677 negative). Three negative rows that repeat a positive chain pair were removed. 924 negatives have the same two sequences as some positive pair.
- UniProt-component split in `model_data/splits/`: 6,181 train, 1,324 val, 1,325 test. No UniProt id and no PDB id is shared across splits.
- MEME and FIMO for all 11,944 chains in `model_data/meme/` and `model_data/fimo/`. This is not a refill of the old `Positive_data/hseq_1` and `fimo_1` folders.
- Motif channels (FIMO hit, -log10 p-value, PSSM score) in `model_data/motifs/`.
- Motif-free baseline weights: `model_data/baseline_linker.pt`.
- Motif-linker weights: `model_data/motif_linker.pt` (last epoch of the first run) and `model_data/motif_linker_best.pt` (epoch 2 of the second run, validation AUPRC 0.442).
- First test score, last epoch of the first run: AUROC 0.510, AUPRC 0.479. That is chance. Validation loss rose after epoch 3.

## Scored on 16 Nov 2026

`ppi/evaluate.py` scored both motif checkpoints. Temperature was fit on validation. The interval resamples UniProt components, 1,000 draws.

Last-epoch `motif_linker.pt`, test, after calibration: AUROC 0.510, AUPRC 0.479, ECE 0.028. The 95% interval is 0.490 to 0.529. That includes 0.50.

Best-epoch `motif_linker_best.pt`, test, after calibration: AUROC 0.503, AUPRC 0.469, ECE 0.028. The 95% interval is 0.482 to 0.524.

Identical-sequence negatives and the other negatives got the same mean probability (0.434 on the last-epoch model, 0.490 on the best epoch). The linker is not separating pairs.

## Homolog repair

245 chains had a MEME/FIMO result without a real homolog set: 127 had no homolog file, and 118 had only one sequence. Those motifs were discovered from the chain sequence alone. `ppi/repair_homolog_motifs.py` searches Swiss-Prot again, writes the homolog FASTA, and runs MEME then FIMO only when at least two sequences remain. If PHMMER still finds nothing, the motif channels are set to zero instead of a single-sequence motif. Log: `model_data/repair_homologs.log`.

## Mean-pool follow-up

`ppi/train_meanpool.py` pools each chain, then uses the sum, absolute difference, and product of those vectors. Same features as the linker. Stopped after epoch 3. Best validation AUPRC 0.450. Test AUROC **0.541**, test AUPRC **0.501**. Weights: `model_data/meanpool.pt`. Log: `model_data/train_meanpool.log`.

That is above the linker (test AUROC 0.510) and above chance (0.50). It is still a weak pair classifier. The base rate of positives on the test split is 0.46.

## Audit fixes, 17 Nov 2026

- Conservation rebuilt, index-aligned: 11,744 MSA profiles, 200 one-hot, 0 length mismatches against the chain, the embedding, and the motif array.
- Train-only scaler in `model_data/feature_scaler.npz`. Embeddings z-scored, PSSM log1p then z-scored, -log10 p capped at 10.
- Split repacked to positive rates 0.363 / 0.356 / 0.329. Still no shared UniProt or PDB. Details in `model_data/split_balance.md` and `model_data/stats.md`.
- 1,837 sequence-contradictory pairs flagged `seq_ambiguous`. They stay in the table, leave the training loss, and are reported separately from the primary test metric.
- Mean-pool retrain on the fixed features: `model_data/meanpool_scaled.pt`, log `model_data/train_meanpool_scaled.log`. Stopped at epoch 4, best validation epoch 2 (AUROC 0.553, AUPRC 0.367). Unambiguous test AUROC 0.537, AUPRC 0.345 (n=1113, positive rate 0.297). Full test AUROC 0.533, AUPRC 0.370 (n=1325, positive rate 0.329). Contradictory pairs AUROC 0.509 (n=212).

## Not started

- Readout comparison (top-k, noisy-OR, learned pool) and loss comparison (focal, subsampling).
- Temperature scaling was added in the evaluation script; it has not been reported yet.
- Structure download, rebuilt Cα contact maps, SaProt, DSSP/RSA, residue-pair contact loss.
- Pfam-family slices, held-out family test, comparison to other PPI models.
