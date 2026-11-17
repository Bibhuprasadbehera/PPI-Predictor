# Audit, 17 Nov 2026

Checked the data, features, training, and evaluation against the plan. The near-chance scores are partly a method problem and partly data bugs. The bugs below are ordered by how much they affect the result.

## Bugs that corrupt training

### 1. Conservation profiles are misaligned on 3,031 of 11,944 chains
The Dataset concatenates embedding, conservation, and motif per residue, then trims to the shortest. For 3,031 chains the conservation length does not equal the chain length (1,924 shorter, 1,107 longer; median gap 6, 95th percentile 286, max 1,897). `hmmalign --trim` and `phmmer -A` focus-trimming keep match columns, which is not the query length. So on a quarter of chains, conservation row i is not residue i, and the trim silently shifts or drops residues. The plan's whole point was index-aligned features. This channel is wrong for 25% of chains.

### 2. Feature channels are never standardized (preprocessing was skipped)
Motif PSSM values reach ~20,000 and the -log10 p-value channel reaches ~68, while ESM embeddings are ~unit scale and conservation is 0–1. They are concatenated with no per-feature normalization. The single input LayerNorm cannot fix a 20,000:1 scale gap; the huge raw channels dominate the projection. There is no mean/variance standardization, no clipping, no log transform on the PSSM channel.

### 3. Train and test class balance differ a lot
Positive rate is 0.318 in train, 0.434 in val, 0.462 in test. The greedy component packing filled train first with the largest components, and those are more negative than average. The model trains under one prior and is scored under another, and the fixed `pos_weight` is computed from the train prior only. The split is leakage-free but not stratified.

## Data ceilings the model cannot beat

### 4. The same sequence pair carries both labels
20.8% of pairs (1,837) have an identical (seq1, seq2) pair that also appears with the opposite label, from distal chain copies in the same complex. A sequence-only model must give identical inputs identical scores, so at least 905 pairs (10.2% of the set) are guaranteed wrong. That caps achievable AUROC well below 1. The 924 "identical to a positive" negatives flagged earlier were only one side of this.

### 5. Same-protein pairs dominate both classes
66.5% of pairs share a UniProt pair that appears with both labels. Whether two specific chains touch is largely a property of the quaternary arrangement, not of the two sequences. A sequence-pair model has limited information for this label as defined.

## Where we are against the plan

Stage 1A, done and correct: canonical Cα chains, the pair table, the leakage-free component split, cut.md.

Stage 1A, conservation: done but bug 1 makes it unreliable.

Stage 1B and 1C motifs: MEME, FIMO, and the three motif channels exist for all chains (after the 245-chain repair). Channel scaling is bug 2.

Stage 1B/1C training: ran, but every model is at chance (test AUROC 0.50–0.53) and the plan's method comparisons were not done: no top-k vs noisy-OR vs learned-pool sweep, no focal vs subsampling, no stratified handling, no per-family or OOD slice. Calibration and the cluster bootstrap were added.

Not started: Iteration 2 (structure download, Cα contact maps, SaProt, residue-pair contact loss), and the identical-sequence diagnostic slice was counted but never scored.

## Fix order

1. Rebuild conservation so row i is residue i (fill query-gap columns, or recompute from a query-anchored alignment), then assert every feature length equals the chain length before training.
2. Standardize features: z-score the embeddings, keep conservation 0–1, log1p then z-score the PSSM channel, cap -log10 p. Persist the scaler from train only.
3. Stratify the split by label within components, or at least recompute pos_weight and report per-split prior.
4. Decide the label question: either accept the 10% sequence-contradiction ceiling and report against it, or move toward the residue-pair contact target in Iteration 2 where the structure disambiguates the copies.

## Fixes applied, 17 Nov 2026

1. Conservation was rebuilt. The query is aligned to its own HMM with the homologs, under a header that cannot collide with a homolog id, and only columns where that query residue is present are kept. Row i is residue i, including query inserts. 11,744 chains use the alignment. 200 chains with no homolog are one-hot. A full pass found 0 chains where conservation, embedding, or motif length differs from the chain length. Entropy is divided by log2(21), so conservation stays in [0, 1].

2. `model_data/feature_scaler.npz` was fit on unambiguous train chains only (6,931 chains, 1,553,985 residues). Embeddings are z-scored per dimension. Conservation is left in [0, 1]. The PSSM channel is log1p then z-scored. -log10 p is capped at 10 and scaled to [0, 1]. The dataset asserts equal lengths and no longer trims to the shortest array.

3. The component split was repacked. Sizes are still 6,181 / 1,324 / 1,325, with no shared UniProt id and no shared PDB id. Positive rates are 0.363 / 0.356 / 0.329 (global 0.357). On the unambiguous pairs used for the loss they are 0.326 / 0.318 / 0.297. `pos_weight` is taken from the unambiguous train prior.

4. The 1,837 pairs whose (seq1, seq2) also appears with the opposite label are flagged `seq_ambiguous` and kept. They are out of the training loss and out of the primary test metric. The full test set is still scored, so the ceiling stays visible. Different-chain pairs that share a UniProt pair were not dropped. The residue-pair contact target still needs structures, which stays deferred.

The mean-pool model was retrained on these features (`meanpool_scaled.pt`). Best validation epoch was epoch 2 (AUROC 0.553, AUPRC 0.367, val positive rate 0.318). Test, unambiguous pairs: AUROC 0.537, AUPRC 0.345 on 1,113 pairs (positive rate 0.297). Test, all pairs: AUROC 0.533, AUPRC 0.370 on 1,325 pairs (positive rate 0.329). The contradictory slice scores at chance (AUROC 0.509, n=212), which is what a sequence-only model has to do on that slice. The lift over the positive rate is real and small. Bugs 1–3 are no longer in the features. Bug 5 is unchanged: most pairs still share a UniProt pair that carries both labels, and the two sequences do not record which copies touch.
