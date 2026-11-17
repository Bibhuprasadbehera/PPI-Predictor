# Corners cut

These items are not in Stage 1A. Each line says what is missing and what closes it.

- Structures for 2,876 PDB ids are not downloaded. Contact matrices, DSSP/RSA, and SaProt 3Di wait on that download. Closes in Iteration 2.
- Positive distance matrices in `dataset/positive/distM` are null bytes. Readable negative matrices are not training labels, because some of them do not match first-model Cα lengths. Closes when contact matrices are rebuilt from structures.
- Stage 1B is the motif-free baseline: ESM-2, conservation, and Pfam. Motif channels are zero. The motif model is Stage 1C, after MEME, FIMO, and PSSM scores exist for the chains in that run.
- MEME uses the full homolog set. `-maxsize 20000000` is set so large files are not rejected. Homologs are not truncated.
- Cross-accession sequence-identity clustering is not a split. It is an OOD check for a later iteration.
- Chain order is fixed by the symmetric linker. A one-order ablation is not in Stage 1A.
- Loss, readout (`k` in top-k LogSumExp, noisy-OR, learned pool), and subsampling are compared only after a model exists, one change at a time.

Correction, 17 Nov 2026: motif channels are not zero. MEME, FIMO, and the three motif channels exist for every chain. Conservation was rebuilt so it is index-aligned, and features are scaled with a train-only scaler. The line above about a motif-free Stage 1B describes the original cut, not the current features.
