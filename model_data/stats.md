# Stage 1A stats

- Chains written: 11944
- Pairs: 8830 (positive 3153, negative 5677)
- Zero-contact negatives: 3015
- Negatives with a sequence pair identical to a positive: 924
- Chain length min/median/max: 18 / 179 / 4045
- Chains longer than 1022: 140
- Chains longer than 2048: 13
- Table contact density: 0.03618% (226984 / 627453881)

## Split

Components stay whole. The pack matches both the 70/15/15 sizes and the global positive rate (0.357). See `split_balance.md`.

- train: 6181 pairs, positive 2246 (rate 0.363). Unambiguous 4835, positive rate 0.326.
- val: 1324 pairs, positive 471 (rate 0.356). Unambiguous 1045, positive rate 0.318.
- test: 1325 pairs, positive 436 (rate 0.329). Unambiguous 1113, positive rate 0.297.
- Shared UniProt train∩val: 0
- Shared UniProt train∩test: 0
- Shared UniProt val∩test: 0
- Shared PDB train∩val: 0
- Shared PDB train∩test: 0
- Shared PDB val∩test: 0
- Sequence pairs that carry both labels: 1837 (20.8%), flagged `seq_ambiguous` and kept in the table.

The first pack filled train with the largest components and left the positive rates at 0.318 / 0.434 / 0.462.

## Features

Conservation was rebuilt from a query-anchored HMM alignment. 11,744 chains have an MSA profile and 200 chains with no homolog are one-hot. Profile length equals the chain length on all 11,944 chains, and it matches the embedding and motif lengths. Entropy is stored as a fraction of log2(21).

`feature_scaler.npz` was fit on the 6,931 unambiguous-train chains (1,553,985 residues). Embeddings are z-scored per dimension. The PSSM channel is log1p then z-scored (train mean 2.356, std 3.781). The -log10 p-value channel is capped at 10 and divided by 10.
