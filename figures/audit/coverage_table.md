Table 1. Coverage of the built pair table. Counts are from `figures/audit/coverage_metrics.csv` for run 2026-11-17T13-28-37Z.

| Check | n | Total | Rate |
| --- | ---: | ---: | ---: |
| source / positive_rows | 3153 | 3153 | 1.0000 |
| source / negative_rows | 5680 | 5680 | 1.0000 |
| conflict / same_chains_both_labels | 3 | 8833 | 0.0003 |
| built_table / rows | 8830 | 8833 | 0.9997 |
| built_table / positive | 3153 | 8830 | 0.3571 |
| sequence / chains_found_in_combined_seq | 11944 | 11944 | 1.0000 |
| sequence / aa_len_ne_combined_seq | 11469 | 17666 | 0.6492 |
| feature_presence / conservation | 11944 | 11944 | 1.0000 |
| length_match / conservation | 11944 | 11944 | 1.0000 |
| length_match / embedding | 11944 | 11944 | 1.0000 |
| length_match / motif | 11944 | 11944 | 1.0000 |
| conservation_source / onehot/v2 | 200 | 11944 | 0.0167 |
| motif_coverage / chains_with_any_fimo_hit | 11792 | 11944 | 0.9873 |
| label_conflict / pairs_with_both_labels_on_same_sequences | 1837 | 8830 | 0.2080 |
| label_conflict / pairs_whose_uniprot_pair_has_both_labels | 5871 | 8830 | 0.6649 |
| split / test_positive | 436 | 1325 | 0.3291 |
| split_leak / shared_uniprot_train_test | 0 | 1 | 0.0000 |
| index_check / query_letters_match_fasta | 12 | 12 | 1.0000 |

aa_len minus the Cα length was positive for 11469 of 17666 chain ends (median 2, maximum 1897). No chain end had a longer Cα sequence than aa_len.

The letter check is 12 chains, chosen across homolog-file sizes. It is not a check of all 11,944 chains. Length equality was counted on every chain.
