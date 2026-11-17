# Split after label balancing

- Pairs: 8830  positive rate 0.3571
- Sequence pairs with both labels: 1837 (20.8%)
- Distinct contradictory sequence pairs: 905

- train: 6181 pairs (target 6181), positive 2246 rate 0.3634; unambiguous 4835 rate 0.3262; ambiguous 1346
- val: 1324 pairs (target 1324), positive 471 rate 0.3557; unambiguous 1045 rate 0.3177; ambiguous 279
- test: 1325 pairs (target 1325), positive 436 rate 0.3291; unambiguous 1113 rate 0.2974; ambiguous 212

Ambiguous pairs stay in the table. The training loss and the primary
test metric use the unambiguous pairs. The full test set is still scored.

