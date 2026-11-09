# nanochat autoresearch program

## Objective

Minimize `val_bpb` (validation bits-per-byte; **lower is better**) printed by
`train.py`. This is a faithful, single-GPU character-level GPT — the real thing,
not a stand-in. Only `train.py` is mutable; `prepare.py` (the data half) is fixed.

## The loop (never stop until the budget is exhausted)

1. `read_file(train.py)` and study the `HYPERPARAMETERS` block and the model.
2. Form a hypothesis for a change that should lower `val_bpb` — e.g. adjust
   `LEARNING_RATE`, `N_LAYER`/`N_EMBD`/`N_HEAD`, `BLOCK_SIZE`, `DROPOUT`,
   `WEIGHT_DECAY`, the warmup/cosine schedule, or the architecture itself.
3. `write_file(train.py, ...)` with the edit.
4. `run()` — read back `score` (parsed `val_bpb`), `status`, and `memory_gb`.
   Watch `memory_gb`: a change that OOMs shows up as `status = "crash"`.
5. **Keep-if-improved-else-reset:**
   - If `status == "ok"` and `score` is lower than the best score so far →
     `keep()` (advance the git ledger) and record the new best.
   - Otherwise (worse, crash, or timeout) → `reset()` to discard the change.
6. `ledger(commit, score, memory_gb, status, description)` — append one row per
   attempt so the run is auditable.
7. Return to step 2 with a new hypothesis.

## Rules

- Change one axis at a time so each attempt is a clean signal.
- Respect the wall-clock budget: a larger model may not finish; a `timeout`
  status means the change was too expensive — reset and scale back.
- Requires a single GPU (CUDA or Apple MPS). `train.py` auto-runs `prepare.py`
  on first launch to build the corpus.
- Never stop iterating on your own — keep proposing and testing improvements
  until the budget ends.
