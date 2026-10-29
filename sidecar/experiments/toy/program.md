# Toy autoresearch program

## Objective

Minimize `val_bpb` (bits-per-byte; **lower is better**) printed by `train.py`.
This is a zero-setup, CPU-only stand-in for a real training run — it exists so
the autoresearch loop works on any machine with no GPU and no dependencies.

## The loop (never stop until the budget is exhausted)

1. `read_file(train.py)` and study the `HYPERPARAMETERS` block.
2. Form a hypothesis for a single, small change that should lower `val_bpb`
   (e.g. raise `LEARNING_RATE`, add `MOMENTUM`, increase `STEPS`).
3. `write_file(train.py, ...)` with the edit.
4. `run()` — read back `score` (the parsed `val_bpb`), `status`, and `memory_gb`.
5. **Keep-if-improved-else-reset:**
   - If `status == "ok"` and `score` is lower than the best score so far →
     `keep()` (advance the git ledger) and record it as the new best.
   - Otherwise → `reset()` to discard the change and return to the best state.
6. `ledger(commit, score, memory_gb, status, description)` — append one row
   describing the attempt and its outcome.
7. Go back to step 2 with a new hypothesis.

## Rules

- Change **one thing at a time** so each attempt is a clean signal.
- A too-large `LEARNING_RATE` diverges (poor/large `val_bpb`) — that is real
  signal, not a bug; reset and try a smaller step.
- Only `train.py` is mutable.
- Never stop iterating on your own — keep proposing and testing improvements
  until the wall-clock budget ends.
