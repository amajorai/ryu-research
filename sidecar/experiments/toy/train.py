"""Toy autoresearch objective — pure Python, no dependencies, no GPU.

Stands in for a real training run: it optimizes a smooth loss surface with a
handful of hyperparameters and prints a single scalar `val_bpb` (lower = better).
The autoresearch agent edits the HYPERPARAMETERS block below, re-runs, and keeps
the change only when `val_bpb` drops — exactly the loop the real `nanochat`
experiment follows, but finishing in seconds on any machine.

The surface is deterministic, so an improvement is a real improvement, never
noise. Tune LEARNING_RATE / STEPS / MOMENTUM (and DIM) to converge faster/closer.
"""

import math

# ── HYPERPARAMETERS (the agent edits these) ──────────────────────────────────
LEARNING_RATE = 0.002
STEPS = 150
MOMENTUM = 0.0
DIM = 24
# ─────────────────────────────────────────────────────────────────────────────


def target(i: int) -> float:
    """A fixed, irregular target vector with real magnitude — the optimum to
    reach. Larger targets mean the strong quartic term below bites when the
    learning rate is too aggressive."""
    return 2.0 * math.sin(i * 0.7) + 0.5 * ((i % 7) - 3)


def loss_and_grad(x):
    """A bowl around `target` with a pronounced quartic term, so the surface is a
    genuine tradeoff: too small a learning rate converges slowly (needs more
    STEPS), too large a one overshoots and diverges. Raises on overflow, which the
    caller treats as divergence."""
    loss = 0.0
    grad = [0.0] * len(x)
    for i, xi in enumerate(x):
        d = xi - target(i)
        loss += d * d + 0.25 * d**4
        grad[i] = 2.0 * d + 1.0 * d**3
    # Loss is averaged (a stable reported scalar); gradients are per-coordinate
    # (NOT averaged), so an over-large learning rate really overshoots the basin
    # and diverges rather than being damped by the dimensionality.
    return loss / len(x), grad


# A large-but-finite loss standing in for a diverged run (NaN/inf/overflow), so
# an over-aggressive setting reports a poor score instead of crashing — the agent
# still gets clean signal that it went too far.
DIVERGED_LOSS = 1e9


def main() -> None:
    x = [0.0] * DIM
    velocity = [0.0] * DIM
    last_loss = float("inf")

    for _ in range(STEPS):
        try:
            last_loss, grad = loss_and_grad(x)
            if not math.isfinite(last_loss):
                last_loss = DIVERGED_LOSS
                break
            for i in range(DIM):
                velocity[i] = MOMENTUM * velocity[i] - LEARNING_RATE * grad[i]
                x[i] += velocity[i]
        except OverflowError:
            last_loss = DIVERGED_LOSS
            break

    # Map the final loss to a bits-per-byte-style scalar (lower = better).
    val_bpb = math.log2(1.0 + max(last_loss, 0.0)) + 0.05
    print(f"val_bpb = {val_bpb:.6f}")


if __name__ == "__main__":
    main()
