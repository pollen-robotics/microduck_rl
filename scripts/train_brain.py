#!/usr/bin/env python3
"""Train the safe 5→hidden→5 MicroDuck behavior selector from reviewed JSONL feedback.

This is deliberately offline: feedback is reviewed before training and the output only selects an
affect. It cannot be uploaded to or self-updated by a physical robot.
"""
import argparse
import json
import math
import random
from pathlib import Path

STIMULI = ("idle", "person_nearby", "gentle_touch", "fallen", "human_control")
AFFECTS = ("resting", "curious", "warm", "recovering", "yielding")


def load_examples(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("feedback dataset is empty")
    result = []
    for row in rows:
        if set(row) != {"stimulus", "affect"} or row["stimulus"] not in STIMULI or row["affect"] not in AFFECTS:
            raise ValueError("each row must be exactly {stimulus, affect} using known labels")
        result.append((STIMULI.index(row["stimulus"]), AFFECTS.index(row["affect"])))
    return result


def train(examples, hidden=8, epochs=400, rate=0.1, seed=7):
    rng = random.Random(seed)
    # A trainable ReLU MLP. One-hot inputs make the tiny model transparent and easy to audit.
    w1 = [[rng.uniform(-0.2, 0.2) for _ in STIMULI] for _ in range(hidden)]
    b1 = [0.1] * hidden
    w2 = [[rng.uniform(-0.2, 0.2) for _ in range(hidden)] for _ in AFFECTS]
    b2 = [0.0] * len(AFFECTS)
    for _ in range(epochs):
        for x, target in examples:
            z = [w1[h][x] + b1[h] for h in range(hidden)]
            a = [max(0.0, value) for value in z]
            logits = [sum(w2[o][h] * a[h] for h in range(hidden)) + b2[o] for o in range(len(AFFECTS))]
            peak = max(logits); exps = [math.exp(value - peak) for value in logits]; total = sum(exps)
            grad = [value / total - float(index == target) for index, value in enumerate(exps)]
            old_w2 = [row[:] for row in w2]
            for o in range(len(AFFECTS)):
                for h in range(hidden): w2[o][h] -= rate * grad[o] * a[h]
                b2[o] -= rate * grad[o]
            for h in range(hidden):
                dz = sum(old_w2[o][h] * grad[o] for o in range(len(AFFECTS))) * float(z[h] > 0.0)
                w1[h][x] -= rate * dz; b1[h] -= rate * dz
    return {"schema_version": 1, "input_size": 5, "hidden_size": hidden, "output_size": 5,
            "hidden_weights": w1, "hidden_bias": b1, "output_weights": w2, "output_bias": b2}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("dataset", type=Path); parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hidden", type=int, default=8); parser.add_argument("--epochs", type=int, default=400); parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(); examples = load_examples(args.dataset)
    if args.hidden < 1 or args.epochs < 1: raise ValueError("hidden and epochs must be positive")
    args.out.write_text(json.dumps(train(examples, args.hidden, args.epochs, seed=args.seed), indent=2) + "\n")


if __name__ == "__main__": main()
