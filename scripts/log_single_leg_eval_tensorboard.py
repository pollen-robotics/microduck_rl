#!/usr/bin/env python3
"""Append single-leg evaluation percentiles to TensorBoard."""

import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


SERIES = ("p0", "p10", "p25", "p50", "p75", "p90", "p100", "mean")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument(
        "evaluations",
        nargs="+",
        metavar="STEP=JSON",
        help="Evaluation step and JSON path, for example 2250=eval.json.",
    )
    args = parser.parse_args()

    writer = SummaryWriter(args.log_dir)
    for item in args.evaluations:
        step_text, path_text = item.split("=", 1)
        step = int(step_text)
        results = json.loads(Path(path_text).read_text())
        for result in results:
            side = result["side"]
            continuity = result["continuity"]
            for metric, source in (
                ("longest_no_contact_s", "longest_swing_no_contact_s"),
                ("longest_all_gates_s", "longest_all_gates_s"),
            ):
                for series in SERIES:
                    writer.add_scalar(
                        f"Evaluation/{side}/{metric}/{series}",
                        continuity[source][series],
                        step,
                    )
            writer.add_scalar(
                f"Evaluation/{side}/success_rate",
                result["success_rate"],
                step,
            )
            writer.add_scalar(
                f"Evaluation/{side}/contact_transitions_per_s",
                continuity["mean_contact_transitions_per_s"],
                step,
            )

    layout = {}
    for side in ("left", "right"):
        label = f"{side.title()} support"
        for metric, title in (
            ("longest_no_contact_s", "Longest no-contact duration (s)"),
            ("longest_all_gates_s", "Longest valid hold duration (s)"),
        ):
            layout[f"{label}: {title}"] = [
                "Multiline",
                [f"Evaluation/{side}/{metric}/{series}" for series in SERIES],
            ]
    writer.add_custom_scalars({"Single-leg evaluation": layout})
    writer.close()


if __name__ == "__main__":
    main()
