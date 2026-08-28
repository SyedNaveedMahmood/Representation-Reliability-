"""Analyze completed E01A discovery traces without running a model."""

from __future__ import annotations

import argparse
from pathlib import Path

from representation_reliability.analysis.e01a_trace_mechanism import (
    analyze_e01a_trace_mechanism,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-0-6b", type=Path, required=True)
    parser.add_argument("--run-1-7b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=240817)
    args = parser.parse_args()
    analyze_e01a_trace_mechanism(
        [args.run_0_6b, args.run_1_7b],
        args.output,
        n_bootstraps=args.bootstrap_samples,
        seed=args.seed,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
