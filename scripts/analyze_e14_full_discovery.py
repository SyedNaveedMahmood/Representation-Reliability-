"""Analyze the frozen E14 full-discovery runs."""

from pathlib import Path

from representation_reliability.analysis.e14_full import analyze_full_discovery

if __name__ == "__main__":
    analyze_full_discovery(
        {
            "BF16": Path("runs/E14/E14_98b8d199abbc"),
            "INT8": Path("runs/E14/E14_31ff5aa1a9a2"),
            "INT4": Path("runs/E14/E14_5cb70b93e672"),
        },
        output_path=Path("E14_FULL_DISCOVERY_SUMMARY.md"),
    )
