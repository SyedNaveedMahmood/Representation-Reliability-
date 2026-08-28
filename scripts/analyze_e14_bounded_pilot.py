from pathlib import Path

from representation_reliability.analysis.e14_bounded import analyze_bounded_pilot

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    analyze_bounded_pilot(
        {
            "BF16": root / "runs/E14/E14_ffbf9491977c",
            "INT8": root / "runs/E14/E14_bb53a2c07035",
            "INT4": root / "runs/E14/E14_5f683bd893de",
        },
        output_path=root / "E14_BOUNDED_PILOT_SUMMARY.md",
    )
