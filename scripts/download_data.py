"""Download the Sparkov dataset and print a summary. Run once."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data  # noqa: E402


def main():
    data.download_sparkov()
    for which in ("train", "test"):
        df = data.load_raw(which)
        summary = data.dataset_summary(df)
        print(f"\n[{which}] {summary}")


if __name__ == "__main__":
    main()
