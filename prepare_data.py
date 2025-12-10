"""
Optional data preparation script.

For this project we already ship a cleaned/labeled dataset at:
    data/smartwatch_labeled.csv

This script is here only to document that step in the pipeline.
You can extend it if you want to regenerate the labeled file from
the raw `smartwatch_genai.csv` dump.
"""

import pandas as pd
from pathlib import Path


RAW_PATH = Path("smartwatch_genai.csv")
OUT_PATH = Path("data/smartwatch_labeled.csv")


def main():
    if not RAW_PATH.exists():
        print(f"Raw file {RAW_PATH} not found. Nothing to do.")
        return

    if OUT_PATH.exists():
        print(f"Labeled file {OUT_PATH} already exists. Nothing to do.")
        return

    print("NOTE: In this starter project the labeled file already exists.")
    print("If you want to rebuild it from the raw dump, implement your")
    print("own cleaning + labeling logic here.")


if __name__ == "__main__":
    main()


