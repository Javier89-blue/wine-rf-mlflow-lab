"""
Part D: Reduce the Wine dataset size by 20% (i.e. keep 80%) in a
reproducible way using a fixed random_state.
"""
import pandas as pd
from pathlib import Path

SOURCE_PATH = "data/wine.csv"
OUTPUT_PATH = "data/wine_reduced_80.csv"
RANDOM_STATE = 42
KEEP_FRACTION = 0.8


def main():
    df = pd.read_csv(SOURCE_PATH)

    df_reduced = df.sample(frac=KEEP_FRACTION, random_state=RANDOM_STATE).reset_index(drop=True)

    Path("data").mkdir(exist_ok=True)
    df_reduced.to_csv(OUTPUT_PATH, index=False)

    print(f"Original dataset size: {len(df)}")
    print(f"Reduced dataset size:  {len(df_reduced)}")
    print(f"Reduced dataset saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
