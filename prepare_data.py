"""
Part A: Load the Wine dataset from sklearn and save it as a CSV file.
Run this once at the start of the project.
"""
import pandas as pd
from sklearn.datasets import load_wine
from pathlib import Path

Path("data").mkdir(exist_ok=True)

wine = load_wine(as_frame=True)
df = wine.frame  # feature columns + "target" column

df.to_csv("data/wine.csv", index=False)

print("Dataset saved to data/wine.csv")
print(f"Shape: {df.shape}")
