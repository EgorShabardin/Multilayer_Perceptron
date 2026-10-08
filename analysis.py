import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


features = [f"feature_{i}" for i in range(1, 31)]
df = pd.read_csv("data/data.csv", header=None, names=["id", "diagnosis"] + features)

print("Размер:", df.shape)
print("Пропуски:", df.isna().sum().sum())
print(df["diagnosis"].value_counts())

rng = np.random.default_rng(42)
idx = rng.permutation(len(df))
split = int(0.7 * len(df))

df.iloc[idx[:split]].to_csv("data/train.csv", index=False)
df.iloc[idx[split:]].to_csv("data/valid.csv", index=False)
print("train:", split, "valid:", len(df) - split)

corr = df[features].corr()
plt.figure(figsize=(12, 10))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.show()