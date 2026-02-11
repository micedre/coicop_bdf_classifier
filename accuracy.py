#%%

import pandas as pd
from src.evaluation_metrics import *

df = pd.read_parquet("predictions_top5.parquet")
rows = df[~df["receips_from_app"]].index
df.drop(rows, inplace=True)
df.head()


#%%

 # Derive true level labels from the full coicop code
parts = df["code"].str.split(".")
df["true_level1"] = parts.str[0].str.zfill(2)
df["true_level2"] = parts.str[:2].str.join(".")
df["true_level3"] = parts.str[:3].str.join(".")
df["true_level4"] = parts.str[:4].str.join(".")
df["true_level5"] = parts.str[:5].str.join(".")

#%%

for level in ["level1", "level2", "level3", "level4", "level5"]:
      if f"predicted_{level}" not in df.columns:
          continue
      true_col = f"true_{level}"
      for k in [1, 3, 5]:
          pred_cols = [f"predicted_{level}"] + [f"predicted_{level}_top{i}" for i in range(2, k + 1)]
          pred_cols = [c for c in pred_cols if c in df.columns]
          acc = df[pred_cols].eq(df[true_col], axis=0).any(axis=1).mean()
          print(f"{level} top-{k}: {acc:.4f}")


#%%

df = pd.read_parquet("predictions_top5.parquet")

  # Check a few rows
print("coicop samples:", df["coicop"].head(5).tolist())
print("predicted_level1 samples:", df["predicted_level1"].head(5).tolist())
if "predicted_level2" in df.columns:
    print("predicted_level2 samples:", df["predicted_level2"].head(5).tolist())

# Derive and compare
parts = df["coicop"].str.split(".")
print("\nDerived level1:", parts.str[0].str.zfill(2).head(5).tolist())
print("Derived level2:", parts.str[:2].str.join(".").head(5).tolist())
#%%

from src.evaluation_metrics import *


results=evaluate_by_confidence(predictions, levels=[1,2,3,4])

#%%

plot_accuracy_vs_coverage(results)

#%%

results = evaluate_all_sources(predictions)
plot_accuracy_vs_coverage(results)
# %%

results = evaluate_by_confidence(predictions, sources=["manual_from_app", "manual_from_books", "receips_from_app", "suggester"])
plot_accuracy_vs_coverage(results)



# %%

