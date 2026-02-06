#%%

import pandas as pd

predictions = pd.read_parquet("predictions2.parquet")

predictions.head()

#%%

from src.evaluation_metrics import *

results=evaluate_by_confidence(predictions, levels=[1,2,3,4])

#%%

plot_accuracy_vs_coverage(results)