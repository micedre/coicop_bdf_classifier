from src.data_preparation import load_annotations
from src.predict import HierarchicalCOICOPPredictor
from sklearn.metrics import accuracy_score, classification_report


df = load_annotations('data/annotations.parquet')

df.to_parquet('annotations_clean.parquet')

predictor = HierarchicalCOICOPPredictor('checkpoints/hierarchical/hierarchical_model')


# Predict
result_df = predictor.predict_dataframe(
    df,
    text_column='product',
    batch_size=32,
)

# Calculate metrics
y_true = result_df['code'].tolist()
y_pred = result_df["predicted_code"].tolist()

# Exact match accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nExact Match Accuracy: {accuracy:.4f}")

# Level 1 accuracy
y_true_l1 = [c.split(".")[0].zfill(2) for c in y_true]
y_pred_l1 = result_df["predicted_level1"].tolist()
accuracy_l1 = accuracy_score(y_true_l1, y_pred_l1)
print(f"Level 1 Accuracy: {accuracy_l1:.4f}")

print("\nLevel 1 Classification Report:")
print(classification_report(y_true_l1, y_pred_l1))

