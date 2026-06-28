import os
import torch
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import OUTPUT_MODELS_DIR, OUTPUT_REPORTS_DIR, MAX_LENGTH, DATA_PATH
from src.data_preprocessing import load_and_prepare_data

def evaluate_best_model():
    # 1. Get best model name
    best_model_path = os.path.join(OUTPUT_REPORTS_DIR, "best_model.txt")
    if not os.path.exists(best_model_path):
        print(" Best model file not found. Please run train.py first.")
        return

    with open(best_model_path, "r") as f:
        best_model_key = f.read().strip()

    print(f" Evaluating best model: {best_model_key}")

    # 2. Load model and tokenizer
    model_dir = os.path.join(OUTPUT_MODELS_DIR, best_model_key)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # 3. Load label encoder
    le_path = os.path.join(OUTPUT_MODELS_DIR, "label_encoder.pkl")
    label_encoder = joblib.load(le_path)

    # 4. Load test data
    _, _, test_df, _, _, _ = load_and_prepare_data()

    # 5. Prediction loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    prompts = test_df["prompt"].tolist()
    true_labels = test_df["label_id"].tolist()
    all_preds = []

    print(" Running predictions on test set...")
    with torch.no_grad():
        for i in range(0, len(prompts), 32):
            batch = prompts[i:i+32]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

    # 6. Report and Visualization
    report = classification_report(true_labels, all_preds, target_names=label_encoder.classes_)
    print("\n Classification Report:")
    print(report)

    cm = confusion_matrix(true_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - Best Model ({best_model_key})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(OUTPUT_REPORTS_DIR, "best_model_confusion_matrix.png"))
    plt.show()

if __name__ == "__main__":
    evaluate_best_model()
