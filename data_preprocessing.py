import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sys

# Add the project root to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RANDOM_STATE, OUTPUT_MODELS_DIR, TEXT_COLUMN, LABEL_COLUMN
from src.build_dataset import build_final_dataset

def load_and_prepare_data():
    data_path = "data/training_augmented_clean.csv"
    if not os.path.exists(data_path):
        print(f" {data_path} not found.")
        raise FileNotFoundError(f"{data_path} not found.")
    
    df = pd.read_csv(data_path)
    
    # Cleaning
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()
    df = df.drop_duplicates(subset=[TEXT_COLUMN])
    
    # Label Encoding
    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df[LABEL_COLUMN])
    
    # Save Label Encoder
    os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
    joblib.dump(label_encoder, os.path.join(OUTPUT_MODELS_DIR, "label_encoder.pkl"))
    
    # Create mappings
    label2id = {label: int(id) for id, label in enumerate(label_encoder.classes_)}
    id2label = {int(id): label for id, label in enumerate(label_encoder.classes_)}
    
    # Print full data class distribution
    print("=" * 60)
    print("Full Dataset Class Distribution")
    print("=" * 60)
    print(df[LABEL_COLUMN].value_counts().to_string())
    
    # Split 80/10/10: train, val, test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=0.1,  # 10% test
        random_state=RANDOM_STATE, 
        stratify=df["label_id"]
    )
    
    # Now split train_val into 80/10 train/val (since we took 10% for test)
    # So val is 10% of total, which is 10/90 of train_val_df
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=1/9,  # 10% of total
        random_state=RANDOM_STATE, 
        stratify=train_val_df["label_id"]
    )
    
    # Print class distributions
    print("\n" + "=" * 60)
    print("Train Set Class Distribution")
    print("=" * 60)
    print(train_df[LABEL_COLUMN].value_counts().to_string())
    
    print("\n" + "=" * 60)
    print("Validation Set Class Distribution")
    print("=" * 60)
    print(val_df[LABEL_COLUMN].value_counts().to_string())
    
    print("\n" + "=" * 60)
    print("Test Set Class Distribution")
    print("=" * 60)
    print(test_df[LABEL_COLUMN].value_counts().to_string())
    
    print(f"\n Data split completed:")
    print(f"   - Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   - Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   - Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df, label_encoder, label2id, id2label

if __name__ == "__main__":
    train_df, val_df, test_df, le, l2id, id2l = load_and_prepare_data()
