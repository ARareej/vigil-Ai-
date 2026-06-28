from datasets import Dataset
import torch
import os
import sys

# Add the project root to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TEXT_COLUMN

def create_hf_dataset(df, tokenizer, max_length):
    # Convert pandas to HF Dataset
    # We only need prompt and label_id
    hf_data = {
        "text": df[TEXT_COLUMN].tolist(),
        "labels": df["label_id"].tolist()
    }
    dataset = Dataset.from_dict(hf_data)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Remove the original text column
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    
    # Set format for PyTorch
    tokenized_dataset.set_format("torch")
    
    return tokenized_dataset
