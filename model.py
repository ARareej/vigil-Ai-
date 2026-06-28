from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import sys

# Add the project root to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_tokenizer(model_name):
    print(f" Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        print(f" Error loading tokenizer {model_name}: {e}")
        # Fallback for MiniLM if needed
        if "all-MiniLM-L6-v2" in model_name:
            fallback = "nreimers/MiniLM-L6-H384-uncased"
            print(f" Trying fallback: {fallback}")
            return AutoTokenizer.from_pretrained(fallback)
        raise e

def load_classification_model(model_name, num_labels, id2label, label2id):
    print(f" Loading model: {model_name}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        return model
    except Exception as e:
        print(f" Error loading model {model_name}: {e}")
        # Fallback for MiniLM if needed
        if "all-MiniLM-L6-v2" in model_name:
            fallback = "nreimers/MiniLM-L6-H384-uncased"
            print(f" Trying fallback: {fallback}")
            return AutoModelForSequenceClassification.from_pretrained(
                fallback,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id
            )
        raise e
