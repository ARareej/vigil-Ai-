import os
import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_MODELS_DIR, MAX_LENGTH
from src.translation import prepare_prompt_for_classification

# Global variables for caching
_BEST_MODEL = None
_TOKENIZER = None
_LABEL_ENCODER = None
_ID2LABEL = None

def _load_resources():
    global _BEST_MODEL, _TOKENIZER, _LABEL_ENCODER, _ID2LABEL
    
    if _BEST_MODEL is not None:
        return
        
    model_dir = os.path.join(OUTPUT_MODELS_DIR, "best_model")
    if not os.path.exists(model_dir):
        raise FileNotFoundError("Best model directory not found. Please run train.py first.")

    _TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
    _BEST_MODEL = AutoModelForSequenceClassification.from_pretrained(model_dir)
    _LABEL_ENCODER = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    _ID2LABEL = {int(i): label for i, label in enumerate(_LABEL_ENCODER.classes_)}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _BEST_MODEL.to(device)
    _BEST_MODEL.eval()

def predict_prompt(prompt):
    _load_resources()
    
    # Prepare prompt (language detection + translation)
    prep_result = prepare_prompt_for_classification(prompt)
    processed_prompt = prep_result["processed_prompt"]
    
    device = next(_BEST_MODEL.parameters()).device
    
    inputs = _TOKENIZER(
        processed_prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=MAX_LENGTH, 
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = _BEST_MODEL(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf, pred_id = torch.max(probs, dim=-1)
        
    pred_label = _ID2LABEL[int(pred_id[0])]
    confidence = float(conf[0])
    
    # Probabilities for all classes
    all_probs = { _ID2LABEL[i]: float(probs[0][i]) for i in range(len(_ID2LABEL)) }
    
    decision = "ALLOW" if pred_label == "SAFE" else "BLOCK"
    
    return {
        "original_prompt": prep_result["original_prompt"],
        "processed_prompt": prep_result["processed_prompt"],
        "detected_language": prep_result["detected_language"],
        "was_translated": prep_result["was_translated"],
        "predicted_label": pred_label,
        "confidence": confidence,
        "probabilities": all_probs,
        "decision": decision
    }

if __name__ == "__main__":
    test_prompt = "Generate a realistic photo of a celebrity arrested for drug use"
    result = predict_prompt(test_prompt)
    print(result)
    
    test_prompt_ar = "إنشاء صورة واقعية لشخصية مشهورة مقبوضة بسبب استخدام المخدرات"
    result_ar = predict_prompt(test_prompt_ar)
    print(result_ar)
