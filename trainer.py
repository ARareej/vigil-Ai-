import torch
from torch import nn
from transformers import Trainer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import sys

# Add the project root to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CLASS_WEIGHT_MODE, CLASS_WEIGHT_MIN, CLASS_WEIGHT_MAX, USE_FOCAL_LOSS, FOCAL_GAMMA

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Make sure alpha is on the same device and dtype as inputs
        alpha = self.alpha
        if alpha is not None:
            alpha = alpha.to(inputs.device, inputs.dtype)
        
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute custom loss using CrossEntropyLoss with class weights
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_custom_class_weights(train_df, num_labels):
    print(" Computing class weights...")
    y_train = train_df["label_id"].values
    
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_labels),
        y=y_train
    )
    
    if CLASS_WEIGHT_MODE == "sqrt":
        weights = np.sqrt(weights)
    elif CLASS_WEIGHT_MODE == "clip":
        weights = np.clip(weights, CLASS_WEIGHT_MIN, CLASS_WEIGHT_MAX)
        
    return weights
