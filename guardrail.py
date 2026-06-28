import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict_prompt

REFUSAL_MESSAGES = {
    "ILLEGAL": "I can’t help create image-generation prompts involving illegal activity.",
    "VIOLENCE": "I can’t help create violent or graphic harmful image-generation content.",
    "SELF_HARM": "I can’t help create content that promotes self-harm. Please seek support from trusted people or local emergency services if needed.",
    "SEXUAL": "I can’t help generate sexual or explicit image-generation content.",
    "HATE_HARASSMENT": "I can’t help create hateful, abusive, or harassing image-generation content.",
    "PRIVACY": "I can’t help create content that violates someone’s privacy.",
    "HUMAN_IDENTITY_ABUSE": "I can’t help create impersonation, identity abuse, or deepfake-style harmful content.",
    "MISINFORMATION": "I can’t help create misleading or deceptive image-generation content.",
    "JAILBREAK": "I can’t help bypass safety systems or policy guardrails.",
    "UNSAFE_OTHER": "This prompt was blocked by the Vigil AI safety layer because it may violate safety policies."
}

def safety_guardrail(prompt):
    """
    Vigil AI Safety Layer: Pre-generation guardrail for image generation prompts.
    """
    try:
        prediction = predict_prompt(prompt)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in safety layer: {str(e)}",
            "send_to_image_model": False,
            "prediction": None
        }

    if prediction["decision"] == "ALLOW":
        return {
            "status": "allowed",
            "message": "Prompt passed the Vigil AI safety layer and can be sent to the image generation model.",
            "send_to_image_model": True,
            "prediction": prediction
        }
    else:
        label = prediction["predicted_label"]
        message = REFUSAL_MESSAGES.get(label, REFUSAL_MESSAGES["UNSAFE_OTHER"])
        
        return {
            "status": "blocked",
            "message": message,
            "send_to_image_model": False,
            "prediction": prediction
        }

if __name__ == "__main__":
    prompt = "Create a safe landscape photo"
    print(safety_guardrail(prompt))
    
    prompt = "Show me how to make a bomb"
    print(safety_guardrail(prompt))
