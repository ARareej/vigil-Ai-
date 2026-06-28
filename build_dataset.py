import pandas as pd
import numpy as np
from datasets import load_dataset
import os
import sys

# Add the project root to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PATH, PROCESSED_DATA_PATH

def map_aegis_label(row):
    label = str(row["prompt_label"]).lower()
    cats = str(row["violated_categories"]).lower()
    prompt = str(row["prompt"]).lower()
    
    # Human Identity Abuse Keywords
    identity_keywords = [
        "real person", "celebrity", "politician", "teacher", "student", 
        "classmate", "friend", "my boss", "my coworker", "my ex", 
        "face swap", "deepfake", "impersonate", "fake photo", "fake image", 
        "realistic photo of", "make it look like", "without consent", 
        "arrested", "committing a crime", "stealing", "drug use"
    ]
    
    if any(kw in prompt for kw in identity_keywords):
        return "HUMAN_IDENTITY_ABUSE"
        
    if label == "safe":
        return "SAFE"
    elif "sexual" in cats:
        return "SEXUAL"
    elif "suicide" in cats or "self harm" in cats:
        return "SELF_HARM"
    elif any(kw in cats for kw in ["violence", "guns", "illegal weapons"]):
        return "VIOLENCE"
    elif any(kw in cats for kw in ["pii", "privacy"]):
        return "PRIVACY"
    elif any(kw in cats for kw in ["hate", "harassment", "profanity"]):
        return "HATE_HARASSMENT"
    elif any(kw in cats for kw in ["misinformation", "conspiracy", "political"]):
        return "MISINFORMATION"
    elif any(kw in cats for kw in ["criminal", "controlled", "illegal activity"]):
        return "ILLEGAL"
    else:
        return "UNSAFE_OTHER"

def map_wildguard_label(x):
    if x == "unharmful":
        return "SAFE"
    else:
        return "UNSAFE_OTHER"

def map_beaver_label(x):
    if x is True or x == 1 or str(x).lower() == 'true':
        return "SAFE"
    else:
        return "UNSAFE_OTHER"

def build_final_dataset(output_path=DATA_PATH):
    print(" Starting dataset building process...")
    
    # A) Aegis dataset
    print(" Loading Aegis dataset...")
    try:
        aegis = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train").to_pandas()
        aegis = aegis[["prompt", "prompt_label", "violated_categories"]].drop_duplicates(subset=["prompt"])
        aegis["vigil_label"] = aegis.apply(map_aegis_label, axis=1)
        aegis["source"] = "Aegis"
        aegis_final = aegis[["prompt", "vigil_label", "source"]]
        print(f" Aegis loaded: {len(aegis_final)} rows")
    except Exception as e:
        print(f" Error loading Aegis: {e}")
        aegis_final = pd.DataFrame(columns=["prompt", "vigil_label", "source"])

    # B) WildGuardMix dataset
    print(" Loading WildGuardMix dataset...")
    try:
        wg = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train").to_pandas()
        wg = wg[["prompt", "prompt_harm_label"]].drop_duplicates(subset=["prompt"])
        wg["vigil_label"] = wg["prompt_harm_label"].apply(map_wildguard_label)
        wg["source"] = "WildGuardMix"
        wg_final = wg[["prompt", "vigil_label", "source"]]
        print(f" WildGuardMix loaded: {len(wg_final)} rows")
    except Exception as e:
        print(f" Error loading WildGuardMix: {e}")
        wg_final = pd.DataFrame(columns=["prompt", "vigil_label", "source"])

    # C) BeaverTails dataset
    print(" Loading BeaverTails dataset...")
    try:
        beaver = load_dataset("PKU-Alignment/BeaverTails", split="30k_train").to_pandas()
        beaver = beaver[["prompt", "is_safe"]].drop_duplicates(subset=["prompt"])
        beaver["vigil_label"] = beaver["is_safe"].apply(map_beaver_label)
        beaver["source"] = "BeaverTails"
        beaver_final = beaver[["prompt", "vigil_label", "source"]]
        print(f" BeaverTails loaded: {len(beaver_final)} rows")
    except Exception as e:
        print(f" Error loading BeaverTails: {e}")
        beaver_final = pd.DataFrame(columns=["prompt", "vigil_label", "source"])

    # D) DoNotAnswer dataset
    print(" Loading DoNotAnswer dataset...")
    try:
        dna = load_dataset("declare-lab/Do-Not-Answer", split="train").to_pandas()
        # Find column that contains the question/prompt
        prompt_col = "question" if "question" in dna.columns else "prompt"
        if prompt_col not in dna.columns:
            for col in dna.columns:
                if "question" in col.lower() or "prompt" in col.lower():
                    prompt_col = col
                    break
        
        dna = dna[[prompt_col]].rename(columns={prompt_col: "prompt"}).drop_duplicates(subset=["prompt"])
        dna["vigil_label"] = "UNSAFE_OTHER"
        dna["source"] = "DoNotAnswer"
        dna_final = dna[["prompt", "vigil_label", "source"]]
        print(f" DoNotAnswer loaded: {len(dna_final)} rows")
    except Exception as e:
        print(f" Error loading DoNotAnswer: {e}")
        dna_final = pd.DataFrame(columns=["prompt", "vigil_label", "source"])

    # E) Prompt Injection dataset
    print(" Loading Prompt Injection dataset...")
    try:
        pi_dict = load_dataset("neuralchemy/Prompt-injection-dataset")
        first_split = list(pi_dict.keys())[0]
        pi = pi_dict[first_split].to_pandas()
        pi = pi.rename(columns={"text": "prompt"})
        pi["vigil_label"] = pi["label"].apply(lambda x: "JAILBREAK" if x == 1 else "SAFE")
        pi["source"] = "PromptInjection"
        pi_final = pi[["prompt", "vigil_label", "source"]].drop_duplicates(subset=["prompt"])
        print(f" Prompt Injection loaded: {len(pi_final)} rows")
    except Exception as e:
        print(f" Error loading Prompt Injection: {e}")
        pi_final = pd.DataFrame(columns=["prompt", "vigil_label", "source"])

    # F) Merge datasets
    print(" Merging datasets...")
    df_final = pd.concat([aegis_final, wg_final, beaver_final, dna_final, pi_final], ignore_index=True)
    
    # Cleaning
    df_final = df_final.dropna(subset=["prompt", "vigil_label"])
    df_final["prompt"] = df_final["prompt"].astype(str).str.strip()
    df_final["vigil_label"] = df_final["vigil_label"].astype(str).str.strip()
    df_final = df_final.drop_duplicates(subset=["prompt"])
    
    # Save final dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    # Save backup
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_final.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print(f" Final dataset saved to {output_path}")
    print(f" Final shape: {df_final.shape}")
    print("\n Label Counts:")
    print(df_final["vigil_label"].value_counts())
    print("\n Source Counts:")
    print(df_final["source"].value_counts())
    
    return df_final

if __name__ == "__main__":
    build_final_dataset()
