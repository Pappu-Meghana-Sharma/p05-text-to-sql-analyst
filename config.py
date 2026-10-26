# Shared settings. Everyone imports this: from config import CONFIG
import random
import numpy as np

SEED = 42
QUICK_RUN = False      # True = small sample for fast runs
TRAIN = False         # False = load saved LoRA adapter

CONFIG = {
    "dataset": "gretelai/synthetic_text_to_sql",
    "llm": "Qwen/Qwen2.5-Coder-1.5B-Instruct",     # agent / few-shot
    "lora_base": "Qwen/Qwen2.5-0.5B-Instruct",     # fine-tuning
    "embedder": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k": 3,
    "test_size": 200,
    "quick_test_size": 20,
    "max_repairs": 3,
    "lora_train_size": 1000,
    "test_ids_path": "test_ids.json",
    "adapter_path": "lora_adapter/",
}

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
