# model_loader.py
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)

def load_model(model_name: str, device: int = -1):
    """
    Load a Hugging Face model and tokenizer.
    Automatically handles CausalLM (GPT-like) and Seq2SeqLM (T5-like) models.
    """
    logging.info(f"Loading model '{model_name}' on device {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Detect model type automatically
    if "t5" in model_name.lower() or "flan" in model_name.lower() or "bart" in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        task = "text2text-generation"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        task = "text-generation"

    generator = pipeline(task=task, model=model, tokenizer=tokenizer, device=device)
    logging.info(f"Model '{model_name}' loaded successfully as {task} pipeline.")
    return generator