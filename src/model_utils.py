from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name, cache_dir):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map="auto"
    )
    return model, tokenizer
