from peft import LoraConfig, get_peft_model

def apply_lora(model):

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.1
    )

    return get_peft_model(model, config)