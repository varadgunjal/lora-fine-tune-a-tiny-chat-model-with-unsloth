"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel

def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    # TODO: call FastLanguageModel.from_pretrained with 4-bit loading and return (model, tokenizer)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True
    )
    return model, tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    # TODO: sum p.numel() over every parameter tensor in the module
    return sum(p.numel() for p in model.parameters())

# Step 3 - is_model_4bit_quantized
from bitsandbytes.nn import Linear4bit

def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # TODO: walk the model's submodules and check for a bitsandbytes Linear4bit instance
    for module in model.modules():
        if isinstance(module, Linear4bit):
            return True
    
    return False

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    # TODO: if the tokenizer is missing a pad token, reuse its eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    # TODO: return the list of attention projection module names LoRA should adapt
    return ["q_proj", "k_proj", "v_proj", "o_proj"]

# Step 6 - attach_lora_adapters
from unsloth import FastLanguageModel
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # TODO: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules
    target_modules = target_modules or get_lora_target_modules()
    return FastLanguageModel.get_peft_model(model, r=r, lora_alpha=lora_alpha, target_modules=target_modules)

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    # TODO: sum p.numel() over model.parameters() where requires_grad is True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    # TODO: return the fraction of parameters that are trainable.
    return trainable_count / total_count

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    # TODO: return a tiny hand-written list of instruction/response example dicts.
    return [
        {"instruction": """A store sells notebooks for $3 each or a pack of 5 for $12. What is the cheapest way to buy exactly 13 notebooks, and how much does it cost?""", "response": """Packs are cheaper per notebook ($12 / 5 = $2.40, compared with $3.00 single), so use as many packs as possible.\n\n- 2 packs = 10 notebooks for $24\n- 3 singles = 3 notebooks for $9\n- Total: 13 notebooks for $33\n\nCheck the alternatives: 1 pack + 8 singles = $12 + $24 = $36, and 13 singles = $39. A third pack would give 15 notebooks for $36, which is more than $33 and doesn't meet the 'exactly 13' requirement.\n\n**Answer: buy 2 packs and 3 single notebooks, for $33."""},
        {"instruction": """Write a Python function that returns the first non-repeating character in a string, or None if every character repeats.""", "response": """```python\nfrom collections import Counter\n\ndef first_unique_char(s: str) -> str | None:\n    counts = Counter(s)\n    for ch in s:\n        if counts[ch] == 1:\n            return ch\n    return None\n```\n\n**How it works:** The first pass counts how often each character appears. The second pass walks the string in its original order and returns the first character with a count of 1, so the result respects position, not alphabetical order.\n\n**Complexity: **O(n) time and O(k) space, where k is the number of distinct characters.\n\n**Examples:**\n```python\nfirst_unique_char(\"swiss\")   #  'w'\nfirst_unique_char(\"aabb\")    # None\nfirst_unique_char(\"\")        # None\n```"""},
        {"instruction": """Rewrite this message to my landlord so it's polite but firm: 'The heater has been broken for two weeks and nobody has come. This is unacceptable. Fix it now.'""", "response": """Subject: Urgent: Heater Repair Still Outstanding\n\nHi [Landlord's name],\n\nI'm writing to follow up on the broken heater in my unit, which I first reported on [date]. It has now been two weeks without heat, and no one has come to inspect or repair it yet.\n\nWith the colder weather, this is becoming a serious problem, and I'd appreciate it being treated as a priority. Could you please arrange a repair visit within the next 48 hours and let me know the scheduled time?\n\nI'm happy to be flexible about access to the unit. Thank you for your prompt attention to this.\n\nBest regards,\n[Your name]\n[Unit number]\n[Phone number]"""}
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    # TODO: combine example['instruction'] and example['response'] into one string
    return f"""### Instruction:\n{example["instruction"]}\n\n### Response:\n{example["response"]}"""

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    # TODO: apply format_instruction_example to every example and return the list
    return [format_instruction_example(example) for example in examples]

# Step 12 - build_text_dataset
import datasets

def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    # TODO: return a datasets.Dataset with one 'text' column holding the given strings
    return datasets.Dataset.from_dict({"text": texts})

# Step 13 - tokenize_text (not yet solved)
# TODO: implement

# Step 14 - count_tokens (not yet solved)
# TODO: implement

# Step 15 - build_training_arguments (not yet solved)
# TODO: implement

# Step 16 - build_sft_trainer (not yet solved)
# TODO: implement

# Step 17 - run_sft_training (not yet solved)
# TODO: implement

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

