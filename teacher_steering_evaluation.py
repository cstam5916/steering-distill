import os
import re
import torch
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
from transformers.pipelines.pt_utils import KeyDataset
from peft import AutoPeftModelForCausalLM
import sys
from sae_lens import SAE
from steering_utils import *

logging.set_verbosity_error()

def accuracy_eval(outputs, targets, device=None):
    pat = re.compile(r'(?<!\w)[\$]?(\d+(?:,\d{3})*(?:\.\d+)?)\D*$')
    extract_last_number = lambda s: float(m.group(1).replace(",", "")) if (m := pat.search(s)) else float("nan")

    output_texts = [o[0]["generated_text"][-1]['content'] for o in outputs]
    output_nums = [extract_last_number(txt) for txt in output_texts]
    target_nums = [extract_last_number(t) for t in targets]

    out_t = torch.tensor(output_nums, dtype=torch.float32, device=device)
    tgt_t = torch.tensor(target_nums, dtype=torch.float32, device=device)

    valid = torch.isfinite(out_t) & torch.isfinite(tgt_t)
    acc = (out_t[valid] == tgt_t[valid]).float().sum() if valid.any() else torch.tensor(float("nan"), device=device)
    return acc.item()

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    print("Loading dataset...")
    ds = load_dataset("openai/gsm8k", "main")

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    sae, _, _ = SAE.from_pretrained(
        release="andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts",  # check sae_lens for exact release name
        sae_id="blocks.19.hook_resid_post"               # layer 19, residual stream post
    ).to(device=device, dtype=model.dtype)

    feature_indices = [61104, 4395, 46691, 46379, 48026]
    strength = 2.0

    pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    n = len(ds['test'])
    train_queries = ds["test"][:n]["question"]
    train_targets = ds["test"][:n]["answer"]

    for feature_idx in feature_indices:
        dir_name = f'Steering vec {feature_idx}'
        steering_vec = sae.W_dec[feature_idx].clone()
        max_activation = get_max_activation(model, tokenizer, sae, feature_idx, 
                                    list(ds['train']['question'])[:100],
                                    device)

        hook_fn = get_clamp_hook(steering_vec, max_activation=max_activation, strength=strength)
        handle = pipe.model.model.layers[19].register_forward_hook(hook_fn)

        dataset_messages = [
            [
                {"role": "system", "content": "Answer math questions step-by-step. End your response by clearly stating your final answer."},
                {"role": "user", "content": q},
            ]
            for q in train_queries
        ]

        try:
            print('Forward Pass...')
            outputs = pipe(dataset_messages, batch_size=2)
        finally:
            handle.remove()

        result_str = f"Feature {feature_idx} - Correct Answers: {accuracy_eval(outputs, train_targets, device=None)} out of {n}"
        print(result_str)
        results.append(result_str)
        with open(f"checkpoints/{dir_name}/results.txt", "w") as f:
            f.write("\n".join(results))
        print(results)
        with open(f"checkpoints/{dir_name}/results.txt", "w") as f:
            f.write("\n".join(results))

if __name__ == "__main__":
    main()
