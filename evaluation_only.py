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

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    checkpoint_dirs = [f"checkpoints/no-distil/checkpoint-{num}" for num in [935, 1870, 2805, 3740, 4675]]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token   # enable batching

    results = []
    for checkpoint_dir in checkpoint_dirs:
        print(f'Testing {checkpoint_dir}')
        lora_model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_dir)
        pipe = pipeline(
            "text-generation",
            model=lora_model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        n = len(ds['test'])
        train_queries = ds["test"][:n]["question"]
        train_targets = ds["test"][:n]["answer"]

        dataset_messages = [
            [
                {"role": "system", "content": "Answer math questions step-by-step. End your response by clearly stating your final answer."},
                {"role": "user", "content": q},
            ]
            for q in train_queries
        ]

        print('Forward Pass...')
        outputs = pipe(dataset_messages, batch_size=8)

        result_str = f"{checkpoint_dir} - Correct Answers: {accuracy_eval(outputs, train_targets, device=None)} out of {n}"
        print(result_str)
        results.append(result_str)
        with open("checkpoints/no-distil/results.txt", "w") as f:
            f.write("\n".join(results))
    print(results)
    with open("checkpoints/no-distil/results.txt", "w") as f:
        f.write("\n".join(results))

if __name__ == "__main__":
    main()
