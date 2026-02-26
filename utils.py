import torch

def tokenize(element, tokenizer, is_eval=False):
    prompt = "Answer the math question step-by-step. End your response by clearly stating your final answer."
    qs = element["question"]
    ans = element.get("answer", None)
    prompt_message = [
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": q},
        ]
        for q in qs
    ]
    tokens_prompt = tokenizer.apply_chat_template(prompt_message, tokenize=True, add_generation_prompt=True, return_dict=True)
    if is_eval:
        targets = [[-100]*len(ids) for ids in tokens_prompt["input_ids"]]
        return {"input_ids": tokens_prompt["input_ids"], "labels": targets, "attention_mask": tokens_prompt["attention_mask"]}
    else:
        train_message = [
            prompt_message[i] + [{"role": "assistant", "content": a}]
            for i, a in enumerate(ans)
        ]
        tokens_train = tokenizer.apply_chat_template(train_message, tokenize=True, add_generation_prompt=False, return_dict=True)
        targets = [
            ([-100] * len(p)) + t[len(p):]
            for p, t in zip(tokens_prompt["input_ids"], tokens_train["input_ids"])
        ]
        return {"input_ids": tokens_train["input_ids"], "labels": targets, "attention_mask": tokens_train["attention_mask"]}

def get_data_collator(tokenizer):
    def data_collator(features):
        is_eval = "labels" not in features[0]
        feats = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = tokenizer.pad(
            feats,
            padding=True,
            return_tensors="pt",
        )
        if not is_eval:
            max_len = batch["input_ids"].shape[1]
            labels = [f["labels"] for f in features]
            padded_labels = []
            for lab in labels:
                pad_len = max_len - len(lab)
                if tokenizer.padding_side == "left":
                    padded_lab = ([-100] * pad_len) + lab
                else:
                    padded_lab = lab + ([-100] * pad_len)
                padded_labels.append(padded_lab)
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch
    return data_collator

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )