from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from utils import tokenize, get_data_collator, print_trainable_parameters

api = HfApi()
try:
    api.whoami()
except Exception as e:
    raise RuntimeError(
        "Not logged in to Hugging Face. Run `huggingface-cli login` "
        "or set HUGGINGFACE_HUB_TOKEN."
    ) from e
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")



def main():
    print('Loading dataset...')
    ds = load_dataset("openai/gsm8k", "main") # Load dataset
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    print('Tokenizing dataset...')
    tokenized_train = ds['train'].map(
        tokenize,
        batched=True,
        remove_columns=ds["train"].column_names,
        fn_kwargs={"tokenizer":tokenizer, "is_eval":False},
        load_from_cache_file=False
    )
    tokenized_test = ds['test'].map(
        tokenize,
        batched=True,
        remove_columns=ds["test"].column_names,
        fn_kwargs={"tokenizer":tokenizer, "is_eval":True},
        load_from_cache_file=False
    )

    config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    lora_model = get_peft_model(model, config)
    print_trainable_parameters(lora_model)

    args = Seq2SeqTrainingArguments(
        output_dir='no-distil',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        gradient_accumulation_steps=8,
        num_train_epochs=5,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
    )

    trainer = Seq2SeqTrainer(
        model=lora_model,
        processing_class=tokenizer,
        args=args,
        data_collator=get_data_collator(tokenizer),
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    print('Training...')
    trainer.train()
   

if(__name__ == '__main__'):
    main()