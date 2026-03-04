from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model
import torch
import argparse
import os
from datasets import load_dataset
from huggingface_hub import HfApi
from utils import tokenize, get_data_collator, print_trainable_parameters
from losses import loss_token_ce, KDTrainer

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

parser = argparse.ArgumentParser(description='Training GNN')
parser.add_argument("--loss", type=str, default="token_ce", choices=["token_ce", "kd", "steer_kd"], help="loss function")
parser.add_argument("--output_dir", type=str, help="output directory (under checkpoints)")
args = parser.parse_args()

os.makedirs("checkpoints", exist_ok=True)


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

    train_args = Seq2SeqTrainingArguments(
        output_dir=f'checkpoints/{args.output_dir}',
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

    if(args.loss == "token_ce"):
        trainer = Seq2SeqTrainer(
            model=lora_model,
            processing_class=tokenizer,
            args=train_args,
            data_collator=get_data_collator(tokenizer),
            compute_loss_func=loss_token_ce,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
        )
    elif(args.loss == "kd"):
        teacher_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_id, device_map="auto")
        trainer = KDTrainer(
            model=lora_model,
            teacher_model = teacher_model,
            processing_class=tokenizer,
            args=train_args,
            data_collator=get_data_collator(tokenizer),
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
        )
    else:
        raise NotImplementedError('The model type you have inputted is not yet implemented.')

    print('Training...')
    trainer.train()
   

if(__name__ == '__main__'):
    main()