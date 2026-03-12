import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sae_lens import SAE
from tqdm.auto import tqdm

from steering_utils import get_max_activation

def main():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()

    sae, _, _ = SAE.from_pretrained(
        release="andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts",  # check sae_lens for exact release name
        sae_id="blocks.19.hook_resid_post"               # layer 19, residual stream post
    ).to(device=device, dtype=model.dtype)

    ds = load_dataset("openai/gsm8k", "main")

    feature_indices = [61104, 4395, 46691, 46379, 48026]

    for idx in tqdm(feature_indices):
        steering_vec = sae.W_dec[idx].clone()
        max_activation = get_max_activation(model, tokenizer, sae, idx, 
                                     list(ds['train']['question'])[:100],
                                     device)
        torch.save((steering_vec, max_activation), f'steering_vectors/vec_{idx}.pt')


if __name__ == "__main__":
    main()