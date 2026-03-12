'''
Disclaimer: This utils file contains functions taken from the open-source implementation of the paper 
"I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse 
Autoencoders." The links to their paper and Github repository are included below.
Paper: https://arxiv.org/pdf/2503.18878
Repository: https://github.com/AIRI-Institute/SAE-Reasoning/tree/main
'''

import torch

def get_max_activation(model, tokenizer, sae, feature_idx, prompts, device, hook_layer=19):
    max_act = 0.0
    hook_layer = model.model.layers[hook_layer]

    def capture_hook(module, input, output):
        nonlocal max_act
        hidden = output[0] if isinstance(output, tuple) else output
        hidden_cast = hidden.to(device=sae.device, dtype=sae.W_enc.dtype)
        features = sae.encode(hidden_cast)
        peak = features[..., feature_idx].abs().max().item()
        max_act = max(max_act, peak)

    handle = hook_layer.register_forward_hook(capture_hook)
    try:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)
    finally:
        handle.remove()

    return max_act

def get_clamp_hook(direction, max_activation=1.0, strength=1.0):
    direction = direction / torch.linalg.norm(direction)
    scaled = max_activation * strength
    d_cast = None

    def hook_fn(module, input, output):
        nonlocal d_cast
        hidden = output[0] if isinstance(output, tuple) else output
        if d_cast is None:
            d_cast = direction.to(device=hidden.device, dtype=hidden.dtype)
        proj = torch.sum(hidden * d_cast, dim=-1, keepdim=True)
        hidden = hidden.clone()
        hidden = hidden - proj * d_cast + scaled * d_cast
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook_fn