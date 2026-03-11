'''
Disclaimer: This utils file contains functions taken from the open-source implementation of the paper 
"I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse 
Autoencoders." The links to their paper and Github repository are included below.
Paper: https://arxiv.org/pdf/2503.18878
Repository: https://github.com/AIRI-Institute/SAE-Reasoning/tree/main
'''

import torch
from typing import List, Tuple, Callable
from torch import Tensor

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

from typing import List, Tuple, Callable
from torch import Tensor

def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_clamp_hook(
    direction: Tensor,
    max_activation: float = 1.0,
    strength: float = 1.0
):
    def hook_fn(module, input, output):
        nonlocal direction
        if torch.is_tensor(output):
            activations = output.clone()
        else:
            activations = output[0].clone()
        
        direction = direction / torch.norm(direction)
        direction = direction.type_as(activations)
        proj_magnitude = torch.sum(activations * direction, dim=-1, keepdim=True)
        orthogonal_component = activations - proj_magnitude * direction

        clamped = orthogonal_component + direction * max_activation * strength

        if torch.is_tensor(output):
            return clamped
        else:
            return (clamped,) + output[1:] if len(output) > 1 else (clamped,)
    return hook_fn