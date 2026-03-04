import torch.nn.functional as F

def loss_token_ce(outputs, labels, num_items_in_batch=None):
    # Shift the logits and labels such that they are aligned for next-token comparison
    logits = outputs.logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    # Just getting the size of the vocaublary (dimension of output layer)
    vocab_size = logits.size(-1)

    # number of non-ignored tokens
    num_active_elements = labels.ne(-100).sum()

    # summed CE over active tokens
    loss_sum = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )

    # Regularize over active tokens, to exactly match the built-in Huggingface implementation
    loss = loss_sum / num_active_elements
    return loss