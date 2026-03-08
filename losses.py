import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer

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

def loss_kd(inputs, labels, student_logits, teacher):
    T=2 # temperature
    teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
    with torch.no_grad():
        teacher_logits = teacher(**teacher_inputs).logits
        
    # next-token alignment + mask (matches CE convention)
    s = (student_logits[..., :-1, :] / T)
    t = (teacher_logits[..., :-1, :] / T)
    y = labels[..., 1:]

    mask = y.ne(-100)  # [B, T-1]

    # compute KL per token then mean over active tokens
    log_p_s = F.log_softmax(s, dim=-1)
    p_t = F.softmax(t, dim=-1)

    kl = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)  # [B, T-1]
    kl = kl.masked_fill(~mask, 0.0)

    denom = mask.sum().clamp_min(1)
    return (kl.sum() / denom) * (T * T)

class KDTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_model=None,  **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # ce_loss = loss_token_ce(logits, labels, inputs)
        kd_loss = loss_kd(inputs, labels, logits, self.teacher_model)

        loss = kd_loss
        if return_outputs:
            return loss, outputs
        else:
            return loss

class SteeredKDTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_model, v_steer, l_t, l_s, **kwargs):
        '''
        teacher_model: the trained model that we use as a teacher
        v_steer: a single extracted steering vector from the teacher
        l_t: int, the layer number of the teacher 
        '''
        super().__init__(*args, **kwargs)