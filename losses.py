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

class SteeringProjector(nn.Module):
    def __init__(self, teacher_dim, student_dim):
        super().__init__()
        self.proj = nn.Linear(teacher_dim, student_dim, bias=False)

    def forward(self, teacher_vec):
        return self.proj(teacher_vec)

class SteeredKDTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_model, v_teacher, l_t, l_s, temperature=2.0, alpha=1.0, **kwargs):
        """
        teacher_model: frozen teacher model
        v_teacher: teacher steering vector, shape [teacher_hidden_dim]
        l_t: teacher layer index to steer
        l_s: student layer index to steer
        """
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.eval() # Teacher model should never require gradients
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)
        self.l_t = l_t # Steering layer num for teacher
        self.l_s = l_s # Steering layer num for student
        self.temperature = temperature # temperature (higher temp makes softmaxes more uniform)
        self.alpha = alpha

        teacher_dim = v_teacher.shape[0] # dimensionality of the teacher's latent space
        student_dim = self.model.config.hidden_size # dimensionality of the student's latent space
        self.projector = nn.Linear(teacher_dim, student_dim, bias=False) # learnable linear projection layer
        self.register_buffer("v_teacher", v_teacher) # steering vec will be saved/loaded with teacher model

    def make_hook(self, mode="teacher", coeff=1.0, token_position="last"):
        def hook(module, inputs, output):
            hidden = output
            if mode == "student":
                v_steer = self.projector(self.v_teacher.to(hidden.device, hidden.dtype)) # Project teacher steering vector into student hidden dimension
            else:
                v_steer = self.v_teacher.to(hidden.device, hidden.dtype) # Use teacher steering vector directly

            v_steer = coeff * v_steer  # optional scaling

            if token_position == "last":  # apply steering vector only to the most recent token
                hidden = hidden.clone()
                hidden[:, -1, :] += v_steer
            elif token_position == "all": # apply steering vector to all tokens
                hidden = hidden + v_steer.view(1, 1, -1)
            else:
                raise ValueError("token_position must be 'last' or 'all'")
            return hidden
        return hook

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        teacher_layer = self.teacher_model.model.layers[self.l_t]
        student_layer = model.model.layers[self.l_s]

        # -------------------------
        # First pass: with steering
        # -------------------------
        teacher_handle = teacher_layer.register_forward_hook(self.make_hook(mode="teacher"))
        student_handle = student_layer.register_forward_hook(self.make_hook(mode="student"))
        # handles are registered hook functions that get called during the forward pass
        # essentially, they add the steering vectors

        try:
            with torch.no_grad():
                teacher_outputs_steered = self.teacher_model(**inputs) # Forward pass of teacher w steering
            student_outputs_steered = model(**inputs) # Forward pass of student w steering
        finally:
            # After this is done, remove both handles
            teacher_handle.remove()
            student_handle.remove()

        T = self.temperature # temperature smooths the softmax
        teacher_probs_steered = F.softmax(teacher_outputs_steered.logits / T, dim=-1)
        student_log_probs_steered = F.log_softmax(student_outputs_steered.logits / T, dim=-1)
        kd_loss_steered = F.kl_div(
            student_log_probs_steered,
            teacher_probs_steered,
            reduction="batchmean"
        ) * (T ** 2) # Compute KL divergence

        # ----------------------------
        # Second pass: without steering
        # ----------------------------
        # Here we're doing the same thing, but since hooks have been removed the steering vectors aren't added
        with torch.no_grad():
            teacher_outputs_plain = self.teacher_model(**inputs)
        student_outputs_plain = model(**inputs)

        teacher_probs_plain = F.softmax(teacher_outputs_plain.logits / T, dim=-1)
        student_log_probs_plain = F.log_softmax(student_outputs_plain.logits / T, dim=-1)
        kd_loss_plain = F.kl_div(
            student_log_probs_plain,
            teacher_probs_plain,
            reduction="batchmean"
        ) * (T ** 2)

        # Sum both KL terms
        loss = self.alpha * (kd_loss_steered + kd_loss_plain)

        return (loss, student_outputs_steered) if return_outputs else loss