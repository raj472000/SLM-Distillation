import torch
from app.config import Config

def distillation_loss(student_logits, teacher_logits, labels):

    ce = torch.nn.functional.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    kl = torch.nn.functional.kl_div(
        torch.log_softmax(student_logits, dim=-1),
        torch.softmax(teacher_logits, dim=-1),
        reduction="batchmean"
    )

    return Config.ALPHA * kl + (1 - Config.ALPHA) * ce