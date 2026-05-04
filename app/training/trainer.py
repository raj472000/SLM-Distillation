from transformers import Trainer, TrainingArguments
from app.training.loss import distillation_loss
from app.config import Config

class DistillTrainer(Trainer):

    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs["labels"]

        student_out = model(**inputs)
        student_logits = student_out.logits

        with torch.no_grad():
            teacher_out = self.teacher(**inputs)
            teacher_logits = teacher_out.logits

        loss = distillation_loss(student_logits, teacher_logits, labels)

        return loss


def train(student, teacher, dataset):

    args = TrainingArguments(
        output_dir="./model",
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUM,
        num_train_epochs=Config.EPOCHS,
        fp16=True,
        deepspeed="app/training/ds_config.json"
    )

    trainer = DistillTrainer(
        model=student,
        args=args,
        train_dataset=dataset,
        teacher_model=teacher
    )

    trainer.train()
    student.save_pretrained("./model")