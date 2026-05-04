class Config:
    TEACHER_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
    STUDENT_MODEL = "distilgpt2"

    MAX_LENGTH = 512
    BATCH_SIZE = 2
    GRAD_ACCUM = 8
    EPOCHS = 3

    ALPHA = 0.7
    PROMPT_FLAG = False