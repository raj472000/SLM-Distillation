from datasets import Dataset
from transformers import AutoTokenizer
from app.config import Config

tokenizer = AutoTokenizer.from_pretrained(Config.STUDENT_MODEL)

def prepare_dataset(data):

    texts = [
        f"Instruction: {d['instruction']}\nOutput: {d['output']}"
        for d in data
    ]

    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LENGTH
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(tokenize, batched=True)