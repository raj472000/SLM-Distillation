from transformers import AutoModelForCausalLM
from app.config import Config

def load_student():
    return AutoModelForCausalLM.from_pretrained(Config.STUDENT_MODEL)