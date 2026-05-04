from app.data.data_generation import generate_data
from app.data.data_filtering import filter_data
from app.data.dataset import prepare_dataset
from app.models.teacher import load_teacher
from app.models.student import load_student
from app.training.trainer import train
from app.models.lora import apply_lora
from app.config import Config
from app.ingestion.ingest import create_prompt

if Config.PROMPT_FLAG:
    prompts=create_prompt("../prompts.xlsx")
else:
    prompts = [
    "Explain machine learning",
    "What is deep learning?"]

data = generate_data(prompts)
data = filter_data(data)
dataset = prepare_dataset(data)

teacher = load_teacher()
student = load_student()

student = apply_lora(student)

train(student, teacher, dataset)