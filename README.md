# 🧠 SLM Distillation System (LLM → SLM)

### Production-Grade Pipeline using Hugging Face, QLoRA, DeepSpeed, and Multi-Adapter LoRA

---

## 📌 Overview

This project implements a **production-ready pipeline to distill a Small Language Model (SLM)** from a larger teacher model using:

* Hugging Face Transformers
* QLoRA (memory-efficient training)
* DeepSpeed (scalable distributed training)
* Multi-Adapter LoRA (modular specialization)
* FastAPI + vLLM (high-performance inference)

The system is designed to **optimize latency, cost, and deployability** while preserving as much performance as possible from the teacher model.

---

## 🎯 Objectives

* Distill knowledge from a large LLM into a smaller SLM
* Reduce inference cost and latency
* Enable deployment on constrained environments (CPU / low GPU)
* Support multi-domain adaptability via LoRA adapters

---

## ⚙️ Key Features

* 🔹 Synthetic data generation using teacher model
* 🔹 Data filtering and preprocessing pipeline
* 🔹 Tokenization and dataset preparation
* 🔹 Knowledge distillation (CE + KL loss)
* 🔹 QLoRA for low-memory training
* 🔹 DeepSpeed integration for scaling
* 🔹 Multi-adapter LoRA system (domain-specific fine-tuning)
* 🔹 FastAPI inference server
* 🔹 vLLM support for high-throughput serving

---

## 🏗️ Architecture

```text
Teacher Model (LLM)
        ↓
Data Generation (Synthetic + Self-Instruct)
        ↓
Data Filtering & Processing
        ↓
Tokenization (HF Tokenizer)
        ↓
Distillation Training (Student + LoRA + DeepSpeed)
        ↓
Evaluation
        ↓
Inference (FastAPI / vLLM)
```

---

## 📂 Project Structure

```text
slm_distillation/
│
├── app/
│   ├── config.py
│   ├── utils/logger.py
│
│   ├── data/
│   │   ├── data_generation.py
│   │   ├── data_filtering.py
│   │   └── dataset.py
│
│   ├── models/
│   │   ├── teacher.py
│   │   ├── student.py
│   │   └── lora.py
│
│   ├── training/
│   │   ├── loss.py
│   │   ├── trainer.py
│   │   └── ds_config.json
│
│   ├── evaluation/
│   │   └── evaluate.py
│
│   ├── inference/
│   │   ├── app.py
│   │   └── vllm_server.py
│
├── scripts/
│   └── run_pipeline.py
│
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Run Full Pipeline

```bash
python scripts/run_pipeline.py
```

This will:

* Generate synthetic data
* Filter and preprocess it
* Train student model using distillation
* Save the distilled model

---

### 3️⃣ Start Inference API

```bash
uvicorn app.inference.app:app --host 0.0.0.0 --port 8000
```

---

### 4️⃣ Test API

```bash
GET /generate?prompt=Explain AI
```

---

## 🧠 Multi-Adapter LoRA System

The system supports **multiple LoRA adapters**:

* `general`
* `finance`
* `coding`

### Example:

```python
model.set_adapter("finance")
```

### Use Cases:

* Multi-domain chatbot
* SaaS multi-tenant systems
* Personalized AI agents

---

## ⚡ DeepSpeed Integration

DeepSpeed is used for:

* Memory optimization (ZeRO Stage 2)
* Faster training
* Scaling to larger models

Config:

```json
{
  "fp16": { "enabled": true },
  "zero_optimization": { "stage": 2 }
}
```

---

## 📊 Distillation Strategy

Loss function:

```text
Loss = α * KL Divergence + (1 - α) * Cross Entropy
```

* KL → Mimics teacher distribution
* CE → Learns correct outputs

---

## 📈 Evaluation

Metrics include:

* Task-specific outputs
* Latency measurement
* Qualitative comparison with teacher

---

## ⚖️ Trade-offs

| Aspect       | Benefit              | Trade-off                     |
| ------------ | -------------------- | ----------------------------- |
| QLoRA        | Low memory usage     | Slight accuracy drop          |
| DeepSpeed    | Scalable training    | Complex setup                 |
| Distillation | Smaller model        | Loss of reasoning depth       |
| Multi-LoRA   | Multi-domain support | Adapter management complexity |

---

## ⚠️ Failure Modes

* Teacher hallucination propagation
* Overfitting to synthetic data
* Incorrect adapter routing
* Domain leakage between adapters

---

## 🏭 Production Considerations

* Use dynamic batching for efficiency
* Add monitoring (Prometheus/Grafana)
* Implement adapter routing logic
* Use vLLM for high-throughput inference
* Version adapters for A/B testing

---

## 📌 When to Use This System

✅ Cost-sensitive applications
✅ Edge deployment
✅ High-throughput APIs
✅ Multi-domain systems

---

## ❌ When Not to Use

❌ Complex reasoning-heavy tasks
❌ Safety-critical systems
❌ Very small datasets

---

## 🔮 Future Improvements

* CI/CD pipeline integration
* Kubernetes deployment
* RAG + Distillation hybrid system
* Automated adapter routing using ML
* Evaluation dashboards

---

## 🏁 Conclusion

This system provides a **scalable, modular, and efficient approach** to:

* Distill large models
* Deploy smaller models
* Support multi-domain adaptability

It is suitable for **production-grade AI systems** requiring performance, efficiency, and flexibility.

---
