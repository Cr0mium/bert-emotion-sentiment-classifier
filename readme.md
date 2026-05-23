# 🧠 BERT-based Emotion & Sentiment Classification (NLP)

An end-to-end **Machine Learning / AI Engineering focused NLP project** that fine-tunes **BERT** for multi-class emotion classification using **PyTorch** and **Hugging Face Transformers**.

This project is designed to reflect **real-world ML engineering practices** including:

- modular project structure
- reproducible training
- checkpointing
- evaluation pipelines
- inference workflows
- GPU-aware optimization

---

# 🔍 Problem Statement

Given a short text input, predict the **underlying emotion** expressed in the sentence.

This is a:

- **multi-class NLP classification task**
- built using a **Transformer-based deep learning pipeline**
- focused on practical ML engineering workflows rather than notebook-only experimentation

---

# 🧠 ML / AI Engineering Highlights

This project focuses on how modern NLP systems are trained and evaluated in production-style workflows.

### ✅ Features Implemented

- Fine-tuning **BERT (`bert-base-uncased`)**
- Custom PyTorch training loop
- Validation-based evaluation pipeline
- Early stopping
- Mixed precision training (AMP)
- Gradient clipping
- Learning rate warmup + scheduler
- Checkpoint saving/loading
- Standalone evaluation script
- Interactive inference pipeline
- CPU/GPU device-aware execution

---

# 🧰 Tech Stack

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **Hugging Face Datasets**
- **BERT**
- **scikit-learn**
- **tqdm**
- **Google Colab (GPU training)**
- **Git & GitHub**

---

# 📂 Project Structure

```text
bert-emotion-sentiment-classifier/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── checkpoints/
├── README.md
└── requirements.txt
```

### File Overview

| File          | Purpose                                       |
| ------------- | --------------------------------------------- |
| `dataset.py`  | Tokenization and DataLoader preparation       |
| `model.py`    | BERT model initialization                     |
| `train.py`    | Training loop, scheduler, AMP, early stopping |
| `evaluate.py` | Test evaluation and metrics                   |
| `predict.py`  | Interactive inference script                  |

---

# 📊 Dataset

This project uses the **Hugging Face `emotion` dataset**.

### Emotion Classes

- sadness
- joy
- love
- anger
- fear
- surprise

Dataset splits:

- Train
- Validation
- Test

---

# ⚙️ Training Pipeline

The model is trained using:

- **AdamW optimizer**
- **Linear learning rate scheduler**
- **10% warmup steps**
- **Gradient clipping**
- **Mixed precision training (AMP)**

### Run Training

```bash
python src/train.py
```

During training:

- training loss is tracked
- validation loss is tracked
- validation accuracy & macro F1 are computed
- best checkpoints are automatically saved
- early stopping prevents overfitting

---

# 📈 Model Performance

### Best Validation Metrics

| Metric              | Score      |
| ------------------- | ---------- |
| Validation Accuracy | **93.55%** |
| Macro F1 Score      | **0.9112** |

The model achieved strong balanced performance across all emotion classes.

---

# 🧪 Evaluation

Evaluate the model on the held-out test set:

```bash
python src/evaluate.py
```

Metrics computed:

- Loss
- Accuracy
- Macro F1 Score
- Classification Report

Example output:

```text
Test Accuracy: 0.93+
Macro F1 Score: 0.91+
```

---

# 🔮 Inference / Prediction

Interactive inference with the fine-tuned model:

```bash
python src/predict.py
```

Example:

```text
Enter text (or type 'exit'): I feel really excited about this project!
Predicted Emotion: joy
```

Inference pipeline includes:

- tokenizer loading
- checkpoint loading
- evaluation mode (`model.eval()`)
- disabled gradients (`torch.no_grad()`)
- logits → predicted class conversion

---

# 🧠 Key ML Concepts Demonstrated

- Transformer fine-tuning
- Transfer learning
- Multi-class classification
- Logits vs probabilities
- Gradient clipping
- Learning rate scheduling
- Mixed precision training
- Early stopping
- Checkpointing
- GPU-aware training
- Evaluation metrics (Accuracy + Macro F1)

---

# 🚀 Engineering Improvements Added

Compared to a basic tutorial pipeline, this project includes:

✅ Proper validation pipeline  
✅ Macro F1 evaluation  
✅ Early stopping  
✅ Mixed precision GPU training  
✅ Best-model checkpointing  
✅ Modular project structure  
✅ Reusable evaluation & inference scripts

---

# 👨‍💻 Author

## Chinmoy Deka

Machine Learning / AI Engineering Enthusiast focused on:

- NLP
- Deep Learning
- LLM Systems
- Retrieval-Augmented Generation (RAG)
- Applied AI Engineering

This project is part of my ML/AI engineering portfolio.

---

# 📌 Future Improvements

- TensorBoard / experiment tracking
- YAML-based config management
- Hyperparameter sweeps
- FastAPI deployment
- ONNX / TorchScript export
- Dockerized inference pipeline
- Larger transformer architectures
- Emotion intensity estimation

---

# ⭐ Final Notes

This project emphasizes:

- engineering clarity
- reproducible ML workflows
- practical transformer training
- clean NLP system design

rather than only achieving benchmark accuracy.

The same architecture can easily scale to:

- larger datasets
- more emotion classes
- production inference APIs
- deployment pipelines
