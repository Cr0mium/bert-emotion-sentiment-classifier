# BERT-based Emotion & Sentiment Classification (NLP)

An end-to-end **Machine Learning / AI engineering–focused NLP project** that fine-tunes **BERT** for multi-class emotion classification, with a clean training–evaluation–inference pipeline built using **PyTorch** and **Hugging Face Transformers**.

This repository is designed to reflect **real-world ML engineering practices**: modular code structure, reproducible training, checkpointing, evaluation, and interactive inference.

---

## 🔍 Problem Statement

Given a short text input, predict the **underlying emotion** expressed by the sentence.

- Multi-class classification (6 emotion labels)
- Transformer-based deep learning approach
- Focus on model training, optimization, and inference (not just experimentation)

---

## 🧠 ML / AI Engineering Focus

This project emphasizes **how models are trained and used in practice**, not just accuracy:

- Fine-tuning a **pretrained Transformer (BERT)**
- Custom **training loop** with:
  - Learning rate warmup
  - Linear scheduler
  - Gradient clipping
- Proper **train / validation / test separation**
- **Checkpoint saving & loading** for reproducibility
- Standalone **evaluation and prediction scripts**
- GPU-aware execution (CPU / CUDA)

---

## 🧰 Tech Stack

- **Python**
- **PyTorch**
- **Hugging Face Transformers & Datasets**
- **BERT (bert-base-uncased)**
- **tqdm** (progress tracking)
- **Google Colab** (GPU training)
- **Git & GitHub**

---

## 📂 Project Structure

```
bert-emotion-sentiment-classifier/
│
├── src/
│   ├── dataset.py      # Tokenization & DataLoader preparation
│   ├── model.py        # BERT model initialization
│   ├── train.py        # Training loop + checkpointing
│   ├── evaluate.py     # Test set evaluation
│   └── predict.py      # Interactive inference
│
├── checkpoints/        # Saved model checkpoints (optional)
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

- **Hugging Face ****emotion**** dataset**
- 6 emotion classes
- Automatically split into:
  - Train
  - Validation
  - Test

---

## 🚀 Training

The model is trained using **BERT for sequence classification** with:

- AdamW optimizer
- Learning rate warmup (10%)
- Linear decay scheduler
- Gradient clipping to stabilize training

Run training:

```bash
python src/train.py
```

During training:

- Training & validation loss are tracked
- Validation accuracy is computed each epoch
- Model checkpoints are saved per epoch

---

## 📈 Evaluation

Evaluate the trained model on the test set:

```bash
python src/evaluate.py
```

**Sample Result:**

```
Test Loss: 0.0379
Test Accuracy: 0.9861
```

---

## 🔮 Inference / Prediction

Interactive, loop-based prediction using the fine-tuned model:

```bash
python src/predict.py
```

Example:

```
Enter text (or type 'exit'): I feel really excited about this project!
Predicted Emotion: joy
```

The inference pipeline:

- Loads tokenizer & trained checkpoint
- Runs model in `eval()` mode
- Disables gradients for efficiency
- Converts logits → predicted label

---

## 🧪 Key ML Concepts Demonstrated

- Transformer fine-tuning
- Logits vs probabilities
- Gradient clipping
- Learning rate scheduling
- Model checkpointing
- Torch `no_grad()` for inference
- Device-aware ML code (CPU/GPU)

---

## 👨‍💻 Author

**Chinmoy Deka**\
ML / AI Engineering Enthusiast

> This project reflects my focus on **machine learning engineering**, particularly in NLP and deep learning systems. It is part of my portfolio for ML / AI engineering roles.

---

## 📌 Notes

- This project prioritizes **engineering clarity and correctness** over dataset scale.
- The same pipeline can be extended to:
  - Larger datasets
  - More emotion classes
  - Deployment (API / batch inference)

---

## ⭐ Future Improvements

- Add experiment tracking (e.g., TensorBoard)
- Hyperparameter configuration via YAML
- Model export for deployment
- Support for larger emotion taxonomies

---

