# sentiment_analysis_tensorflow_transformers

# Sentiment Analysis on Movie Reviews

This project fine-tunes a transformer-based language model to classify movie review phrases into five sentiment categories using the Kaggle *Sentiment Analysis on Movie Reviews* dataset.

---

## 🎯 Project Objective

The goal is to predict sentiment on a 5-class scale:

| Label | Meaning |
|-------|---------|
| 0 | Negative |
| 1 | Somewhat Negative |
| 2 | Neutral |
| 3 | Somewhat Positive |
| 4 | Positive |

The model is fine-tuned using TensorFlow + Transformers.

---

## 📊 Dataset

**Source:** Kaggle — *Sentiment Analysis on Movie Reviews*  
**Format:** Tab-separated (`train.tsv`, `test.tsv`)

Key characteristics:

- Short phrases rather than full reviews
- Imbalanced label distribution (Neutral most common)
- Multi-class classification task

Data split:

- 90% training
- 10% validation (stratified)

---

## 🧠 Methodology

| Component | Choice | Reason |
|----------|--------|--------|
| Model | DistilBERT (uncased) | Lightweight, fast, good performance |
| Framework | TensorFlow + tf.keras | Works with HF TF models |
| Tokenizer | WordPiece tokenizer | Matches pretrained model |
| Loss | Sparse Categorical Cross-Entropy | Multi-class labels |
| Optimization | Adam fine-tuning | Stable for transformers |

To reduce resource use on local Mac environment:

- CPU-only mode (disabled Metal backend)
- Reduced sequence length (`MAX_LENGTH = 64`)
- Smaller batch size (`BATCH_SIZE = 8–16`)

---

## 📈 Evaluation

Performance measured using:

- **Accuracy**
- **Confusion Matrix**
- **Precision, Recall, and F1 per class**

Observations:

- Strong performance on clear positive/negative sentiment
- More confusion between subtle labels (1 vs 2, 2 vs 3)
- Neutral class dominant, making accuracy misleading alone

---

## 📁 Outputs & Submission

Predictions on `test.tsv` are exported to Kaggle format submission_distilbert_tf.csv.


