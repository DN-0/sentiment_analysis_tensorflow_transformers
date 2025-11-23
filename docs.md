# 🎬 Sentiment Analysis on Text Using Transformers

This project fine-tunes a pretrained transformer model to classify text into sentiment categories. It can be applied to movie reviews or any other labeled text dataset.

---

## 🚀 Project Goals

- Build a multi-class sentiment classifier
- Fine-tune a pretrained transformer model
- Evaluate performance using real metrics (accuracy, F1, confusion matrix)
- Generate predictions for unseen text

---

## 📂 Project Workflow (Step-by-Step)

1. **Set up environment**
   - Create virtual environment
   - Install TensorFlow / Transformers / data tools
   - Optional: force CPU-only or enable GPU

2. **Load and inspect dataset**
   - Load training data
   - Inspect columns (text + labels)
   - Check label distribution / imbalance

3. **Split into train + validation**
   - Stratified sampling recommended

4. **Choose & load pretrained model**
   - Example: DistilBERT (fast + efficient)
   - Load matching tokenizer

5. **Tokenize dataset**
   - Convert text to input IDs + attention masks
   - Decide padding + max sequence length

6. **Build data pipeline**
   - Create batched datasets with shuffling
   - Use `tf.data.Dataset` (TensorFlow) or `DataLoader` (PyTorch)

7. **Train the model**
   - Loss: Cross-entropy
   - Optimizer: Adam
   - Track train vs validation accuracy and loss

8. **Evaluate performance**
   - Accuracy + F1 score
   - Classification report
   - Confusion matrix
   - Error inspection

9. **Predict on new/unseen text**
   - Tokenize inputs
   - Convert logits → predicted label

10. **Export results**
   - Save model weights
   - Export CSV predictions if doing a competition
   - Save plots to `results/`

---

## 📈 Evaluation Metrics to Use

- Accuracy
- Precision / Recall / F1 per class
- Confusion Matrix
- Class distribution analysis

---


