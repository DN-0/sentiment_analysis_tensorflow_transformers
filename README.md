# sentiment_analysis_tensorflow_transformers

Sentiment Analysis on Movie Reviews — Project Summary
1. Project Objective

The goal of this project is to build a machine learning model that predicts the sentiment of movie review phrases from the Kaggle Sentiment Analysis on Movie Reviews dataset. The task is a five-class classification problem, where each phrase is labeled as:

0 – Negative

1 – Somewhat Negative

2 – Neutral

3 – Somewhat Positive

4 – Positive

The goal is to use transformer-based models with TensorFlow to fine-tune a pretrained language model and generate predictions suitable for Kaggle submission.

2. Dataset

Source: Kaggle — Sentiment Analysis on Movie Reviews

Format: Tab-separated text (train.tsv and test.tsv)

Size: 156,060 labeled phrases, each paired with a sentiment score

Characteristics:

Short text phrases rather than full reviews

Distribution is imbalanced, with Neutral dominating

Labels represent a nuanced emotional scale, not binary polarity

The dataset is split into:

90% training

10% validation (stratified by sentiment)

3. Methodology

The project uses fine-tuning of a pretrained transformer model rather than training from scratch, leveraging linguistic knowledge learned from large-scale corpora.

Key components:

Component	Choice	Reason
Pretrained Model	DistilBERT (uncased)	Smaller, faster, still high accuracy; beginner-friendly
Tokenization	WordPiece via pretrained tokenizer	Ensures consistency with model training
Framework	TensorFlow + tf.keras	Integration with Hugging Face TF models
Training Objective	Sparse categorical cross-entropy	Multi-class classification with integer labels

To reduce memory usage on a local Mac system:

CPU-only execution (disabled Metal GPU backend)

Reduced sequence length (MAX_LENGTH = 64)

Smaller batch size (BATCH_SIZE = 8–16)

Optionally truncated dataset for development runs

4. Model Training

The model was trained for several epochs using mini-batch gradient descent with the Adam optimizer.

Training monitored:

Loss vs. epochs

Accuracy vs. epochs

Observations:

Loss reduced steadily

Validation loss tracked training loss fairly well

No severe overfitting under conservative hyperparameters

5. Evaluation Metrics

To assess performance beyond accuracy (which can be misleading due to class imbalance), the following were evaluated on the validation set:

Accuracy

Confusion Matrix

Precision, Recall, and F1 Score per Class

Insights:

Neutral class was easiest to predict

Negative vs. somewhat negative was a common source of confusion

Model performed well on strongly positive and strongly negative phrases but less consistently on subtle sentiment distinctions

6. Prediction & Submission

The trained model was applied to the unlabeled test.tsv dataset. Outputs were converted to Kaggle-formatted submission:

PhraseId,Sentiment


A CSV file was generated:

submission_distilbert_tf.csv

7. Future Improvements

Potential enhancements include:

Fine-tuning a larger model (BERT-base, RoBERTa-base, DeBERTa)

Class-balanced sampling or weighted loss to address label imbalance

Using full review context instead of isolated phrases

Hyperparameter tuning (learning rate schedules, warmup, longer training)

GPU acceleration to allow larger batch sizes and longer sequences
