"""
MindWatch - AI-Powered Mental Health Awareness Platform
DistilBERT Model Training Module
"""

import os
import torch
import numpy as np
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
from ml.data_preprocessing import TextPreprocessor

class DepressionDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for depression detection"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_model(
    dataset_path='data/sample_dataset.csv',
    model_name='distilbert-base-uncased',
    output_dir='models/distilbert_depression',
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5
):
    """Train DistilBERT model for depression detection"""
    
    print("=" * 60)
    print("MindWatch - Depression Detection Model Training")
    print("=" * 60)
    
    # Initialize preprocessor
    print("\n[1/6] Loading and preprocessing data...")
    preprocessor = TextPreprocessor(model_name)
    
    # Load and prepare dataset
    df = preprocessor.prepare_dataset(dataset_path)
    train_df, val_df, test_df = preprocessor.split_dataset(df)
    
    # Tokenize datasets
    print("\n[2/6] Tokenizing texts...")
    train_encodings = preprocessor.tokenize_texts(train_df['cleaned_text'].tolist())
    val_encodings = preprocessor.tokenize_texts(val_df['cleaned_text'].tolist())
    test_encodings = preprocessor.tokenize_texts(test_df['cleaned_text'].tolist())
    
    # Create PyTorch datasets
    train_dataset = DepressionDataset(train_encodings, train_df['label_binary'].tolist())
    val_dataset = DepressionDataset(val_encodings, val_df['label_binary'].tolist())
    test_dataset = DepressionDataset(test_encodings, test_df['label_binary'].tolist())
    
    # Load pre-trained DistilBERT model
    print("\n[3/6] Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=learning_rate,
        save_total_limit=2,
    )
    
    # Initialize Trainer
    print("\n[4/6] Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train model
    print("\n[5/6] Training model...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print("\nTraining started...\n")
    
    trainer.train()
    
    # Evaluate on test set
    print("\n[6/6] Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nTest Set Results:")
    print(f"  Accuracy:  {test_results['eval_accuracy']:.4f}")
    print(f"  F1 Score:  {test_results['eval_f1']:.4f}")
    print(f"  Precision: {test_results['eval_precision']:.4f}")
    print(f"  Recall:    {test_results['eval_recall']:.4f}")
    
    # Save model and tokenizer
    print(f"\nSaving model to: {output_dir}")
    trainer.save_model(output_dir)
    preprocessor.tokenizer.save_pretrained(output_dir)
    
    # Save test results
    results_df = pd.DataFrame([test_results])
    results_df.to_csv(f'{output_dir}/test_results.csv', index=False)
    
    print("\n✓ Model training complete!")
    print(f"✓ Model saved to: {output_dir}")
    
    return trainer, test_results


if __name__ == '__main__':
    # Create sample dataset if it doesn't exist
    from ml.data_preprocessing import create_sample_dataset
    
    if not os.path.exists('data/sample_dataset.csv'):
        print("Creating sample dataset...")
        create_sample_dataset()
    
    # Train model
    trainer, results = train_model(
        dataset_path='data/sample_dataset.csv',
        num_epochs=3,
        batch_size=8  # Smaller batch size for demo data
    )
