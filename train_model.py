# ===============================
# Depression Detection - DistilBERT (PyTorch)
# ===============================

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords

# ===============================
# 1. CONFIGURATION
# ===============================
BATCH_SIZE = 16
EPOCHS = 2       # Increase for better accuracy
MAX_LEN = 64

# ===============================
# 2. PREPROCESSING
# ===============================
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# ===============================
# 3. LOAD DATA
# ===============================
print("Loading dataset...")
df = pd.read_csv('dataset.csv')
df.columns = df.columns.str.strip()
print("Columns found:", df.columns)

TEXT_COL = 'clean_text'      # Adjust if different
LABEL_COL = 'is_depression'  # Adjust if different

df['text'] = df[TEXT_COL].apply(preprocess_text)
data_texts = df['text'].tolist()
data_labels = df[LABEL_COL].tolist()

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data_texts, data_labels, test_size=0.2, random_state=42
)

# ===============================
# 4. TOKENIZATION
# ===============================
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class DepressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = DepressionDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = DepressionDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ===============================
# 5. MODEL INITIALIZATION
# ===============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# ===============================
# 6. TRAINING LOOP
# ===============================
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    model.train()
    total_loss = 0
    correct = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
    
    train_acc = correct / len(train_dataset)
    print(f"Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

# ===============================
# 7. SAVE MODEL & TOKENIZER
# ===============================
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
print("Training Complete and Model Saved!")
