"""
MindWatch - AI-Powered Mental Health Awareness Platform
Data Preprocessing Module
"""

import re
import pandas as pd
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split

class TextPreprocessor:
    """Preprocess text data for DistilBERT"""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    def clean_text(self, text):
        """Clean text by removing URLs, extra spaces, etc."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Optional: Remove emojis (keeping for now as they may contain emotional info)
        # text = emoji.replace_emoji(text, replace='')
        
        return text
    
    def tokenize_texts(self, texts, max_length=512):
        """Tokenize texts using DistilBERT tokenizer"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        encodings = self.tokenizer(
            cleaned_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return encodings
    
    def prepare_dataset(self, csv_path, text_column='text', label_column='label'):
        """Load and prepare dataset from CSV"""
        df = pd.read_csv(csv_path)
        
        # Clean texts
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Convert labels to binary (0: Non-Depressed, 1: Depressed)
        label_map = {'Non-Depressed': 0, 'Depressed': 1, 0: 0, 1: 1}
        df['label_binary'] = df[label_column].map(label_map)
        
        # Remove any rows with missing data
        df = df.dropna(subset=['cleaned_text', 'label_binary'])
        
        return df
    
    def split_dataset(self, df, test_size=0.15, val_size=0.15, random_state=42):
        """Split dataset into train, validation, and test sets"""
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['label_binary']
        )
        
        # Second split: separate validation set from training
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_df['label_binary']
        )
        
        print(f"Dataset split:")
        print(f"  Training: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df


def create_sample_dataset(output_path='data/sample_dataset.csv'):
    """Create a sample dataset for demonstration"""
    import os
    
    # Sample data (in production, use real datasets)
    depressed_samples = [
        "I feel so hopeless and empty inside. Nothing brings me joy anymore.",
        "I can't get out of bed. Everything feels pointless.",
        "I'm so tired of pretending to be okay. I just want it all to end.",
        "Nobody understands what I'm going through. I feel so alone.",
        "I hate myself and everything about my life.",
        "I can't stop crying. The pain is unbearable.",
        "I feel like a burden to everyone around me.",
        "What's the point of trying anymore? Nothing ever works out.",
        "I'm exhausted all the time but can't sleep at night.",
        "I feel numb. I can't feel anything anymore.",
    ]
    
    non_depressed_samples = [
        "Had a great day at work today! Feeling accomplished.",
        "Looking forward to the weekend with friends.",
        "Just finished a good book. Highly recommend it!",
        "Beautiful weather today. Going for a walk in the park.",
        "Grateful for my family and their support.",
        "Excited about my new project. Can't wait to get started!",
        "Had a delicious dinner tonight. Trying new recipes is fun.",
        "Feeling energized after my morning workout.",
        "Love spending time with my pets. They always make me smile.",
        "Planning a trip next month. So excited to explore new places!",
    ]
    
    # Create DataFrame
    data = {
        'text': depressed_samples + non_depressed_samples,
        'label': ['Depressed'] * len(depressed_samples) + ['Non-Depressed'] * len(non_depressed_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created at: {output_path}")
    print(f"Total samples: {len(df)}")
    
    return df


if __name__ == '__main__':
    # Create sample dataset
    create_sample_dataset()
    
    # Test preprocessing
    preprocessor = TextPreprocessor()
    df = preprocessor.prepare_dataset('data/sample_dataset.csv')
    train_df, val_df, test_df = preprocessor.split_dataset(df)
    
    print("\nPreprocessing complete!")
