"""
MindWatch - AI-Powered Mental Health Awareness Platform
Prediction and Inference Module
"""

import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import os

class DepressionDetector:
    """Depression detection using fine-tuned DistilBERT"""
    
    def __init__(self, model_path='models/distilbert_depression'):
        """Initialize the detector with trained model"""
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self):
        """Load the trained model and tokenizer"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first using ml/train_model.py"
            )
        
        print(f"Loading model from: {self.model_path}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
    
    def predict(self, text, return_attention=False):
        """
        Predict depression from text
        
        Args:
            text (str): Input text to analyze
            return_attention (bool): Whether to return attention weights
        
        Returns:
            dict: Prediction results with label, confidence, and optionally attention
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=return_attention)
        
        # Get probabilities
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        
        # Get prediction
        prediction_idx = probs.argmax().item()
        confidence = probs.max().item()
        
        # Map to labels
        label = "Depressed" if prediction_idx == 1 else "Non-Depressed"
        
        result = {
            'label': label,
            'confidence': float(confidence),
            'probabilities': {
                'Non-Depressed': float(probs[0][0]),
                'Depressed': float(probs[0][1])
            },
            'is_crisis': float(probs[0][1]) > 0.85  # High threshold for crisis
        }
        
        # Add attention weights if requested
        if return_attention and outputs.attentions is not None:
            # Get attention from last layer, average across heads
            attention = outputs.attentions[-1].mean(dim=1).squeeze().cpu().numpy()
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Create token-attention pairs
            token_attention = [
                {'token': token, 'attention': float(attention[i])}
                for i, token in enumerate(tokens)
                if token not in ['[CLS]', '[SEP]', '[PAD]']
            ]
            
            # Sort by attention weight
            token_attention = sorted(token_attention, key=lambda x: x['attention'], reverse=True)
            
            result['attention'] = token_attention[:20]  # Top 20 tokens
        
        return result
    
    def predict_batch(self, texts):
        """
        Predict depression for multiple texts
        
        Args:
            texts (list): List of texts to analyze
        
        Returns:
            list: List of prediction results
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results
    
    def get_emotion_breakdown(self, text):
        """
        Get detailed emotion analysis (placeholder for future enhancement)
        
        Args:
            text (str): Input text
        
        Returns:
            dict: Emotion breakdown
        """
        # This is a simplified version
        # In production, you could use a multi-label emotion classifier
        
        prediction = self.predict(text)
        
        # Simple keyword-based emotion detection (can be enhanced)
        emotions = {
            'sadness': 0.0,
            'anxiety': 0.0,
            'anger': 0.0,
            'hopelessness': 0.0,
            'loneliness': 0.0
        }
        
        text_lower = text.lower()
        
        # Sadness keywords
        if any(word in text_lower for word in ['sad', 'cry', 'tears', 'depressed', 'down']):
            emotions['sadness'] = 0.7
        
        # Anxiety keywords
        if any(word in text_lower for word in ['anxious', 'worry', 'nervous', 'scared', 'fear']):
            emotions['anxiety'] = 0.6
        
        # Anger keywords
        if any(word in text_lower for word in ['angry', 'hate', 'furious', 'mad']):
            emotions['anger'] = 0.5
        
        # Hopelessness keywords
        if any(word in text_lower for word in ['hopeless', 'pointless', 'give up', 'no point']):
            emotions['hopelessness'] = 0.8
        
        # Loneliness keywords
        if any(word in text_lower for word in ['alone', 'lonely', 'isolated', 'nobody']):
            emotions['loneliness'] = 0.7
        
        return {
            'primary_prediction': prediction,
            'emotions': emotions
        }


# Singleton instance for reuse
_detector_instance = None

def get_detector(model_path='models/distilbert_depression'):
    """Get or create detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DepressionDetector(model_path)
    return _detector_instance


if __name__ == '__main__':
    # Test the detector
    detector = DepressionDetector()
    
    # Test samples
    test_texts = [
        "I'm feeling great today! Life is wonderful.",
        "I feel so hopeless and empty. Nothing matters anymore.",
        "Just had a productive day at work. Feeling accomplished!",
        "I can't stop crying. The pain is unbearable."
    ]
    
    print("=" * 60)
    print("Testing Depression Detector")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\nText: {text}")
        result = detector.predict(text, return_attention=True)
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Crisis Alert: {result['is_crisis']}")
        
        if 'attention' in result:
            print("Top attention tokens:", [t['token'] for t in result['attention'][:5]])
