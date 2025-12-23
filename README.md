# MindWatch - AI-Powered Mental Health Platform

MindWatch is a comprehensive web application that uses Artificial Intelligence (DistilBERT) to detect early signs of depression from user text. The platform provides real-time analysis, emotional tracking, self-help resources, and crisis support while maintaining strict privacy and ethical standards.

## 🚀 Features

- **AI Depression Detection**: Real-time analysis of text using a fine-tuned DistilBERT model.
- **Emotional Timeline**: Visual tracking of mood trends over time.
- **Private Journal**: Secure space to record thoughts with automatic emotion tagging.
- **Self-Help Resources**: Curated list of breathing exercises, diverse activities, and crisis helplines.
- **Privacy First**: Anonymous mode and strict data handling policies.
- **Modern UI**: Glassmorphism design with responsive layout and dark mode aesthetics.

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3 (Custom Design System), JavaScript (Vanilla)
- **Backend**: Python, Flask
- **AI/ML**: PyTorch, Transformers (Hugging Face), DistilBERT
- **Database**: SQLite (Development) / PostgreSQL (Production)
- **Visualization**: Chart.js

## 📋 Prerequisites

- Python 3.9+
- pip (Python package manager)

## 🔧 Installation

1. **Clone the repository** (if applicable) or download the source code.

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Copy `.env.example` to `.env` (optional, defaults provided in code):
   ```bash
   # Windows
   copy .env.example .env
   
   # Mac/Linux
   cp .env.example .env
   ```

## 🧠 Model Setup

Before running the application, you need to train or download the model.

1. **Train the model** (using sample data):
   ```bash
   python ml/train_model.py
   ```
   This will:
   - Generate a sample dataset (`data/sample_dataset.csv`)
   - Fine-tune DistilBERT on this data
   - Save the model to `models/distilbert_depression/`

   *Note: For production, replace the sample dataset with a real academic dataset like CLPsych or RSDD.*

## 🏃‍♂️ Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the application**:
   Open your browser and navigate to `http://localhost:5000`

## 🧪 Testing

- **Run analysis**: Go to the dashboard/analysis page and type "I feel hopeless" vs "I feel great".
- **Check crisis alert**: Type "I want to end it all" to test the crisis detection threshold.
- **Anonymous mode**: Try logging in with "Anonymous Mode" from the login page.

## ⚠️ Medical Disclaimer

**MindWatch is NOT a replacement for professional medical advice, diagnosis, or treatment.** 
The depression scores are based on linguistic patterns and should be treated as informational only. If you or someone you know is in crisis, please use the resources page to find a helpline or contact emergency services immediately.

## 📄 License

This project is for educational purposes. API usage and datasets should comply with their respective terms.
