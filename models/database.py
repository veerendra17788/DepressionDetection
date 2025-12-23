from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication and profile"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=True)
    is_anonymous_user = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    analyses = db.relationship('TextAnalysis', backref='user', lazy=True, cascade='all, delete-orphan')
    journal_entries = db.relationship('JournalEntry', backref='user', lazy=True, cascade='all, delete-orphan')
    timeline_entries = db.relationship('EmotionalTimeline', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class TextAnalysis(db.Model):
    """Store text analysis results"""
    __tablename__ = 'text_analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    text_content = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(20), nullable=False)  # 'Depressed' or 'Non-Depressed'
    confidence_score = db.Column(db.Float, nullable=False)
    is_crisis = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<TextAnalysis {self.id} - {self.prediction}>'


class JournalEntry(db.Model):
    """Private journal entries"""
    __tablename__ = 'journal_entries'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=True)
    content = db.Column(db.Text, nullable=False)
    emotion_summary = db.Column(db.JSON, nullable=True)  # Store emotion breakdown
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<JournalEntry {self.id} - {self.title}>'


class EmotionalTimeline(db.Model):
    """Track emotional state over time"""
    __tablename__ = 'emotional_timeline'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    mood_score = db.Column(db.Float, nullable=False)  # 0-10 scale
    depression_probability = db.Column(db.Float, nullable=False)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<EmotionalTimeline {self.user_id} - {self.date}>'


class Feedback(db.Model):
    """User feedback for model improvement"""
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('text_analyses.id'), nullable=True)
    feedback_type = db.Column(db.String(50), nullable=False)  # 'correct', 'incorrect', 'suggestion'
    comment = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Feedback {self.id} - {self.feedback_type}>'
