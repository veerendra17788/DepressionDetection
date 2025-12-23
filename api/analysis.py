from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from models.database import db, TextAnalysis, EmotionalTimeline
from ml.predict import get_detector
import datetime

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/analyze', methods=['POST'])
@login_required
def analyze_text():
    data = request.get_json()
    
    if not data or not data.get('text'):
        return jsonify({'error': 'No text provided'}), 400
        
    text = data['text']
    
    # Get detector instance
    try:
        detector = get_detector()
        result = detector.predict(text, return_attention=True)
        
        # Save analysis to database
        analysis = TextAnalysis(
            user_id=current_user.id,
            text_content=text,
            prediction=result['label'],
            confidence_score=result['confidence'],
            is_crisis=result['is_crisis']
        )
        db.session.add(analysis)
        
        # Update emotional timeline
        # Calculate a mood score (0-10) based on prediction
        # Non-Depressed -> 5-10, Depressed -> 0-5
        base_score = 8.0 if result['label'] == 'Non-Depressed' else 2.0
        # Adjust by confidence: higher confidence in depression lowers score
        if result['label'] == 'Depressed':
            mood_score = max(0, 5.0 - (result['confidence'] * 4))
        else:
            mood_score = min(10, 5.0 + (result['confidence'] * 4))
            
        today = datetime.date.today()
        timeline = EmotionalTimeline.query.filter_by(
            user_id=current_user.id, 
            date=today
        ).first()
        
        if timeline:
            # Update average
            timeline.mood_score = (timeline.mood_score + mood_score) / 2
            timeline.depression_probability = result['probabilities']['Depressed']
        else:
            timeline = EmotionalTimeline(
                user_id=current_user.id,
                date=today,
                mood_score=mood_score,
                depression_probability=result['probabilities']['Depressed']
            )
            db.session.add(timeline)
            
        db.session.commit()
        
        return jsonify({
            'result': result,
            'analysis_id': analysis.id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/history', methods=['GET'])
@login_required
def get_history():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    analyses = TextAnalysis.query.filter_by(user_id=current_user.id)\
        .order_by(TextAnalysis.created_at.desc())\
        .paginate(page=page, per_page=per_page)
        
    return jsonify({
        'history': [{
            'id': a.id,
            'text': a.text_content[:100] + '...' if len(a.text_content) > 100 else a.text_content,
            'prediction': a.prediction,
            'confidence': a.confidence_score,
            'date': a.created_at.isoformat(),
            'is_crisis': a.is_crisis
        } for a in analyses.items],
        'total': analyses.total,
        'pages': analyses.pages,
        'current_page': page
    })

@analysis_bp.route('/timeline', methods=['GET'])
@login_required
def get_timeline():
    days = request.args.get('days', 30, type=int)
    start_date = datetime.date.today() - datetime.timedelta(days=days)
    
    entries = EmotionalTimeline.query.filter(
        EmotionalTimeline.user_id == current_user.id,
        EmotionalTimeline.date >= start_date
    ).order_by(EmotionalTimeline.date.asc()).all()
    
    return jsonify({
        'timeline': [{
            'date': e.date.isoformat(),
            'mood_score': e.mood_score,
            'depression_prob': e.depression_probability
        } for e in entries]
    })
