from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from models.database import db, JournalEntry
from ml.predict import get_detector

journal_bp = Blueprint('journal', __name__)

@journal_bp.route('/', methods=['GET'])
@login_required
def list_entries():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    entries = JournalEntry.query.filter_by(user_id=current_user.id)\
        .order_by(JournalEntry.created_at.desc())\
        .paginate(page=page, per_page=per_page)
        
    return jsonify({
        'entries': [{
            'id': e.id,
            'title': e.title,
            'preview': e.content[:100] + '...' if len(e.content) > 100 else e.content,
            'created_at': e.created_at.isoformat(),
            'emotion_summary': e.emotion_summary
        } for e in entries.items],
        'total': entries.total,
        'pages': entries.pages
    })

@journal_bp.route('/', methods=['POST'])
@login_required
def create_entry():
    data = request.get_json()
    
    if not data or not data.get('content'):
        return jsonify({'error': 'Content is required'}), 400
        
    # Analyze entry content
    try:
        detector = get_detector()
        result = detector.predict(data['content'])
        emotion_summary = {
            'label': result['label'],
            'confidence': result['confidence']
        }
    except Exception:
        emotion_summary = None
    
    entry = JournalEntry(
        user_id=current_user.id,
        title=data.get('title', 'Untitled Entry'),
        content=data['content'],
        emotion_summary=emotion_summary
    )
    
    db.session.add(entry)
    db.session.commit()
    
    return jsonify({
        'message': 'Journal entry created',
        'entry': {
            'id': entry.id,
            'title': entry.title,
            'created_at': entry.created_at.isoformat()
        }
    }), 201

@journal_bp.route('/<int:entry_id>', methods=['GET'])
@login_required
def get_entry(entry_id):
    entry = JournalEntry.query.get_or_404(entry_id)
    
    if entry.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
        
    return jsonify({
        'id': entry.id,
        'title': entry.title,
        'content': entry.content,
        'created_at': entry.created_at.isoformat(),
        'updated_at': entry.updated_at.isoformat(),
        'emotion_summary': entry.emotion_summary
    })

@journal_bp.route('/<int:entry_id>', methods=['PUT'])
@login_required
def update_entry(entry_id):
    entry = JournalEntry.query.get_or_404(entry_id)
    
    if entry.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
        
    data = request.get_json()
    
    if 'title' in data:
        entry.title = data['title']
    if 'content' in data:
        entry.content = data['content']
        # Re-analyze if content changes
        try:
            detector = get_detector()
            result = detector.predict(data['content'])
            entry.emotion_summary = {
                'label': result['label'],
                'confidence': result['confidence']
            }
        except Exception:
            pass
            
    db.session.commit()
    
    return jsonify({'message': 'Entry updated successfully'})

@journal_bp.route('/<int:entry_id>', methods=['DELETE'])
@login_required
def delete_entry(entry_id):
    entry = JournalEntry.query.get_or_404(entry_id)
    
    if entry.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
        
    db.session.delete(entry)
    db.session.commit()
    
    return jsonify({'message': 'Entry deleted successfully'})
