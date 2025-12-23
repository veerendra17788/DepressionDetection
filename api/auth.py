from flask import Blueprint, request, jsonify, session
from flask_login import login_user, logout_user, login_required, current_user
from models.database import db, User
from werkzeug.security import generate_password_hash
import datetime

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not data or not data.get('email') or not data.get('password') or not data.get('username'):
        return jsonify({'error': 'Missing required fields'}), 400
        
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 400
        
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already taken'}), 400
        
    user = User(
        email=data['email'],
        username=data['username']
    )
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'message': 'Registration successful', 'user_id': user.id}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Missing required fields'}), 400
        
    user = User.query.filter_by(email=data['email']).first()
    
    if user and user.check_password(data['password']):
        login_user(user)
        user.last_login = datetime.datetime.utcnow()
        db.session.commit()
        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
        })
        
    return jsonify({'error': 'Invalid email or password'}), 401

@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logout successful'})

@auth_bp.route('/anonymous', methods=['POST'])
def anonymous_login():
    """Create a temporary anonymous session"""
    # Create a temporary user or just set a session flag
    # For this implementation, we'll create a temporary user record that can be cleaned up
    username = f"Guest_{datetime.datetime.utcnow().timestamp()}"
    user = User(
        username=username,
        is_anonymous_user=True
    )
    db.session.add(user)
    db.session.commit()
    
    login_user(user)
    return jsonify({
        'message': 'Anonymous session started',
        'user': {
            'id': user.id,
            'username': 'Guest',
            'is_anonymous': True
        }
    })

@auth_bp.route('/status', methods=['GET'])
def auth_status():
    if current_user.is_authenticated:
        return jsonify({
            'is_authenticated': True,
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'is_anonymous': current_user.is_anonymous_user
            }
        })
    return jsonify({'is_authenticated': False})
