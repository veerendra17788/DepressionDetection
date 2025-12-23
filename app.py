from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_login import LoginManager
from models.database import db, User
from config import config
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_app(config_name='default'):
    """Application factory"""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    CORS(app)
    db.init_app(app)
    
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Register blueprints
    from api.auth import auth_bp
    from api.analysis import analysis_bp
    from api.journal import journal_bp
    from api.resources import resources_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(analysis_bp, url_prefix='/api/analysis')
    app.register_blueprint(journal_bp, url_prefix='/api/journal')
    app.register_blueprint(resources_bp, url_prefix='/api/resources')
    
    # Frontend routes
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/dashboard')
    def dashboard():
        return render_template('dashboard.html')
        
    @app.route('/login')
    def login_page():
        return render_template('auth/login.html')
        
    @app.route('/register')
    def register_page():
        return render_template('auth/register.html')
        
    @app.route('/analysis')
    def analysis_page():
        return render_template('analysis.html')
        
    @app.route('/journal')
    def journal_page():
        return render_template('journal.html')
        
    @app.route('/timeline')
    def timeline_page():
        return render_template('timeline.html')
        
    @app.route('/resources')
    def resources_page():
        return render_template('resources.html')
        
    @app.route('/learning')
    def learning_page():
        return render_template('learning.html')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Not found'}), 404
        return render_template('404.html'), 404
        
    @app.errorhandler(500)
    def server_error(e):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Internal server error'}), 500
        return render_template('500.html'), 500
        
    # Create database tables
    with app.app_context():
        db.create_all()
        
    return app

app = create_app(os.getenv('FLASK_ENV', 'default'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
