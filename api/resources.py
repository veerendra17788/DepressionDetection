from flask import Blueprint, jsonify
import json
import os

resources_bp = Blueprint('resources', __name__)

def load_json_data(filename):
    """Helper to load JSON data from data directory"""
    try:
        path = os.path.join(os.getcwd(), 'data', filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

@resources_bp.route('/helplines', methods=['GET'])
def get_helplines():
    """Get crisis helplines"""
    # Try to load from file, fallback to hardcoded defaults if missing
    helplines = load_json_data('crisis_helplines.json')
    
    if not helplines:
        helplines = {
            "India": [
                {"name": "Vandrevala Foundation", "number": "1860-266-2345 (24x7)"},
                {"name": "iCall", "number": "9152987821 (Mon-Sat, 8 AM-10 PM)"},
                {"name": "AASRA", "number": "9820466726 (24x7)"}
            ],
            "Global": [
                {"name": "International Suicide Prevention", "url": "https://www.befrienders.org/"}
            ]
        }
        
    return jsonify(helplines)

@resources_bp.route('/self-help', methods=['GET'])
def get_recommendations():
    """Get self-help recommendations"""
    resources = load_json_data('self_help_resources.json')
    
    if not resources:
        resources = {
            "breathing": [
                {"title": "Box Breathing", "duration": "4 mins", "desc": "Inhale 4s, hold 4s, exhale 4s, hold 4s"},
                {"title": "4-7-8 Breathing", "duration": "5 mins", "desc": "Inhale 4s, hold 7s, exhale 8s"}
            ],
            "meditation": [
                {"title": "Mindfulness for Beginners", "type": "Audio"},
                {"title": "Body Scan", "type": "Audio"}
            ],
            "activities": [
                "Go for a 10-minute walk",
                "Drink a glass of water",
                "Write down 3 things you are grateful for"
            ]
        }
        
    return jsonify(resources)
