from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from model.recommender import JobRecommender
from model.preprocess import TextPreprocessor
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize recommender system
recommender = JobRecommender()
preprocessor = TextPreprocessor()

@app.route('/')
def index():
    """Render home page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return jsonify({'error': 'Failed to load page'}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handle job recommendation requests
    Expected JSON: {'skills': 'python, sql, machine learning'}
    """
    try:
        data = request.get_json()
        
        if not data or 'skills' not in data:
            return jsonify({'error': 'Skills parameter is required'}), 400
        
        user_skills = data['skills'].strip()
        
        if not user_skills:
            return jsonify({'error': 'Please enter at least one skill'}), 400
        
        location = data.get('location', '').strip() if data.get('location') else ''
        
        logger.info(f"Processing recommendation request - Skills: {user_skills}, Location: {location}")
        
        # Get recommendations
        recommendations = recommender.get_recommendations(
            user_skills=user_skills,
            location=location,
            top_n=10
        )
        
        if not recommendations:
            return jsonify({
                'success': True,
                'recommendations': [],
                'message': 'No jobs found matching your criteria. Please try different skills or location.'
            }), 200
        
        # Extract skill gap
        skill_gap = recommender.detect_skill_gap(
            user_skills=user_skills,
            recommendations=recommendations
        )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'skill_gap': skill_gap,
            'message': f'Found {len(recommendations)} job recommendations'
        }), 200
    
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error processing recommendation: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request. Please try again.'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Server is running'}), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Development
    app.run(debug=True, host='0.0.0.0', port=5000)