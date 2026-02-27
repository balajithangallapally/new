from flask import Flask, render_template, request, jsonify
from model.recommender import JobRecommender
from model.preprocess import PreProcessor
import pandas as pd
import logging
from datetime import datetime
import traceback

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
preprocessor = PreProcessor()
recommender = JobRecommender()

@app.route('/')
def home():
    """Render the home page with input form."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handle job recommendation request.
    
    Expected JSON:
    {
        "skills": "python, sql, machine learning",
        "num_results": 10,
        "location": "United States" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'skills' not in data:
            return jsonify({
                'success': False,
                'error': 'Skills field is required'
            }), 400
        
        user_skills = data.get('skills', '').strip()
        num_results = int(data.get('num_results', 10))
        location = data.get('location', '').strip()
        
        if not user_skills:
            return jsonify({
                'success': False,
                'error': 'Please enter at least one skill'
            }), 400
        
        logger.info(f"Recommendation request - Skills: {user_skills}, Location: {location}")
        
        # Fetch jobs from API and fallback dataset
        jobs_df = recommender.fetch_jobs(user_skills, location)
        
        if jobs_df.empty:
            return jsonify({
                'success': False,
                'error': 'No jobs found for the given skills. Please try different keywords.'
            }), 404
        
        # Get recommendations
        recommendations = recommender.recommend(user_skills, jobs_df, num_results)
        
        # Calculate skill gap
        skill_gap = preprocessor.detect_skill_gap(
            user_skills,
            jobs_df['description'].tolist()[:20]
        )
        
        response = {
            'success': True,
            'user_skills': user_skills,
            'num_jobs_found': len(jobs_df),
            'recommendations': recommendations,
            'skill_gap': skill_gap,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in recommendation: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)