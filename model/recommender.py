import requests
import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from model.preprocess import preprocess_text

logger = logging.getLogger(__name__)  

class JobRecommender:  
    """
    Job recommendation engine using TF-IDF vectorization and cosine similarity.
    Integrates with Adzuna API for real-time job data.
    """  
    
    def __init__(self):  
        self.adzuna_app_id = 'YOUR_ADZUNA_APP_ID'  
        self.adzuna_app_key = 'YOUR_ADZUNA_APP_KEY'  
        self.fallback_jobs = self._load_fallback_jobs()  
    
    def _load_fallback_jobs(self):  
        """Load fallback jobs from CSV file if API fails."""
        try:  
            jobs_df = pd.read_csv('data/jobs.csv')  
            logger.info(f"Loaded {len(jobs_df)} jobs from fallback dataset")  
            return jobs_df  
        except Exception as e:  
            logger.error(f"Error loading fallback jobs: {str(e)}")  
            return pd.DataFrame()  
    
    def fetch_jobs(self, keywords, location=''):  
        """  
        Fetch jobs from Adzuna API with fallback to local dataset.  
        
        Args:  
            keywords (str): Search keywords  
            location (str): Job location (optional)  
            
        Returns:  
            pd.DataFrame: Jobs dataframe  
        """  
        try:  
            # Try fetching from Adzuna API  
            jobs_df = self._fetch_from_adzuna(keywords, location)  
            if not jobs_df.empty:  
                logger.info(f"Fetched {len(jobs_df)} jobs from Adzuna API")  
                return jobs_df  
        except Exception as e:  
            logger.warning(f"API fetch failed: {str(e)}, using fallback dataset")  
        
        # Fallback to local dataset  
        return self._search_fallback_jobs(keywords, location)  
    
    def _fetch_from_adzuna(self, keywords, location=''):  
        """Fetch jobs from Adzuna API."""  
        try:  
            url = 'https://api.adzuna.com/v1/jobs/us/search/1'  
            params = {  
                'app_id': self.adzuna_app_id,  
                'app_key': self.adzuna_app_key,  
                'what': keywords,  
                'results_per_page': 50  
            }  
            
            if location:  
                params['where'] = location  
            
            response = requests.get(url, params=params, timeout=10)  
            response.raise_for_status()  
            
            data = response.json()  
            results = data.get('results', [])  
            
            jobs_list = []  
            for job in results:  
                jobs_list.append({  
                    'title': job.get('title', 'N/A'),  
                    'company': job.get('company', {}).get('display_name', 'N/A'),  
                    'location': job.get('location', {}).get('display_name', 'N/A'),  
                    'description': job.get('description', ''),  
                    'redirect_url': job.get('redirect_url', '#'),  
                    'salary_min': job.get('salary_min', None),  
                    'salary_max': job.get('salary_max', None)  
                })  
            
            return pd.DataFrame(jobs_list)  
        except Exception as e:  
            logger.error(f"Adzuna API error: {str(e)}")  
            raise  
    
    def _search_fallback_jobs(self, keywords, location=''):  
        """Search fallback dataset for jobs."""  
        if self.fallback_jobs.empty:  
            return pd.DataFrame()  
        
        keywords_lower = keywords.lower()  
        mask = (  
            self.fallback_jobs['title'].str.lower().str.contains(keywords_lower, na=False) |  
            self.fallback_jobs['description'].str.lower().str.contains(keywords_lower, na=False)  
        )  
        
        if location:  
            location_lower = location.lower()  
            mask &= self.fallback_jobs['location'].str.lower().str.contains(location_lower, na=False)  
        
        return self.fallback_jobs[mask].reset_index(drop=True)  
    
    def recommend(self, user_skills, jobs_df, num_results=10):  
        """  
        Generate job recommendations using TF-IDF and cosine similarity.  
        
        Args:  
            user_skills (str): User's skills as comma-separated string  
            jobs_df (pd.DataFrame): Dataframe of jobs  
            num_results (int): Number of recommendations to return  
            
        Returns:  
            list: List of recommended jobs with similarity scores  
        """  
        try:  
            if jobs_df.empty:  
                logger.warning("No jobs available for recommendation")  
                return []  
            
            # Preprocess user skills  
            user_input = preprocess_text(user_skills)  
            user_input_str = ' '.join(user_input)  
            
            # Prepare job descriptions  
            job_descriptions = jobs_df['description'].fillna('').tolist()  
            
            if not job_descriptions or not user_input_str:  
                return []  
            
            # Create TF-IDF vectors  
            vectorizer = TfidfVectorizer(  
                max_features=1000,  
                stop_words='english',  
                min_df=1,  
                max_df=0.8  
            )  
            
            # Fit on all descriptions + user input  
            all_texts = job_descriptions + [user_input_str]  
            vectors = vectorizer.fit_transform(all_texts)  
            
            # Calculate cosine similarity  
            user_vector = vectors[-1]  # Last vector is user input  
            job_vectors = vectors[:-1]  # Rest are jobs  
            
            similarities = cosine_similarity(user_vector, job_vectors).flatten()  
            
            # Get top N recommendations  
            top_indices = np.argsort(similarities)[::-1][:num_results]  
            
            recommendations = []  
            for idx in top_indices:  
                if similarities[idx] > 0:  # Only include if similarity > 0  
                    recommendation = {  
                        'title': jobs_df.iloc[idx]['title'],  
                        'company': jobs_df.iloc[idx]['company'],  
                        'location': jobs_df.iloc[idx]['location'],  
                        'description': jobs_df.iloc[idx]['description'][:200] + '...',  
                        'similarity_score': round(float(similarities[idx]), 4),  
                        'similarity_percentage': round(float(similarities[idx]) * 100, 2),  
                        'redirect_url': jobs_df.iloc[idx].get('redirect_url', '#'),  
                        'salary_min': jobs_df.iloc[idx].get('salary_min'),  
                        'salary_max': jobs_df.iloc[idx].get('salary_max')  
                    }  
                    recommendations.append(recommendation)  
            
            logger.info(f"Generated {len(recommendations)} recommendations")  
            return recommendations  
        
        except Exception as e:  
            logger.error(f"Error in recommendation: {str(e)}")  
            return []