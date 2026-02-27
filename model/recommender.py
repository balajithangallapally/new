import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class JobRecommender:
    def __init__(self, app_id, app_key):
        self.app_id = app_id
        self.app_key = app_key
        self.jobs = self.fetch_jobs()

    def fetch_jobs(self):
        url = f"https://api.adzuna.com/v1/jobs/us/search/1?app_id={{self.app_id}}&app_key={{self.app_key}}&results_per_page=50"
        response = requests.get(url)
        data = response.json()
        job_posts = data.get('results', [])
        return pd.DataFrame(job_posts)

    def recommend_jobs(self, user_input, n_recommendations=5):
        vectorizer = TfidfVectorizer()
        job_descriptions = self.jobs['description'].fillna('')
        job_vectors = vectorizer.fit_transform(job_descriptions)
        
        user_vector = vectorizer.transform([user_input])
        cosine_similarities = cosine_similarity(user_vector, job_vectors).flatten()

        related_job_indices = cosine_similarities.argsort()[-n_recommendations:][::-1]
        recommended_jobs = self.jobs.iloc[related_job_indices]

        return recommended_jobs[['title', 'company', 'location', 'description']]

# Usage:
# recommender = JobRecommender('your_app_id', 'your_app_key')
# recommendations = recommender.recommend_jobs('data scientist with machine learning experience')
# print(recommendations)