import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function for NLP text preprocessing

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return lemmatized_tokens

# Example skill gap detection function

def detect_skill_gap(user_skills, required_skills):
    gaps = [skill for skill in required_skills if skill not in user_skills]
    return gaps

# Example usage:
text = "This is an example sentence for NLP preprocessing."
tokens = preprocess_text(text)
print("Tokens after preprocessing:", tokens)

user_skills = ['Python', 'Machine Learning']
required_skills = ['Python', 'Machine Learning', 'Data Analysis']
skill_gaps = detect_skill_gap(user_skills, required_skills)
print("Skill gaps detected:", skill_gaps)