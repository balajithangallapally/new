import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    """Cleans the input text by removing special characters and digits."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
    return text.lower()  # Convert to lower case

def tokenize(text):
    """Tokenizes the cleaned text into words."""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Removes stopwords from the tokenized list."""
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def lemmatize(tokens):
    """Lemmatizes the list of tokens to their base form."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def skill_gap_detection(tokens, skills):
    """Detects skill gaps compared to a provided list of skills."""
    missing_skills = [skill for skill in skills if skill not in tokens]
    return missing_skills

def preprocess_text(text, skills):
    """Main function to preprocess the text and detect skill gaps."""
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)
    tokens_without_stopwords = remove_stopwords(tokens)
    lemmatized_tokens = lemmatize(tokens_without_stopwords)
    gaps = skill_gap_detection(lemmatized_tokens, skills)
    return lemmatized_tokens, gaps

# Example usage
if __name__ == "__main__":
    input_text = "Your input text goes here."
    skills_list = ["python", "data analysis", "machine learning"]
    tokens, gaps = preprocess_text(input_text, skills_list)
    print("Tokens:", tokens)
    print("Skill Gaps:", gaps)