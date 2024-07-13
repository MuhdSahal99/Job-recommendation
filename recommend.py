from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

job_descriptions = [
    "Software engineer with experience in Python and machine learning",
    "Data analyst proficient in SQL and data visualization",
    "Marketing specialist with social media expertise"
]

resumes = [
    "Experienced software developer with 5 years of Python programming and ML projects",
    "Data enthusiast with strong SQL skills and Tableau experience",
    "Digital marketer with 3 years of experience in social media campaigns"
]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def generate_embeddings(texts, model):
    preprocessed_texts = [preprocess_text(text) for text in texts]
    return model.encode(preprocessed_texts)

def calculate_similarity(job_embeddings, resume_embeddings):
    return cosine_similarity(resume_embeddings, job_embeddings)

def recommend_candidates(similarity_matrix, job_descriptions, resumes, top_n=2):
    for i, job in enumerate(job_descriptions):
        print(f"\nTop candidates for job: {job}")
        job_similarities = similarity_matrix[:, i]
        top_candidates = np.argsort(job_similarities)[::-1][:top_n]
        
        for rank, candidate_index in enumerate(top_candidates, 1):
            print(f"{rank}. Candidate: {resumes[candidate_index]}")
            print(f"   Similarity score: {job_similarities[candidate_index]:.4f}")

def main():
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Generate embeddings
    job_embeddings = generate_embeddings(job_descriptions, model)
    resume_embeddings = generate_embeddings(resumes, model)

    # Calculate similarity
    similarity_matrix = calculate_similarity(job_embeddings, resume_embeddings)

    # Recommend candidates
    recommend_candidates(similarity_matrix, job_descriptions, resumes)

if __name__ == "__main__":
    main()