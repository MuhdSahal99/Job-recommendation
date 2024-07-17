import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

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
    recommendations = []
    for i, job in enumerate(job_descriptions):
        job_similarities = similarity_matrix[:, i]
        top_candidates = np.argsort(job_similarities)[::-1][:top_n]
        
        job_recommendations = []
        for rank, candidate_index in enumerate(top_candidates, 1):
            job_recommendations.append({
                "rank": rank,
                "candidate": resumes[candidate_index],
                "score": job_similarities[candidate_index]
            })
        recommendations.append({
            "job": job,
            "candidates": job_recommendations
        })
    return recommendations

def main():
    st.title("Job Recommendation System")

    # Load the model
    model = load_model()

    # Input for job descriptions
    st.header("Job Descriptions")
    job_descriptions = []
    for i in range(3):
        job = st.text_area(f"Job Description {i+1}", key=f"job_{i}")
        if job:
            job_descriptions.append(job)

    # Input for resumes
    st.header("Resumes")
    resumes = []
    for i in range(3):
        resume = st.text_area(f"Resume {i+1}", key=f"resume_{i}")
        if resume:
            resumes.append(resume)

    if st.button("Generate Match"):
        if job_descriptions and resumes:
            # Generate embeddings
            job_embeddings = generate_embeddings(job_descriptions, model)
            resume_embeddings = generate_embeddings(resumes, model)

            # Calculate similarity
            similarity_matrix = calculate_similarity(job_embeddings, resume_embeddings)

            # Get recommendations
            recommendations = recommend_candidates(similarity_matrix, job_descriptions, resumes)

            # Display recommendations
            st.header("Recommendations")
            for rec in recommendations:
                st.subheader(f"Job: {rec['job']}")
                for candidate in rec['candidates']:
                    st.write(f"{candidate['rank']}. Candidate: {candidate['candidate']}")
                    st.write(f"   Similarity score: {candidate['score']:.4f}")
                st.write("---")
        else:
            st.warning("Please enter at least one job description and one resume.")

if __name__ == "__main__":
    main()
