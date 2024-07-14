import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import fitz  # PyMuPDF
from docx import Document

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

def read_pdf(file):
    text = ""
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def read_docx(file):
    doc = Document(file)
    text = " ".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def main():
    st.title("Job Recommendation System")

    # Load the model
    model = load_model()

    # Upload job descriptions
    st.header("Job Descriptions")
    job_files = st.file_uploader("Upload Job Descriptions (PDF or DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)
    job_descriptions = []
    if job_files:
        for job_file in job_files:
            if job_file.type == "application/pdf":
                job_text = read_pdf(job_file)
            elif job_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                job_text = read_docx(job_file)
            job_descriptions.append(job_text)

    # Upload resumes
    st.header("Resumes")
    resume_files = st.file_uploader("Upload Resumes (PDF or DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)
    resumes = []
    if resume_files:
        for resume_file in resume_files:
            if resume_file.type == "application/pdf":
                resume_text = read_pdf(resume_file)
            elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = read_docx(resume_file)
            resumes.append(resume_text)

    if st.button("Generate Recommendations"):
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
                st.subheader(f"Job: {rec['job'][:100]}...")  # Display the first 100 characters of the job description
                for candidate in rec['candidates']:
                    st.write(f"{candidate['rank']}. Candidate: {candidate['candidate'][:100]}...")  # Display the first 100 characters of the resume
                    st.write(f"   Similarity score: {candidate['score']:.4f}")
                st.write("---")
        else:
            st.warning("Please upload at least one job description and one resume.")

if __name__ == "__main__":
    main()
