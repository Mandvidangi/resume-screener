import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import pandas as pd

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def load_resumes(resume_folder):
    resumes = {}
    for filename in os.listdir(resume_folder):
        if filename.endswith(".docx"):
            path = os.path.join(resume_folder, filename)
            resumes[filename] = extract_text_from_docx(path)
    return resumes

def load_job_description(jd_path):
    with open(jd_path, 'r', encoding='utf-8') as f:
        return f.read()

def score_resumes(resumes, job_description):
    docs = [job_description] + list(resumes.values())
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(docs)
    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(jd_vector, resume_vectors).flatten()
    return {filename: score for filename, score in zip(resumes.keys(), scores)}

def list_job_descriptions(folder):
    return [f for f in os.listdir(folder) if f.endswith('.txt')]

if __name__ == "__main__":
    RESUME_DIR = "../resumes"
    JD_FOLDER = "../job_descriptions"
    OUTPUT_CSV = "../outputs/scored_resumes.csv"

    jd_files = list_job_descriptions(JD_FOLDER)
    print("Available Job Descriptions:")
    for i, jd_file in enumerate(jd_files):
        print(f"{i+1}. {jd_file}")

    choice = int(input("Select a JD (1–5): "))
    jd_path = os.path.join(JD_FOLDER, jd_files[choice - 1])

    resumes = load_resumes(RESUME_DIR)
    jd = load_job_description(jd_path)
    scores = score_resumes(resumes, jd)

    df = pd.DataFrame(scores.items(), columns=["Resume", "Score"])
    df = df.sort_values(by="Score", ascending=False)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n Scoring complete! Results saved to: {OUTPUT_CSV}")
