import fitz  # PyMuPDF
import docx
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Predict job field using TF-IDF + cosine similarity (Apache-safe)
def predict_job_field(resume_text, top_n=3):
    resume_text = resume_text.strip()
    if not resume_text:
        return [{"title": "Unknown", "confidence": 0.0, "confidence_percent": "0%"}]

    try:
        job_texts = []
        job_titles = []

        with open("job_knowledge_base.csv", newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                job = row["job_title"].strip()
                keywords = row["keywords"].strip()
                if not keywords:
                    continue
                job_titles.append(job)
                job_texts.append(keywords)

        if not job_texts:
            return [{"title": "Unknown", "confidence": 0.0, "confidence_percent": "0%"}]

        # Append resume to job descriptions for vectorization
        job_texts.append(resume_text)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(job_texts)

        resume_vector = tfidf_matrix[-1]
        job_vectors = tfidf_matrix[:-1]

        similarities = cosine_similarity(resume_vector, job_vectors)[0]
        ranked = sorted(zip(job_titles, similarities), key=lambda x: x[1], reverse=True)[:top_n]

        result = [
            {
                "title": title,
                "confidence": round(score, 2),
                "confidence_percent": f"{round(score * 100)}%"
            }
            for title, score in ranked
        ]
        return result

    except Exception as e:
        print(f"Error in job prediction: {e}")
        return [{"title": "Unknown", "confidence": 0.0, "confidence_percent": "0%"}]

# Get recommended courses
def get_recommended_courses(job_title):
    try:
        with open("job_knowledge_base.csv", newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["job_title"].strip().lower() == job_title.strip().lower():
                    return [course.strip() for course in row["courses"].split(";") if course.strip()]
        return []
    except FileNotFoundError:
        print("job_knowledge_base.csv not found. No courses recommended.")
        return []
