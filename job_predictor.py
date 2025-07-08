import fitz  # PyMuPDF
import docx
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

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

# Helper: Normalize text and split for robust token matching
def normalize_and_tokenize(text):
    import re
    tokens = re.findall(r'\b\w+\b', text.lower())
    return set(tokens)

# Load job knowledge base into DataFrame and handle duplicates
def load_job_knowledge_base(csv_path="job_knowledge_base.csv"):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=['job_title'], keep='first')  # Keep first occurrence of each job title
    # Fill NaNs with empty string for robust processing
    df = df.fillna('')
    return df

try:
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print("Could not load sentence-transformer model, falling back to TF-IDF.")
    semantic_model = None

# Predict job field using semantic similarity and robust keyword/token matching
def predict_job_field(resume_text, top_n=3):
    resume_text = resume_text.strip()
    if not resume_text:
        return [{"title": "Unknown", "confidence": 0.0, "confidence_percent": "0%"}]

    try:
        df = load_job_knowledge_base("job_knowledge_base.csv")
        job_titles = df["job_title"].tolist()
        keywords_list = df["keywords"].tolist()
        skills_list = df["skills"].tolist()
        tools_list = df["tools"].tolist()
        education_list = df["education_required"].tolist()
        descriptions = [
            f"{jt}. {kw} {sk} {tl} {ed}"
            for jt, kw, sk, tl, ed in zip(job_titles, keywords_list, skills_list, tools_list, education_list)
        ]

        # === PRIMARY: Use semantic similarity if possible ===
        if semantic_model:
            resume_embedding = semantic_model.encode(resume_text, convert_to_tensor=True)
            job_embeddings = semantic_model.encode(descriptions, convert_to_tensor=True)
            similarities = util.cos_sim(resume_embedding, job_embeddings)[0].cpu().numpy()
            ranked = sorted(
                zip(job_titles, similarities), key=lambda x: x[1], reverse=True
            )[:top_n]
            result = [
                {
                    "title": title,
                    "confidence": float(score),
                    "confidence_percent": f"{round(float(score) * 100)}%"
                }
                for title, score in ranked
            ]
            return result

        # === SECONDARY: TF-IDF fallback ===
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(descriptions + [resume_text])
        resume_vector = tfidf_matrix[-1]
        job_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(resume_vector, job_vectors)[0]
        ranked = sorted(
            zip(job_titles, similarities), key=lambda x: x[1], reverse=True
        )[:top_n]
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

# New: Predict job field using direct token overlap (robust hard evidence, used for tie-breaking)
def predict_job_field_tokens(resume_text, top_n=3):
    resume_tokens = normalize_and_tokenize(resume_text)
    df = load_job_knowledge_base("job_knowledge_base.csv")
    scoring = []
    for idx, row in df.iterrows():
        score = 0
        for col in ["keywords", "skills", "tools", "education_required"]:
            col_tokens = normalize_and_tokenize(str(row[col]))
            score += len(resume_tokens & col_tokens)
        scoring.append((row["job_title"], score))
    # Sort by score, descending
    ranked = sorted(scoring, key=lambda x: x[1], reverse=True)
    # Only return jobs with nonzero score
    result = [
        {"title": title, "confidence": score, "confidence_percent": f"{score} matches"}
        for title, score in ranked if score > 0
    ][:top_n]
    return result if result else [{"title": "Unknown", "confidence": 0, "confidence_percent": "0 matches"}]

# Get recommended courses
def get_recommended_courses(job_title):
    try:
        df = load_job_knowledge_base("job_knowledge_base.csv")
        row = df[df["job_title"].str.lower() == job_title.strip().lower()]
        if not row.empty:
            courses = row.iloc[0]["recommended_courses"]
            return [course.strip() for course in str(courses).split(";") if course.strip()]
        return []
    except Exception as e:
        print(f"Error while fetching recommended courses: {e}")
        return []

# Optionally: Utility to get rich job info for UI (courses, tools, etc)
def get_job_info(job_title):
    try:
        df = load_job_knowledge_base("job_knowledge_base.csv")
        row = df[df["job_title"].str.lower() == job_title.strip().lower()]
        if not row.empty:
            row = row.iloc[0]
            return {
                "job_title": row["job_title"],
                "recommended_courses": [c.strip() for c in str(row["recommended_courses"]).split(";") if c.strip()],
                "skills": [s.strip() for s in str(row["skills"]).split(",") if s.strip()],
                "tools": [t.strip() for t in str(row["tools"]).split(",") if t.strip()],
                "education_required": [e.strip() for e in str(row["education_required"]).split(";") if e.strip()],
                "domain": row["domain"],
            }
        return {}
    except Exception as e:
        print(f"Error while fetching job info: {e}")
        return {}