import fitz  # PyMuPDF
import docx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import re
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

nltk.download('punkt', quiet=True)

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
def load_job_knowledge_base(file_path="job_knowledge_base.csv"):
    try:
        df = pd.read_csv(file_path)
        df = df.fillna('')
        df = df.groupby('job_title').agg({
            'keywords': lambda x: ';'.join(set(';'.join(x).split(';'))),
            'recommended_courses': lambda x: ';'.join(set(';'.join(x).split(';'))),
            'skills': lambda x: ','.join(set(','.join(x).split(','))),
            'education_required': lambda x: ';'.join(set(';'.join(x).split(';'))),
            'domain': 'first',
            'tools': lambda x: ','.join(set(','.join(x).split(','))),
            'courses': lambda x: ','.join(set(','.join(x).split(',')))
        }).reset_index()
        return df
    except Exception as e:
        print(f"❌ Error loading job knowledge base: {str(e)}")
        return None

try:
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print("Could not load sentence-transformer model, falling back to TF-IDF.")
    semantic_model = None

def normalize_and_tokenize(text):
    text = re.sub(r'[^\w\s]', '', text.lower()).strip()
    return set(word_tokenize(text))


# Predict job field using semantic similarity and robust keyword/token matching
def predict_job_field(resume_text, top_n=3):
    resume_text = ' '.join(resume_text.lower().split())
    if not resume_text:
        return [{"title": "Unknown", "confidence": 0.0, "confidence_percent": "0%"}]

    df = load_job_knowledge_base()
    if df is None:
        return [{"title": "Unknown", "confidence": 0.0, "confidence_percent": "0%"}]

    job_titles = df["job_title"].tolist()
    descriptions = [
        f"{row['job_title']}. {row['keywords']} {row['skills']} {row['tools']} {row['education_required']}"
        for _, row in df.iterrows()
    ]

    semantic_scores = [0.0] * len(job_titles)
    token_scores = [0.0] * len(job_titles)

    if semantic_model:
        try:
            resume_embedding = semantic_model.encode(resume_text, convert_to_tensor=True)
            job_embeddings = semantic_model.encode(descriptions, convert_to_tensor=True)
            semantic_scores = util.cos_sim(resume_embedding, job_embeddings)[0].cpu().numpy()
        except Exception as e:
            print(f"⚠️ Semantic model error: {str(e)}. Using token-based scoring.")

    resume_tokens = normalize_and_tokenize(resume_text)
    for i, desc in enumerate(descriptions):
        desc_tokens = normalize_and_tokenize(desc)
        intersection = len(resume_tokens & desc_tokens)
        union = len(resume_tokens | desc_tokens)
        token_scores[i] = intersection / union if union > 0 else 0.0

    combined_scores = [
        (0.7 * s + 0.3 * t) if semantic_model else t
        for s, t in zip(semantic_scores, token_scores)
    ]

    for i, row in df.iterrows():
        skill_matches = sum(1 for skill in row['skills'].split(',') if skill.strip().lower() in resume_text)
        keyword_matches = sum(1 for kw in row['keywords'].split(';') if kw.strip().lower() in resume_text)
        match_boost = (skill_matches * 0.2 + keyword_matches * 0.1) / 10.0
        combined_scores[i] = min(1.0, combined_scores[i] + match_boost)

    # === Step 1: Penalize low-quality resumes ===
    penalty = 1.0
    if len(resume_text.split()) < 150:
        penalty *= 0.8
    weak_signals = ["project", "experience", "internship", "skills", "training"]
    if not any(word in resume_text.lower() for word in weak_signals):
        penalty *= 0.8
    combined_scores = [round(score * penalty, 4) for score in combined_scores]

    print(f"📄 Resume word count: {len(resume_text.split())}")
    print(f"🛑 Weak signals missing: {not any(w in resume_text for w in weak_signals)}")
    print("📊 Top job scores:")
    for i, title in enumerate(job_titles):
        print(f"  - {title}: sem={round(semantic_scores[i], 3)}, tok={round(token_scores[i], 3)}, final={round(combined_scores[i], 3)}")

    for i, row in df.iterrows():
        job_title = row['job_title'].lower()
        keywords = [kw.strip().lower() for kw in row['keywords'].split(';') if kw.strip()]
        if any(kw in resume_text for kw in keywords):
            combined_scores[i] = min(1.0, combined_scores[i] + 0.05)

    ranked = sorted(
        zip(job_titles, combined_scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    results = [
        {
            "title": title,
            "confidence": round(score, 2),
            "confidence_percent": f"{round(score * 100)}%"
        }
        for title, score in ranked
    ]
    is_weak_resume = (
    len(resume_text.split()) < 100 or
    not any(word in resume_text.lower() for word in ["project", "experience", "internship", "skills", "training"])
    )
    if is_weak_resume and max(combined_scores) < 0.25:
        return [{"title": "Unknown", "confidence": 0.0, "confidence_percent": "0%"}]

    if not results or max(combined_scores) < 0.2:
        return [{"title": "Unknown", "confidence": 0.0, "confidence_percent": "0%"}]

    return results


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