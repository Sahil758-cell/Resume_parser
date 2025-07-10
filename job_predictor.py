import fitz  # PyMuPDF
import docx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import re
from nltk.tokenize import word_tokenize
import nltk
import nltk.data
nltk.data.load('tokenizers/punkt/english.pickle')
import numpy as np

nltk.download('punkt', quiet=True)
nltk.download('punky_tab',quiet=True)

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

    # Step 1: Semantic similarity scoring
    semantic_scores = [0.0] * len(job_titles)
    if semantic_model:
        try:
            resume_embedding = semantic_model.encode(resume_text, convert_to_tensor=True)
            job_embeddings = semantic_model.encode(descriptions, convert_to_tensor=True)
            semantic_scores = util.cos_sim(resume_embedding, job_embeddings)[0].cpu().numpy()
            semantic_scores = [float(score) for score in semantic_scores]
        except Exception as e:
            print(f"⚠️ Semantic model error: {str(e)}. Using keyword-based scoring.")

    # Step 2: Token-based similarity scoring
    resume_tokens = normalize_and_tokenize(resume_text)
    token_scores = [0.0] * len(job_titles)
    
    for i, desc in enumerate(descriptions):
        desc_tokens = normalize_and_tokenize(desc)
        if desc_tokens:
            intersection = len(resume_tokens & desc_tokens)
            union = len(resume_tokens | desc_tokens)
            token_scores[i] = intersection / union if union > 0 else 0.0

    # Step 3: Enhanced skill and keyword matching with domain-specific weighting
    combined_scores = [0.0] * len(job_titles)
    
    for i, row in df.iterrows():
        # Base score: combination of semantic and token similarity
        semantic_weight = 0.4 if semantic_model else 0.0
        token_weight = 0.6 if semantic_model else 1.0
        base_score = semantic_weight * semantic_scores[i] + token_weight * token_scores[i]
        
        # Extract skills and keywords for this job
        skills = [skill.strip().lower() for skill in str(row['skills']).split(',') if skill.strip()]
        keywords = [kw.strip().lower() for kw in str(row['keywords']).split(';') if kw.strip()]
        tools = [tool.strip().lower() for tool in str(row['tools']).split(',') if tool.strip()]
        
        # Separate domain-specific vs generic skills
        job_title_lower = row['job_title'].lower()
        
        # Domain-specific keywords that should have high weight
        domain_specific_terms = []
        if 'ios' in job_title_lower:
            domain_specific_terms = ['swift', 'xcode', 'ios', 'objective-c', 'swiftui']
        elif 'android' in job_title_lower:
            domain_specific_terms = ['kotlin', 'android', 'xml', 'android studio']
        elif 'web' in job_title_lower or 'frontend' in job_title_lower or 'backend' in job_title_lower:
            domain_specific_terms = ['html', 'css', 'javascript', 'react', 'angular', 'node.js']
        elif 'data scientist' in job_title_lower:
            domain_specific_terms = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter']
        elif 'ai' in job_title_lower or 'ml' in job_title_lower:
            domain_specific_terms = ['tensorflow', 'pytorch', 'deep learning', 'neural networks', 'keras']
        elif 'mechanical' in job_title_lower:
            domain_specific_terms = ['autocad', 'solidworks', 'thermodynamics', 'manufacturing']
        elif 'electrical' in job_title_lower:
            domain_specific_terms = ['circuit', 'plc', 'power systems', 'electrical']
        elif 'data entry' in job_title_lower:
            domain_specific_terms = ['typing', 'excel', 'data entry', 'spreadsheets']
        
        # Count domain-specific matches (high weight)
        domain_matches = sum(1 for term in domain_specific_terms if term in resume_text)
        
        # Count general skill matches (lower weight)
        skill_matches = sum(1 for skill in skills if skill in resume_text)
        keyword_matches = sum(1 for kw in keywords if kw in resume_text)
        tool_matches = sum(1 for tool in tools if tool in resume_text)
        
        # If job has domain-specific requirements but resume has no domain matches, heavily penalize
        domain_penalty = 1.0
        if domain_specific_terms and domain_matches == 0:
            # Strong penalty for missing all domain-specific skills
            domain_penalty = 0.3
        elif domain_specific_terms and domain_matches > 0:
            # Bonus for having domain-specific skills
            domain_penalty = 1.2
        
        # Calculate weighted match ratio
        total_job_terms = len(skills) + len(keywords) + len(tools)
        if total_job_terms > 0:
            general_match_ratio = (skill_matches + keyword_matches + tool_matches) / total_job_terms
        else:
            general_match_ratio = 0.0
        
        # Domain-specific bonus
        domain_bonus = domain_matches * 0.15 if domain_specific_terms else 0.0
        
        # Combine scores with domain weighting
        match_score = (general_match_ratio * 0.3 + domain_bonus) * domain_penalty
        final_score = base_score * 0.5 + match_score * 0.5
        
        # Small boost for project experience (universal indicator)
        if any(term in resume_text for term in ['project', 'developed', 'designed', 'built', 'created']):
            final_score += 0.05
        
        # Small boost for relevant education
        education_keywords = str(row['education_required']).lower()
        if any(edu in resume_text for edu in education_keywords.split(';') if edu.strip()):
            final_score += 0.03
        
        combined_scores[i] = min(1.0, final_score)

    # Step 4: Normalize scores to prevent all-high or all-low predictions
    if combined_scores:
        max_score = max(combined_scores)
        if max_score > 0:
            # Scale scores to use full range while maintaining relative differences
            combined_scores = [score / max_score * 0.85 for score in combined_scores]

    print(f"📄 Resume word count: {len(resume_text.split())}")
    print("📊 Top job scores:")
    
    # Show top scores for debugging
    temp_ranked = sorted(zip(job_titles, combined_scores), key=lambda x: x[1], reverse=True)[:10]
    for title, score in temp_ranked:
        if score > 0.1:  # Show scores above 10%
            print(f"  - {title}: {round(score, 3)}")

    # Step 5: Final ranking and result formatting
    ranked = sorted(
        zip(job_titles, combined_scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    results = [
        {
            "title": title,
            "confidence": round(float(score), 2),
            "confidence_percent": f"{round(float(score) * 100)}%"
        }
        for title, score in ranked
    ]
    
    # Return "Unknown" only if really no good matches
    if not results or max(combined_scores) < 0.1:
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