from sentence_transformers import SentenceTransformer, util
import json
import csv

def debug_actual_resume():
    # Load the embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load the actual resume text from JSON
    with open("outputs/resume_analysis_report_20250630_095856.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    actual_resume_text = data["raw_text"]
    
    print("=== ACTUAL RESUME TEXT (first 500 chars) ===")
    print(actual_resume_text[:500])
    print("...")
    print()
    
    # Encode actual resume text
    resume_embedding = embedding_model.encode(actual_resume_text, convert_to_tensor=True)
    
    # Test against job keywords
    job_scores = {}
    
    with open("job_knowledge_base.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            job_title = row["job_title"].strip()
            keywords = row["keywords"].strip()
            
            if not keywords:
                continue
            
            # Encode keywords
            keyword_embedding = embedding_model.encode(keywords, convert_to_tensor=True)
            similarity = util.cos_sim(resume_embedding, keyword_embedding).item()
            job_scores[job_title] = similarity
    
    # Sort by similarity score
    sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("=== SENTENCE-BERT SIMILARITY SCORES (ACTUAL RESUME) ===")
    print(f"Threshold: 0.40")
    print()
    
    for i, (job, score) in enumerate(sorted_jobs[:10], 1):
        status = "✓ MATCH" if score >= 0.40 else "✗ BELOW THRESHOLD"
        print(f"{i}. {job}: {score:.4f} {status}")
    
    print()
    print("=== ANALYSIS ===")
    max_score = max(job_scores.values()) if job_scores else 0
    print(f"Highest score: {max_score:.4f}")
    print(f"Above threshold: {sum(1 for score in job_scores.values() if score >= 0.40)} jobs")
    
    if max_score < 0.40:
        print("❌ PROBLEM: All scores are below the 0.40 threshold!")
        print("   This explains why 'No matching job found' is returned.")
    else:
        print("✓ SOLUTION: There are jobs above the threshold.")
        top_job = sorted_jobs[0][0]
        print(f"   Top match should be: {top_job}")

if __name__ == "__main__":
    debug_actual_resume() 