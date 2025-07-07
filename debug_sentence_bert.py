from sentence_transformers import SentenceTransformer, util
import csv

def debug_sentence_bert_similarity():
    # Load the embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Sahil's resume text (simplified version focusing on skills)
    sahil_resume_text = """
    Computer Engineering graduate with strong programming expertise in Data Science, 
    Artificial Intelligence, Machine Learning, and Software Development. 
    Skilled in Python, C#, and Java with hands-on project experience. 
    Programming: Python (NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch)
    ML Algorithms: Regression, Decision Trees, Random Forests, SVM, K-Means, Gradient Boosting
    Data Preprocessing: Feature engineering, handling missing data
    Model Deployment: Flask, Streamlit
    Visualization: pandas, Matplotlib, Seaborn, Plotly
    Computer vision, OpenCV, deep learning, machine learning, data visualization
    """
    
    print("=== SAHIL'S RESUME TEXT ===")
    print(sahil_resume_text.strip())
    print()
    
    # Encode resume text
    resume_embedding = embedding_model.encode(sahil_resume_text, convert_to_tensor=True)
    
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
    
    print("=== SENTENCE-BERT SIMILARITY SCORES ===")
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

if __name__ == "__main__":
    debug_sentence_bert_similarity() 