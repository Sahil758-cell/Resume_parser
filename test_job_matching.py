import csv
import re

def test_job_matching():
    # Sahil's skills from the JSON
    sahil_skills = {
        "python", "machine learning", "pandas", "numpy", "tensorflow", 
        "pytorch", "deep learning", "computer vision", "opencv", 
        "flask", "streamlit", "data visualization", "data preprocessing", 
        "java", "html", "communication", "safety"
    }
    
    print("=== SAHIL'S SKILLS ===")
    print(f"Total skills: {len(sahil_skills)}")
    print(f"Skills: {', '.join(sorted(sahil_skills))}")
    print()
    
    # Read job knowledge base
    job_matches = []
    
    with open("job_knowledge_base.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            job_title = row["job_title"].strip()
            keywords_str = row["keywords"].strip()
            
            if not keywords_str:
                continue
                
            # Extract keywords
            keywords = [kw.strip().lower() for kw in keywords_str.split(",") if kw.strip()]
            
            # Find matching skills
            matches = []
            for skill in sahil_skills:
                for keyword in keywords:
                    if skill.lower() in keyword.lower() or keyword.lower() in skill.lower():
                        matches.append(skill)
                        break
            
            if matches:
                match_percentage = (len(matches) / len(keywords)) * 100
                job_matches.append({
                    'job_title': job_title,
                    'keywords': keywords,
                    'matches': matches,
                    'match_count': len(matches),
                    'total_keywords': len(keywords),
                    'match_percentage': match_percentage
                })
    
    # Sort by match percentage
    job_matches.sort(key=lambda x: x['match_percentage'], reverse=True)
    
    print("=== JOB MATCHES ===")
    for i, job in enumerate(job_matches, 1):
        print(f"{i}. {job['job_title']}")
        print(f"   Keywords: {', '.join(job['keywords'])}")
        print(f"   Matches: {', '.join(job['matches'])}")
        print(f"   Match: {job['match_count']}/{job['total_keywords']} ({job['match_percentage']:.1f}%)")
        print()
    
    # Show top 3 matches
    print("=== TOP 3 RECOMMENDATIONS ===")
    for i, job in enumerate(job_matches[:3], 1):
        print(f"{i}. {job['job_title']} - {job['match_percentage']:.1f}% match")
    
    return job_matches

if __name__ == "__main__":
    test_job_matching() 