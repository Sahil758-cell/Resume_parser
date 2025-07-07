import csv

def debug_job_prediction(resume_text):
    resume_text = resume_text.lower()
    score_dict = {}
    total_weights_dict = {}
    job_keyword_counts = {}
    generic_keywords = {'training', 'tools'}  

    print("=== DEBUGGING JOB PREDICTION ===")
    print(f"Resume text length: {len(resume_text)}")
    print(f"Resume text sample: {resume_text[:200]}...")
    print()

    try:
        with open("job_knowledge_base.csv", newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                job = row["job_title"].strip().lower()
                keywords = [kw.strip().lower() for kw in row["keywords"].split(",") if kw.strip()]
                
                # Calculate score and total weight
                score = 0.0
                total_weight = 0.0
                matched_keywords = []
                
                for kw in keywords:
                    adjusted_weight = 0.5 if kw in generic_keywords else 1.0
                    total_weight += adjusted_weight
                    if kw in resume_text:
                        score += adjusted_weight
                        matched_keywords.append(kw)

                score_dict[job] = score
                total_weights_dict[job] = total_weight
                job_keyword_counts[job] = len(keywords)
                
                # Print details for Software Tester and Customer Support
                if job in ['software tester', 'customer support']:
                    print(f"=== {job.upper()} ===")
                    print(f"Total keywords: {len(keywords)}")
                    print(f"Matched keywords: {matched_keywords}")
                    print(f"Score: {score}")
                    print(f"Total weight: {total_weight}")
                    if total_weight > 0:
                        normalized_score = score / total_weight
                        print(f"Normalized score: {normalized_score:.4f}")
                    print()

        # Normalize scores
        normalized_scores = {}
        for job, score in score_dict.items():
            total_weight = total_weights_dict.get(job, 1.0)
            normalized_scores[job] = score / total_weight if total_weight > 0 else 0.0

        # Find max score
        max_score = max(normalized_scores.values(), default=0.0)
        top_jobs = [job for job, score in normalized_scores.items() if score == max_score]

        print("=== TOP 5 JOBS BY SCORE ===")
        sorted_jobs = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (job, score) in enumerate(sorted_jobs[:5]):
            print(f"{i+1}. {job}: {score:.4f}")

        # Resolve ties
        predicted = top_jobs[0] if top_jobs else "unknown"
        if len(top_jobs) > 1:
            for job in top_jobs:
                if job in resume_text:
                    predicted = job
                    break
            else:
                predicted = min(top_jobs, key=lambda j: job_keyword_counts[j])

        predicted = ' '.join(word.capitalize() for word in predicted.split())
        print(f"\n=== FINAL PREDICTION ===")
        print(f"Predicted job: {predicted}")
        return predicted

    except FileNotFoundError:
        print("Error: job_knowledge_base.csv not found.")
        return "Unknown"
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Unknown"

# Test with the QA tester resume
test_resume = """FIRST LAST
Bay Area, California • +1-234-456-789 • professionalemail@resumeworded.com • linkedin.com/in/username
PROFESSIONAL EXPERIENCE
Resume Worded, New York, NY
Jun 2018 – Present
QA Manual Tester
• Enabled critical test case complexity metrics with support for Rapid adoption of functional automation using
a scriptless test case adaptor by standardizing a Test Case construction method that was built
Automation-ready & supported a test automation framework leading to a 45% increase in reusability with
reductions in TCO approaching 25%.
• Optimized scripting, modularity, & maintenance which resulted in an 18% decrease in workflow friction.
• Increased the company's ability to take and complete projects without increasing manpower by 15% by
reducing QA testing turnaround time by 30%.
Growthsi, New York, NY
Jan 2015 – May 2018
QA Manual Tester
• Restructured utilities & improved the process documentation leading to a 40% reduction in client support
tickets & an 80% increase in uptime.
• Achieved department-wide improvement metrics based on QA scorecard through the 46% & 22% workload
reduction of the customer support & IT departments respectively.
• Standardized Test Plan, Test Scripts/Test Cases, Daily Status Reports, etc., documents leading to a 20%
increase in productivity.
• Established monthly sprint backlog items as well as performed agile meetings while updating the activities in
Microsoft TFS in an optimized manner which resulted in saving 10 hours of monthly lost time.
RW Capital, San Diego, CA
May 2008 – Dec 2014
QA Manual Tester (Nov 2011 – Dec 2014)
• Optimized the build process by increasing the system's quality level and reducing 45% of defects found.
• Established proper team communication that identified, triaged, reproduced, & fixed found issues using JIRA
increasing the overall workflow by 25%.
Junior QA Manual Tester (May 2010 – Oct 2011)
• Wrote & optimized test scripts in towels which led to a 9% reduction in the overall testing hours.
• Created traceability matrix to fill in the gap between requirements and tests covered contributing to the 10%
increase in test case count.
EDUCATION
Resume Worded University, San Francisco, CA
May 2010
BSc. Computer Science
SKILLS
•Test Automation
Frameworks
•Java
•Javascript
•
Microsoft TFS
•
CharlesProxy
•
SQL/NoSQL
•
Selenium/Webdriver
•
TestNG
•
JIRA
•
Jenkins
•
Agile
•
Source Versioning
•
JUnit
•
Scrum
•
…"""

result = debug_job_prediction(test_resume) 