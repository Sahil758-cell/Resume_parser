import docx
import os
import glob
from resume_parser import analyze_resume

# Raw text from the user's example
raw_text = """ / 
 / 
 / 
ABHISHEK PATIL
9022413246
abhishekpatil172002@gmail.com
www.linkedin.com/in/abhishekdpatil9
Pune
OBJECTIVE
Aspiring Software Engineer with expertise in Java, SQL, and web development. Passionate about coding and problem-solving,
 seeking a challenging role to contribute innovative solutions.
EDUCATION
Bachelor of Engineering in Computer Engineering
Pune District Education Associations's College of Engineering,Manjari,Pune
2021 - 2025 
CGPA
8.64
10
HSC
Shri P. B. Patil Junior College Mudal
2020 - 2021 
PERCENTAGE
93.5
100
SSC
Shri P. B. Patil Highschool Mudal
2018 - 2019 
PERCENTAGE
89.6
100
PROJECTS
Covid-19 Information Website
2025 
Created a dynamic Covid-19 informational website that presented real-time updates,user registration, safety measures, and 
vaccination information to inform users.
Handles authentication, user registration, and database management using Spring Boot, Hibernate, and MySQL
E-Commerce Website
2024 
It enhances user experience with a feature allowing customers to check real-time locker availability.
It gives a user-friendly platform where customers can retrieve their information such as transaction history, account
 details, etc.
SKILLS
Programming Languages
Java, SQL,Python
Web Technologies
HTML, CSS,JavaScript, Bootstrap, React 
JS
Database
MySQL
Frameworks
Spring Boot, Hibernate, JSP, Servlets
Developer Tools
VS Code, Eclipse, JyputerNotebook
CERTIFICATES
Java Full Stack Development - YESS INFOTECH
2025 
Cloud Computing Exam Certified by NPTEL
2024 
LANGUAGES
English, Hindi, Marathi
"""

# Create a docx file
doc = docx.Document()
doc.add_paragraph(raw_text)
test_file_path = "test_resume.docx"
doc.save(test_file_path)

print(f"Created test file: {test_file_path}")

# Analyze the resume
result = analyze_resume(test_file_path)

if result:
    location = result.get("personal_info", {}).get("location")
    print(f"Extracted Location: {location}")
else:
    print("Analysis failed.")

# Clean up the test file
if os.path.exists(test_file_path):
    os.remove(test_file_path)
    print(f"Removed test file: {test_file_path}")

# Also remove the generated json file
list_of_files = glob.glob('outputs/*.json')
if list_of_files:
    # Sort files by creation time and remove the latest one
    latest_file = max(list_of_files, key=os.path.getctime)
    if os.path.exists(latest_file):
        os.remove(latest_file)
        print(f"Removed analysis report: {latest_file}")