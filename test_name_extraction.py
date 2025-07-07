import re

def search_for_name_in_text(full_text, blacklist, invalid_name_lines):
    """
    Search for names throughout the entire resume text using various patterns, skipping institution/organization lines.
    """
    institution_keywords = [
        'college', 'university', 'polytechnic', 'foundation', 'school', 'institute', 'academy', 'faculty', 'company', 'organization', 'institute', 'department'
    ]
    def is_institution_line(line):
        return any(kw in line.lower() for kw in institution_keywords)

    # Pattern 1: Look for two consecutive capitalized words (likely first and last name)
    name_pattern_1 = re.compile(r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b')
    lines = full_text.split('\n')
    # Search top and bottom lines for likely names
    search_ranges = [range(min(15, len(lines))), range(max(0, len(lines)-15), len(lines))]
    for search_range in search_ranges:
        for i in search_range:
            line = lines[i].strip()
            if is_institution_line(line):
                continue
            matches_1 = name_pattern_1.findall(line)
            for match in matches_1:
                first, last = match
                if (len(first) >= 2 and len(last) >= 2 and
                    first.lower() not in blacklist and last.lower() not in blacklist and
                    first not in invalid_name_lines and last not in invalid_name_lines):
                    return f"{first.title()} {last.title()}"
    
    # Pattern 2: Look for three consecutive capitalized words (first, middle, last name)
    name_pattern_2 = re.compile(r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b')
    for search_range in search_ranges:
        for i in search_range:
            line = lines[i].strip()
            if is_institution_line(line):
                continue
            matches_2 = name_pattern_2.findall(line)
            for match in matches_2:
                first, middle, last = match
                if (len(first) >= 2 and len(middle) >= 2 and len(last) >= 2 and
                    first.lower() not in blacklist and middle.lower() not in blacklist and last.lower() not in blacklist and
                    first not in invalid_name_lines and middle not in invalid_name_lines and last not in invalid_name_lines):
                    return f"{first.title()} {middle.title()} {last.title()}"
    
    # Pattern 3: Look for single capitalized words that might be names (check context)
    for search_range in search_ranges:
        for i in search_range:
            line = lines[i].strip()
            if is_institution_line(line):
                continue
            if line and line.isalpha() and len(line) >= 3 and len(line) <= 15:
                if (line[0].isupper() and line[1:].islower() and
                    line.lower() not in blacklist and
                    line not in invalid_name_lines):
                    # Check if next line also looks like a name
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if (next_line and next_line.isalpha() and len(next_line) >= 3 and len(next_line) <= 15 and
                            next_line[0].isupper() and next_line[1:].islower() and
                            next_line.lower() not in blacklist and
                            next_line not in invalid_name_lines and not is_institution_line(next_line)):
                            return f"{line.title()} {next_line.title()}"
    
    # Pattern 4: Look for any line with two words, both starting with uppercase, not an institution (robust, fallback to last match)
    last_valid_name = None
    for search_range in search_ranges:
        for i in search_range:
            line = lines[i].strip()
            line = ' '.join(line.split())  # Normalize spaces
            if is_institution_line(line):
                continue
            words = line.split()
            if len(words) == 2:
                first, last = words
                if (first[0].isupper() and last[0].isupper() and
                    first.lower() not in [k.lower() for k in blacklist] and last.lower() not in [k.lower() for k in blacklist] and
                    first not in invalid_name_lines and last not in invalid_name_lines and
                    first.isalpha() and last.isalpha() and
                    2 <= len(first) <= 20 and 2 <= len(last) <= 20):
                    last_valid_name = f"{first.title()} {last.title()}"
                    # Prefer first match, but keep looking for a better one
    if last_valid_name:
        return last_valid_name

    # FINAL fallback: scan entire resume from bottom to top for a likely name line
    for line in reversed(lines):
        line = line.strip()
        line = ' '.join(line.split())  # Normalize spaces
        if is_institution_line(line):
            continue
        words = line.split()
        if len(words) == 2:
            first, last = words
            if (first[0].isupper() and last[0].isupper() and
                first.lower() not in [k.lower() for k in blacklist] and last.lower() not in [k.lower() for k in blacklist] and
                first not in invalid_name_lines and last not in invalid_name_lines and
                first.isalpha() and last.isalpha() and
                2 <= len(first) <= 20 and 2 <= len(last) <= 20):
                return f"{first.title()} {last.title()}"

    return None

# Test with the problematic resume text
test_text = """ Git 
Python(Basic) 
IOT 
Programming 
 
 
 
 
 
 
 
 
 
 
 
bagaleasavari@gmail.com 
 
8698985131 
 
Solapur, Maharashtra 
 
github.com/TechAshi-bagale 
 
https://www.linkedin.com/in/ 
asavari-bagale-81a1aa27b 
 
 
 
SKILLS 
 EDUCATION 
 
B.Tech CSE(2025) 
Shree Siddheshwar Women's College Of Engineering, Solapur 
11/2022 - Present 
 
Diploma in Co(2022) 
Kai.Kalyanrao (Balasaheb)Ingale Polytechnic Akkalkot, Solapur 
82.17% 
 
 
 
HSC in Science(2020) 
C.B.Khedgi's Basaveshwar Science Akkalkot, Solapur 
59.23% 
 
 
 
SSC(2018) 
K.P.Gaikwad High School Badole bk , Solapur 
86.40% 
 
 
 
 
 
 
 
LANGUAGES 
Hindi 
Full Professional Proficiency 
 
English 
Intermediate Proficiency 
 
Kannada 
Native or Bilingual Proficiency 
 
Marathi 
Full Professional Proficiency 
 
COURSES 
 
 
Full Stack Java by FUEL 
 
 
 
Infosys Springboard 
Python Programming 
Certification. 
 
INTERESTS 
 
  INTERNSHIP 
 
Hybrid Application Development 
D K Techno's 
07/2024 - 08/2024 
Achievements/Tasks 
 Attended 6 weeks industrial training in Hybrid Application Development using Flutter Technology. 
 Collaborated with team member to design and deploy cross-platform applications using flutter . 
 
Full Stack Java  
Anudip Foundation 
Achievements/Tasks 
 Developed responsive web pages using HTML,CSS. 
 Used GitHub for version control and efficient collaboration within the development team. 
 
PROJECTS 
 
LeaveGo App 
Technology used: Flutter, Dart and Firebase 
 Hybrid application which reduces paperwork &, allows students to request, track, and manage their leave status digitally. 
 Integrated Firebase as the database to manage student and staff data. 
Student Performance Predictor 
Technology used: Python, Pandas, Scikit-learn, Spyder 
 Implemented a regression model to predict student grades based on student study time, parental education and other 
factors with 85%+ accuracy. 
 
ACHIEVEMENTS 
 
I volunteered for the 'Technical Reel Making' event at a national-level technical competition. 
I am part of the Infosys springboard Pragati: Path to Future Program Cohort-4(Jan-Apr2025). 
AI/ML 
Core java 
Spyder 
Asavari Bagale 
Seeking a job role where i can enhance my skills, strengthen my knowledge by exploring new things. 
Web page designing """

blacklist = {'mobile', 'email', 'contact', 'phone', 'linkedin', 'github', 'education', 'educational details', 'experience', 'skills', 'projects', 'career', 'objectives', 'objective', 'summary', 'profile', 'certifications', 'achievements', 'personal details', 'languages known', 'address', 'date of birth', 'major', 'minor', 'declaration', 'place', 'date', 'css', 'soft skills', 'html', 'java', 'python', 'sql', 'linux', 'mysql', 'machine learning', 'data science', 'web development', 'programming languages', 'computer engineering', 'data visualization', 'artificial intelligence', 'git', 'local address', 'permanent address', 'accomplishments', 'tools', 'technical skills', 'languages', 'interests', 'responsibilities', 'internship', 'hobbies', 'statement', 'statement of purpose', 'curriculum vitae', 'resume', 'biodata', 'reference', 'references', 'web page designing', 'seeking a job role','9082350675','/'}

invalid_name_lines = {'/', '-', '|', '\\', '*', '~', ''}

result = search_for_name_in_text(test_text, blacklist, invalid_name_lines)
print(f"Name found: {result}")

# Let's also check what lines contain "Asavari Bagale"
lines = test_text.split('\n')
for i, line in enumerate(lines):
    if 'Asavari' in line or 'Bagale' in line:
        print(f"Line {i}: '{line}'") 