import re
import os
import csv
import json
import spacy
from datetime import datetime
from collections import OrderedDict
from job_predictor import extract_text_from_pdf, extract_text_from_docx, predict_job_field, get_recommended_courses
from image_pdf_text_extractor import extract_text_from_image_pdf
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.pipelines import pipeline
import unicodedata
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

try:
    client = MongoClient(MONGO_URI, 
                        serverSelectionTimeoutMS=5000,  # 5 second timeout
                        connectTimeoutMS=5000,
                        socketTimeoutMS=5000)
    db = client[DB_NAME]
    jobs_collection = db['ups_resume_analyzers']
    # Test the connection
    client.admin.command('ping')
    print("✅ resume_parser.py connected to MongoDB.")
except Exception as e:
    print(f"❌ resume_parser.py failed to connect to MongoDB: {e}")
    jobs_collection = None

# === Skill Keywords (from MongoDB) ===
SKILL_KEYWORDS = set()
if jobs_collection is not None: 
    try:
        # Fetch only the 'keywords' field from all documents
        all_jobs = jobs_collection.find({}, {"keywords": 1, "_id": 0})
        for job in all_jobs:
            if job.get("keywords"):
                keywords = [kw.strip().lower() for kw in job["keywords"].split(",") if kw.strip()]
                SKILL_KEYWORDS.update(keywords)
    except Exception as e:
        print(f"❌ MongoDB error fetching skills: {e}")


nlp = spacy.load("en_core_web_trf")

# Explicitly set Tesseract path (optional, remove if PATH is working)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Ensure poppler-utils is in PATH
os.environ["PATH"] += os.pathsep + r"C:\\poppler-24.08.0\\Library\\bin"

# === Skill Keywords ===
SKILL_KEYWORDS = set()
try:
    with open("job_knowledge_base.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            keywords = [kw.strip().lower() for kw in row["keywords"].split(",") if kw.strip()]
            SKILL_KEYWORDS.update(keywords)
except FileNotFoundError:
    print("❌ job_knowledge_base.csv not found. Skills detection may be limited.")

# === Degree Keywords ===
G_TERMS = [
    r"b[.\s]*s[.\s]*c[.]*", r"bsc", r"b[.\s]*a[.]*", r"ba", r"b[.\s]*b[.\s]*a[.]*", r"bba", r"b[.\s]*c[.\s]*a[.]*", r"bca", r"b[.\s]*c[.\s]*o[.\s]*m[.]*", r"bcom", r"b[.\s]*e[.\s]*", r"be", r"bachelor", r"ug", r"undergraduate"
]
UG_TERMS = [
    r"diploma", r"polytechnic", r"d\.pharma", r"diploma in pharmacy", r"iti", r"industrial training institute"
]
PG_TERMS = [
    r"m[.\s]*s[.\s]*c[.]*", r"msc",
    r"m[.\s]*b[.\s]*a[.]*", r"mba",
    r"m[.\s]*c[.\s]*a[.]*", r"mca",
    r"m[.\s]*t[.\s]*e[.\s]*c[.\s]*h[.]*", r"mtech",
    r"m[.\s]*e[.]*", r"me",
    r"m[.\s]*a[.]*", r"ma",
    r"m[.\s]*c[.\s]*o[.\s]*m[.]*", r"mcom",
    r"pg", r"pgd", r"pgdm",
    r"post[- ]?graduate", r"post[- ]?graduation", r"post graduate diploma", r"\bpost[ -]?grad\b"
]
DR_TERMS = [r"phd", r"doctorate", r"dphil"]
HSC_TERMS = [
    r"intermediate", r"hsc", r"12th", r"xii", r"higher secondary", r"10\+2", r"senior secondary"
]
SSC_TERMS = [
    r"ssc", r"10th", r"matriculation", r"secondary", r"x", r"10th standard", r"secondary school certificate", r"matric"
]

# === Rating Weights ===
RATING_WEIGHTS = {
    'personal_info': 20,
    'education': 40,
    'projects': 5,
    'internships': 5,
    'work_experience': 10,
    'skills': 10,
    'languages': 10
}

# Load a pre-trained NER model for education extraction
try:
    tokenizer_edu = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model_edu = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    ner_pipe_edu = pipeline("ner", model=model_edu, tokenizer=tokenizer_edu, aggregation_strategy="simple")  # type: ignore
except Exception as e:
    print(f"❌ Could not load NER model for education extraction: {e}")
    ner_pipe_edu = None

def extract_education_entities(text):
    if not ner_pipe_edu:
        return []
    entities = ner_pipe_edu(text)
    # Filter for likely education entities (customize as needed)
    education_keywords = [
        "b.tech", "mca", "bachelor", "master", "phd", "xii", "x", "diploma", "hsc", "ssc",
        "b.e", "msc", "mba", "pgdm", "pgd", "bca", "bsc", "bcom", "bba", "ba", "be", "matriculation", "secondary", "higher secondary"
    ]
    results = []
    for ent in entities:
        if any(kw.lower() in ent['word'].lower() for kw in education_keywords) or ent['entity_group'] == "ORG":
            results.append(ent)
    return results

# Add this at the top of the file with other constants (if not already present)
section_headers = [
    'educational details', 'profile', 'summary', 'skills', 'projects', 'internships', 'work experience',
    'education', 'languages', 'certifications', 'achievements', 'personal details', 'personal profile',
    'personal information', 'contact', 'objective', 'declaration', 'interests', 'hobbies', 'address',
    'date of birth', 'major', 'minor', 'fresher', 'trainee', 'student'
]

def is_section_header(line):
    return any(header in line.lower() for header in section_headers)

def is_degree(line):
    degree_keywords = G_TERMS + UG_TERMS + PG_TERMS + DR_TERMS + HSC_TERMS + SSC_TERMS
    return any(term in line.lower() for term in degree_keywords)

def is_phone_number(line):
    digits = re.sub(r'[^0-9]', '', line)
    return len(digits) == 10 and digits[0] in '6789'

def search_for_name_in_text(full_text, blacklist, invalid_name_lines):
    """
    Search for names throughout the entire resume text using various patterns, skipping institution/organization lines.
    """
    institution_keywords = [
        'college', 'university', 'polytechnic', 'foundation', 'school', 'institute', 'academy', 'faculty', 'company', 'organization', 'institute', 'department'
    ]
    context_blacklist = ['volunteered', 'event', 'competition', 'making', 'technical']
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
            if any(kw in line.lower() for kw in context_blacklist):
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
            if any(kw in line.lower() for kw in context_blacklist):
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
            print(f"SKIP (institution): '{line}'")
            continue
        words = line.split()
        if len(words) == 2:
            first, last = words
            print(f"CHECK: '{line}' -> {first}, {last}")
            if (first[0].isupper() and last[0].isupper() and
                first.lower() not in [k.lower() for k in blacklist] and last.lower() not in [k.lower() for k in blacklist] and
                first not in invalid_name_lines and last not in invalid_name_lines and
                first.isalpha() and last.isalpha() and
                2 <= len(first) <= 20 and 2 <= len(last) <= 20):
                print(f"FOUND NAME: {first.title()} {last.title()}")
                return f"{first.title()} {last.title()}"
            else:
                print(f"SKIP (not valid): '{line}'")

    return None

def extract_name_from_lines(lines):
    for line in lines[:15]:  # Check the first 15 lines
        line = line.strip()
        # Match two uppercase words, possibly followed by a comma
        match = re.match(r'^([A-Z]{2,})\s+([A-Z]{2,}),?$', line)
        if match:
            return f"{match.group(1).title()} {match.group(2).title()}"
        # Match two words, possibly with a comma, and allow for mixed case
        match = re.match(r'^([A-Za-z]{2,})\s+([A-Za-z]{2,}),?$', line)
        if match:
            return f"{match.group(1).title()} {match.group(2).title()}"
    return None

def robust_extract_name(lines, blacklist=None):
    if blacklist is None:
        blacklist = set([
            'resume', 'curriculum vitae', 'cv', 'profile', 'summary', 'career objective', 'contact', 'email', 'mobile', 'phone',
            'address', 'education', 'academic', 'skills', 'experience', 'projects', 'certifications', 'languages', 'interests',
            'project', 'predictor', 'app', 'system', 'generator', 'website', 'application', 'report', 'analysis', 'certificate',
            'internship', 'achievement', 'interest', 'objective', 'training', 'course', 'workshop', 'seminar', 'conference',
            'award', 'publication', 'activity', 'volunteering', 'team', 'leadership', 'hobby', 'reference','Extra Curricular Activities'        ])
    for line in lines[:20]:
        line = line.strip().strip(",.:;|-_")
        if not line or line.lower() in blacklist:
            continue
        # Skip lines containing any blacklist word as a whole word
        if any(re.search(r'\b' + re.escape(word) + r'\b', line.lower()) for word in blacklist):
            continue
        match = re.match(r'^([A-Z][a-z]+|[A-Z]{2,})\s+([A-Z][a-z]+|[A-Z]{2,})(\s+([A-Z][a-z]+|[A-Z]{2,}))?$', line)
        if match:
            name = " ".join([w.title() for w in line.split()[:3]])
            if name.lower() not in blacklist:
                return name
    return None

def extract_name_from_profile_section(lines):
    # Only extract if a personal info section is present
    personal_info_headers = ['personal details', 'personal profile', 'personal information', 'profile']
    has_personal_info = any(any(header in line.lower() for header in personal_info_headers) for line in lines)
    if not has_personal_info:
        return None
    for line in lines:
        # Remove zero-width spaces and normalize whitespace
        clean_line = re.sub(r'[\u200b\s]+', ' ', line).strip()
        # Match 'Name :' pattern anywhere in the line
        match = re.search(r'name\s*[:\-]?\s*([A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){1,2})', clean_line, re.IGNORECASE)
        if match:
            return match.group(1).title()
    return None

def clean_line(line):
    # Remove zero-width spaces and normalize whitespace
    return re.sub(r'[\u200b\s]+', ' ', line).strip(",.:;|-_ ").strip()

def is_likely_name(line, blacklist):
    words = line.split()
    if 2 <= len(words) <= 3 and all(len(w) > 1 for w in words):
        if line.lower() not in blacklist and not any(re.search(r'\b' + re.escape(word) + r'\b', line.lower()) for word in blacklist):
            # Handle ALL CAPS names (like "BHUTHALE ARAVIND")
            if line.isupper():
                return True
            # Handle title case or mixed case names
            elif not line.islower():
                # Additional check: avoid lines that are clearly not names
                not_name_indicators = ['listening', 'playing', 'watching', 'reading', 'cooking', 'music', 'badminton', 'cricket', 'football', 'basketball', 'tennis', 'swimming', 'dancing', 'singing', 'painting', 'drawing', 'photography', 'traveling', 'gaming', 'gardening', 'yoga', 'meditation', 'gym', 'fitness', 'running', 'cycling', 'hiking', 'fishing', 'hunting', 'camping', 'climbing', 'surfing', 'skiing', 'skating', 'boxing', 'karate', 'judo', 'taekwondo', 'wrestling', 'weightlifting', 'bodybuilding', 'pilates', 'aerobics', 'zumba', 'salsa', 'ballet', 'jazz', 'hip hop', 'classical', 'rock', 'pop', 'jazz', 'blues', 'country', 'folk', 'electronic', 'rap', 'reggae', 'soul', 'r&b', 'metal', 'punk', 'indie', 'alternative', 'experimental', 'ambient', 'trance', 'house', 'techno', 'dubstep', 'trap', 'drum and bass', 'breakbeat', 'garage', 'jungle', 'hardcore', 'industrial', 'gothic', 'emo', 'screamo', 'death metal', 'black metal', 'thrash metal', 'power metal', 'progressive metal', 'symphonic metal', 'folk metal', 'viking metal', 'pagan metal', 'celtic metal', 'oriental metal', 'arabic metal', 'indian metal', 'chinese metal', 'japanese metal', 'korean metal', 'thai metal', 'vietnamese metal', 'filipino metal', 'malaysian metal', 'indonesian metal', 'singaporean metal', 'australian metal', 'new zealand metal', 'canadian metal', 'american metal', 'british metal', 'german metal', 'french metal', 'italian metal', 'spanish metal', 'portuguese metal', 'dutch metal', 'belgian metal', 'swiss metal', 'austrian metal', 'polish metal', 'czech metal', 'slovak metal', 'hungarian metal', 'romanian metal', 'bulgarian metal', 'serbian metal', 'croatian metal', 'slovenian metal', 'bosnian metal', 'montenegrin metal', 'macedonian metal', 'albanian metal', 'greek metal', 'turkish metal', 'armenian metal', 'georgian metal', 'azerbaijani metal', 'kazakh metal', 'uzbek metal', 'turkmen metal', 'kyrgyz metal', 'tajik metal', 'afghan metal', 'pakistani metal', 'bangladeshi metal', 'sri lankan metal', 'nepali metal', 'bhutanese metal', 'maldivian metal', 'myanmar metal', 'laotian metal', 'cambodian metal', 'mongolian metal', 'tibetan metal', 'uyghur metal', 'kazakh metal', 'uzbek metal', 'turkmen metal', 'kyrgyz metal', 'tajik metal', 'afghan metal', 'pakistani metal', 'bangladeshi metal', 'sri lankan metal', 'nepali metal', 'bhutanese metal', 'maldivian metal', 'myanmar metal', 'laotian metal', 'cambodian metal', 'mongolian metal', 'tibetan metal', 'uyghur metal']
                if any(indicator in line.lower() for indicator in not_name_indicators):
                    return False
                return True
    return False

def extract_name_after_personal_section(lines, blacklist):
    section_headers = ['personal details', 'personal profile', 'personal information', 'profile']
    start_idx = 0
    for i, line in enumerate(lines):
        cl = clean_line(line).lower()
        if any(header in cl for header in section_headers):
            start_idx = i
            break
    # Scan next 10 lines after the section header
    for line in lines[start_idx:start_idx+10]:
        cl = clean_line(line)
        # Try 'Name :' pattern (case insensitive)
        match = re.match(r'^name\s*[:\-]?\s*([A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){1,2})', cl, re.IGNORECASE)
        if match:
            name = match.group(1)
            # If name is all caps, convert to title case
            if name.isupper():
                return name.title()
            return name.title()
        # Try likely name pattern
        if is_likely_name(cl, blacklist):
            # If line is all caps, convert to title case
            if cl.isupper():
                return cl.title()
            return cl.title()
    return None

def universal_extract_name(lines, blacklist, location_keywords=None):
    if location_keywords is None:
        location_keywords = ['telangana', 'mumbai', 'kolkata', 'pune', 'delhi', 'hyderabad', 'bangalore', 'chennai', 'india', 'maharashtra', 'gujarat', 'uttar pradesh', 'karnataka', 'andhra pradesh', 'kerala', 'tamil nadu', 'rajasthan', 'bihar', 'west bengal', 'haryana', 'punjab', 'odisha', 'chhattisgarh', 'assam', 'jharkhand', 'uttarakhand', 'himachal pradesh', 'goa', 'tripura', 'meghalaya', 'manipur', 'nagaland', 'arunachal pradesh', 'mizoram', 'sikkim']
    # 1. Look for 'Name :' or 'Name:' patterns (case insensitive)
    for line in lines:
        cl = clean_line(line)
        match = re.match(r'^name\s*[:\-]?\s*([A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){1,2})', cl, re.IGNORECASE)
        if match:
            name = match.group(1)
            # If name is all caps, convert to title case
            if name.isupper():
                return name.title()
            return name.title()
    # 2. Scan top 20 lines
    for line in lines[:20]:
        cl = clean_line(line)
        if is_likely_name(cl, blacklist) and not any(loc in cl.lower() for loc in location_keywords):
            # If line is all caps, convert to title case
            if cl.isupper():
                return cl.title()
            return cl.title()
    # 3. Scan last 20 lines
    for line in lines[-20:]:
        cl = clean_line(line)
        if is_likely_name(cl, blacklist) and not any(loc in cl.lower() for loc in location_keywords):
            # If line is all caps, convert to title case
            if cl.isupper():
                return cl.title()
            return cl.title()
    return None

def find_name_in_personal_section(lines):
    """
    Find a name in the personal details section of a resume, which typically comes after
    a "Personal Details" header and contains lines like "Name: John Doe"
    """
    in_personal_section = False
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Check if this line indicates the start of a personal details section
        if ('personal details' in line_lower or 'personal information' in line_lower or 
            'personal profile' in line_lower or 'contact details' in line_lower):
            in_personal_section = True
            continue
        
        # If in personal section, look for name indicators
        if in_personal_section:
            # Look for explicit name indicators
            if line_lower.startswith('name:') or line_lower.startswith('full name:'):
                name_part = line.split(':', 1)[1].strip()
                if name_part and len(name_part.split()) <= 4:  # Reasonable name length
                    return name_part
                    
            # If we hit another section header, exit
            if is_section_header(line):
                break
    
    return None

def extract_all_caps_name(lines):
    """
    Extract an all-caps name from the first few lines of a resume.
    Handles prefixes like MR., MS., DR., etc.
    """
    # Define common section headers that might appear in uppercase
    section_headers_uppercase = [
        'CONTACT DETAILS', 'PERSONAL DETAILS', 'RESUME', 'CV', 'CURRICULUM VITAE', 
        'PROFILE', 'SUMMARY', 'PERSONAL PROFILE', 'PERSONAL INFORMATION',
        'CAREER OBJECTIVE', 'OBJECTIVE', 'PROFESSIONAL SUMMARY', 'EDUCATION'
    ]
    
    for line in lines[:10]:  # Check just first 10 lines
        line = line.strip()
        
        # Skip if line matches any section header
        if any(header in line for header in section_headers_uppercase):
            continue
            
        # Check if line is all caps and looks like a name
        words = line.split()
        
        # Special handling for names with prefixes like MR., MS., DR.
        if len(words) >= 2 and words[0] in ['MR.', 'MS.', 'DR.', 'MISS', 'MRS.', 'ER.', 'PROF.']:
            # Format name properly (Title case)
            return ' '.join(w.capitalize() for w in words)
            
        # Regular all-caps name detection (allowing names with 1-4 words)
        if (line.isupper() and 1 <= len(words) <= 4 and 
            all(w.isalpha() or w.endswith('.') for w in words) and
            all(len(w) >= 2 for w in words)):  # Require at least 2 chars per word
            # Format name properly (Title case)
            return ' '.join(w.capitalize() for w in words)
    return None

def extract_personal_details(text):
    # Clean non-ASCII characters that may corrupt phone numbers or emails
    def clean_unicode(text):
        return ''.join(c if ord(c) < 128 else '' for c in text)
    text = clean_unicode(text)
    details = {
        'name': None,
        'email': None,
        'phone': None,
        'linkedin': None,
        'github': None,
        'location': None
    }
    lines = [l.strip() for l in text.strip().splitlines()]
    full_text = ' '.join([l for l in lines if l.strip()])

    # --- Extract email and phone FIRST ---
    # Improved: Only extract email from lines that do NOT contain a 10-digit number before the email
    email = None
    for line in lines:
        email_match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', line)
        if email_match:
            before_email = line[:email_match.start()]
            before_email_clean = re.sub(r'[^a-zA-Z0-9]', '', before_email)
            if len(before_email_clean) >= 10 and before_email_clean[-10:].isdigit():
                email = email_match.group()
                break
            elif re.search(r'\b\d{10}\b', before_email):
                continue
            else:
                email = email_match.group()
                break
    if not email:
        # fallback: search in full_text as before
        email_match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', full_text)
        email = email_match.group() if email_match else None
    details['email'] = email
    # Remove leading 10-digit number from the username part of the email
    if email:
        username, domain = email.split('@', 1)
        username = re.sub(r'^\d{10}', '', username)
        email = f"{username}@{domain}"
        details['email'] = email

    # --- Enhanced phone extraction for various formats including international ---
    phone = None
    phone_candidates = []
    incomplete_phone = None
    contact_keywords = ['contact', 'phone', 'mobile', 'tel', 'cell']
    
    # More comprehensive phone regex patterns
    phone_patterns = [
        # Standard 10-digit number
        r'(?<!\d)(\d{10})(?!\d)',
        # Numbers with spaces, dashes, or dots as separators
        r'(\d{3}[\s.-]?\d{3}[\s.-]?\d{4})',
        # International format with country code
        r'\+\d{1,4}\s*\d{5,}',
        # International format with parentheses
        r'\(\+\d{1,4}\)\s*\d{5,}',
        # Specifically match formats like "(+91) 76203 28192"
        r'\(\+\d{1,4}\)\s*\d{5}\s*\d{5}',
        # Numbers with spaces in specific groupings (e.g., 76203 28192)
        r'(?<!\d)(\d{5}\s+\d{5})(?!\d)'
    ]
    
    # First check for phone numbers in context of contact keywords
    for idx, line in enumerate(lines):
        if any(kw in line.lower() for kw in contact_keywords):
            # Apply all phone patterns
            for pattern in phone_patterns:
                found = re.findall(pattern, line)
                for match in found:
                    # Clean the match to get just digits
                    clean_number = re.sub(r'[^0-9]', '', match)
                    phone_candidates.append(clean_number)
            
            # Check next lines too
            for offset in [1, 2]:
                if idx + offset < len(lines):
                    next_line = lines[idx + offset]
                    for pattern in phone_patterns:
                        found_next = re.findall(pattern, next_line)
                        for match in found_next:
                            clean_number = re.sub(r'[^0-9]', '', match)
                            phone_candidates.append(clean_number)
    
    # If no phone found in contact keywords context, scan all lines
    if not phone_candidates:
        for line in lines:
            # Skip lines that look like dates or percentages
            if re.search(r'\d{1,2}\s*[a-zA-Z]{3,9}\s*\d{2,4}', line) or '%' in line:
                continue
            
            # Apply all phone patterns
            for pattern in phone_patterns:
                found = re.findall(pattern, line)
                for match in found:
                    clean_number = re.sub(r'[^0-9]', '', match)
                    phone_candidates.append(clean_number)
    
    # If still nothing, check entire text
    if not phone_candidates:
        for pattern in phone_patterns:
            fallback_found = re.findall(pattern, full_text)
            for match in fallback_found:
                clean_number = re.sub(r'[^0-9]', '', match)
                phone_candidates.append(clean_number)
    
    # Special check for contact info in the first few lines (common resume format)
    if not phone_candidates and len(lines) > 1:
        # Check first 3 lines where contact info often appears
        for line_idx in range(min(3, len(lines))):
            line = lines[line_idx]
            
            # Direct check for the specific format in Shrushti's resume: "(+91) 76203 28192"
            specific_match = re.search(r'\(\+(\d{1,3})\)\s*(\d{5})\s*(\d{5})', line)
            if specific_match:
                country_code = specific_match.group(1)
                first_part = specific_match.group(2)
                second_part = specific_match.group(3)
                digits = first_part + second_part
                phone_candidates.append(digits)
                print(f"DEBUG: Found phone in specific format: country code {country_code}, number {digits}")
                
            # Look for patterns like "email | phone | location" or "email / phone / location"
            if '|' in line or '/' in line:
                segments = re.split(r'[|/]', line)
                for segment in segments:
                    segment = segment.strip()
                    # If segment has digits and is not just a year
                    if re.search(r'\d', segment) and not re.match(r'^\d{4}$', segment.strip()):
                        # Extract all digits
                        digits = re.sub(r'[^0-9]', '', segment)
                        if len(digits) >= 10:
                            phone_candidates.append(digits)
                            print(f"DEBUG: Found phone in contact line: {digits} from segment '{segment}'")
    
    # Process candidates, prioritizing Indian phone numbers (10 digits)
    if phone_candidates:
        # First look for Indian 10-digit numbers
        phone = next((num for num in phone_candidates if len(num) == 10), None)
        
        # If not found, try numbers with country code (usually 91 for India + 10 digits = 12)
        if not phone:
            indian_phone = next((num for num in phone_candidates if len(num) == 12 and num.startswith('91')), None)
            if indian_phone:
                # Remove country code for consistency
                phone = indian_phone[2:]
            else:
                # Try any other number within reasonable length
                phone = next((num for num in phone_candidates if len(num) >= 10 and len(num) <= 15), None)
        
        # Fallback to any number with reasonable length
        if not phone:
            phone = next((num for num in phone_candidates if 8 <= len(num) < 10), None)
    if phone:
        details['phone'] = phone
    elif incomplete_phone:
        details['phone'] = f"{incomplete_phone} (possibly incomplete)"
    else:
        # Try more specific Indian phone number formats
        indian_formats = [
            # Standard 10-digit
            r'(?<!\d)(\d{10})(?!\d)',
            # With country code
            r'(?<!\d)(91\d{10})(?!\d)',
            # With + country code
            r'\+91[\s.-]?(\d{10})',
            # With formatted country code
            r'\(\+91\)[\s.-]?(\d{10})',
            # With spaces after 5 digits (common format)
            r'(\d{5})[\s.-](\d{5})',
            # With spaces after 3 and 6 digits
            r'(\d{3})[\s.-](\d{3})[\s.-](\d{4})'
        ]
        
        for pattern in indian_formats:
            match = re.search(pattern, full_text)
            if match:
                if len(match.groups()) == 1:
                    details['phone'] = match.group(1)
                    break
                elif len(match.groups()) == 2:
                    details['phone'] = match.group(1) + match.group(2)
                    break
                elif len(match.groups()) == 3:
                    details['phone'] = match.group(1) + match.group(2) + match.group(3)
                    break
        
        # Final fallback
        if not details['phone']:
            # Look for any 10+ digit sequence not part of another number
            digits = re.findall(r'(?<!\d)[1-9]\d{9,14}(?!\d)', full_text)
            if digits:
                if len(digits[0]) > 10 and digits[0].startswith('91'):
                    details['phone'] = digits[0][2:] if len(digits[0]) - 2 == 10 else digits[0]
                else:
                    details['phone'] = digits[0]
            else:
                details['phone'] = None

    # --- Name Extraction ---
    name_candidate = None
    
    # Filter out common section headers that might be mistakenly identified as names
    name_blacklist_keywords = [
        # Section headers
        'seminars', 'digital marketing', 'training', 'resume', 'cv', 'curriculum vitae', 
        'contact', 'details', 'personal details', 'profile', 'summary', 'professional',
        'objective', 'career', 'education', 'experience', 'skills', 'projects',
        
        # Education-related terms
        'intermediate', 'college', 'university', 'school', 'institute', 'academy',
        'bachelor', 'master', 'phd', 'b.', 'm.', 'dr.', 'degree', 'diploma', 'board',
        'secondary', 'senior', 'junior', 'high school', 'primary', 'elementary',
        'pursuing', 'expected', 'graduation', 'management', 'technology', 'engineering',
        
        # Document sections
        'certification', 'declaration', 'references', 'hobbies', 'interests',
        'strengths', 'achievements', 'languages', 'technical', 'personal',
        
        # Social media and contact terms
        'linkedin', 'account', 'accomplishments', 'work history', 'phone', 'email',
        'address', 'contact', 'mobile', 'tel', 'www', 'http', '.com', '@',
        'fathers name', 'mothers name', 'parent name', 'guardian name', 'spouse name',
        'date of birth', 'gender', 'nationality', 'marital status', 'language known',
        
        # Skills and accomplishment items that might be mistaken as names
        'cost savings', 'cost reduction', 'operational control', 'inventory management',
        'transportation', 'sourcing', 'procurement', 'customer relationship',
        'time management', 'problem solving', 'analytical skills', 'leadership',
        'communication', 'negotiation', 'ms office'
    ]
    
    # Conditional: Only use robust email-prefix-based extraction for likely concatenated email names
    email_match = re.search(r'[^\s]+@[\w\-]+(?:\.[\w\-]+)+', text)
    use_robust_email_name = False
    if email_match:
        email_prefix = re.sub(r'\d+', '', email_match.group().split('@')[0]).lower()
        if len(email_prefix) > 8 and all(sep not in email_prefix for sep in ['.', '_', '-']):
            use_robust_email_name = True

    # Initialize lines variable first so all extraction methods can use it
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    
    # Priority 1: Look for names in the personal details section
    personal_section_name = find_name_in_personal_section(lines)
    if personal_section_name:
        details['name'] = personal_section_name
        print(f"DEBUG: details['name'] set to: {details['name']} (from personal section)")
    
    # Priority 1.1: Look for names that appear after section headers like "LANGUAGES" and before skills/accomplishments
    if not details.get('name'):
        # Find lines with section headers that often precede names
        section_patterns = [
            r'\b(?:LANGUAGES|ACCOMPLISHMENTS|SKILLS)\b',
            r'\b(?:CONTACT|PERSONAL\s+DETAILS)\b'
        ]
        
        for pattern in section_patterns:
            section_match = re.search(pattern, text, re.IGNORECASE)
            if section_match:
                # Get text after this section
                text_after_section = text[section_match.end():]
                lines_after = text_after_section.strip().split('\n')
                
                # Look in the first few lines after the section
                for line in lines_after[:5]:
                    line = line.strip()
                    if line and len(line.split()) >= 2 and len(line) <= 50:
                        words = line.split()
                        
                        # Check for all caps names (very common in resumes)
                        if all(word.isupper() and word.isalpha() and len(word) > 1 for word in words):
                            line_lower = line.lower()
                            if not any(blacklist_term in line_lower for blacklist_term in name_blacklist_keywords):
                                if not re.search(r'\b(phone|email|linkedin|www|http|\.com|@)\b', line_lower):
                                    details['name'] = line.strip()
                                    print(f"DEBUG: details['name'] set to: {details['name']} (from after {pattern} section - all caps)")
                                    break
                if details.get('name'):
                    break
    
    # Priority 1.2: Look for standalone name lines that appear between major sections
    if not details.get('name'):
        # Find lines that are likely to be standalone names (2-4 words, all alphabetic, properly capitalized)
        for i, line in enumerate(lines):
            line = line.strip()
            words = line.split()
            
            # Skip if it's empty or too long
            if not line or len(words) > 4 or len(words) < 2:
                continue
                
            # Check if all words are alphabetic and properly capitalized
            if all(w.isalpha() and w[0].isupper() for w in words):
                # Additional checks to ensure it's likely a name
                line_lower = line.lower()
                
                # Skip if it contains education-related terms
                if any(term in line_lower for term in ['pursuing', 'expected', 'university', 'college', 'institute', 'management', 'technology']):
                    continue
                    
                # Skip if it's a section header or common resume terms
                if is_section_header_name(line_lower):
                    continue
                    
                # Use comprehensive blacklist to filter out non-name terms
                if any(blacklist_term in line_lower for blacklist_term in name_blacklist_keywords):
                    continue
                    
                # Skip if it's right after certain section headers (within 3 lines)
                is_after_section = False
                for j in range(max(0, i-3), i):
                    prev_line = lines[j].strip().lower()
                    if any(section in prev_line for section in ['education', 'work history', 'experience', 'skills', 'contact']):
                        is_after_section = True
                        break
                
                # If it's not immediately after a section header, it's likely a name
                if not is_after_section:
                    details['name'] = line
                    print(f"DEBUG: details['name'] set to: {details['name']} (from standalone name line)")
                    break
    
    # Priority 1.3: Check for name patterns right before "SKILLS" section (added for LinkedIn issue)
    if not details.get('name'):
        skills_pattern = r'\b(?:SKILLS|TECHNICAL\s+SKILLS|CORE\s+SKILLS)\b'
        skills_match = re.search(skills_pattern, text, re.IGNORECASE)
        if skills_match:
            text_before_skills = text[:skills_match.start()]
            lines_before_skills = text_before_skills.strip().split('\n')
            
            # Look in the last few lines before SKILLS section
            for line in reversed(lines_before_skills[-10:]):  # Increased range to 10 lines
                line = line.strip()
                if line and len(line.split()) >= 2 and len(line) <= 50:
                    # Check if this looks like a name (multiple words, proper capitalization)
                    words = line.split()
                    
                    # Special check for all caps names (common in resumes)
                    if all(word.isupper() and word.isalpha() and len(word) > 1 for word in words):
                        line_lower = line.lower()
                        if not any(blacklist_term in line_lower for blacklist_term in name_blacklist_keywords):
                            if not re.search(r'\b(district|pin|code|email|phone|mobile|linkedin|account)\b', line_lower):
                                details['name'] = line.strip()
                                print(f"DEBUG: details['name'] set to: {details['name']} (from before SKILLS section - all caps)")
                                break
                    
                    # Check for normal capitalized names
                    elif all(word[0].isupper() and len(word) > 1 for word in words if word.isalpha()):
                        line_lower = line.lower()
                        if not any(blacklist_term in line_lower for blacklist_term in name_blacklist_keywords):
                            if not re.search(r'\b(district|pin|code|email|phone|mobile|linkedin|account)\b', line_lower):
                                details['name'] = line.strip()
                                print(f"DEBUG: details['name'] set to: {details['name']} (from before SKILLS section)")
                                break
    
    # Priority 1.4: Look for names in PERSONAL DETAILS section (common in Indian resumes)
    if not details.get('name'):
        personal_details_idx = None
        for i, line in enumerate(lines):
            if re.search(r'\b(personal\s+details|personal\s+information)\b', line, re.IGNORECASE):
                personal_details_idx = i
                break
        
        if personal_details_idx is not None:
            # Look in the next 15 lines after PERSONAL DETAILS
            for i in range(personal_details_idx + 1, min(len(lines), personal_details_idx + 16)):
                line = lines[i].strip()
                
                # Look for "NAME : actual_name" pattern
                name_match = re.search(r'name\s*:?\s*([A-Za-z\s\.]+)', line, re.IGNORECASE)
                if name_match:
                    name_part = name_match.group(1).strip()
                    # Validate it's not a blacklisted term
                    if not any(blacklist_term in name_part.lower() for blacklist_term in name_blacklist_keywords):
                        details['name'] = name_part
                        print(f"DEBUG: details['name'] set to: {details['name']} (from PERSONAL DETAILS section)")
                        break
                
                # Also check if line after "NAME" label contains the actual name
                if re.match(r'^\s*name\s*:?\s*$', line, re.IGNORECASE) and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and len(next_line.split()) <= 4:
                        words = next_line.split()
                        if all(word.isalpha() for word in words):
                            if not any(blacklist_term in next_line.lower() for blacklist_term in name_blacklist_keywords):
                                details['name'] = next_line
                                print(f"DEBUG: details['name'] set to: {details['name']} (from line after NAME label)")
                                break
    
    # Priority 1.5: Look for names that appear after education section and before contact section
    if not details.get('name'):
        education_idx = None
        contact_idx = None
        
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            if line_lower in ['education', 'educational details', 'academic qualification']:
                education_idx = i
            elif line_lower in ['contact', 'contact details', 'contact information', 'personal details']:
                contact_idx = i
                break
        
        # Look for name candidates between education and contact sections
        if education_idx is not None and contact_idx is not None:
            name_parts = []
            for i in range(education_idx + 1, contact_idx):
                line = lines[i].strip()
                # Skip empty lines and lines with numbers/years
                if not line or re.search(r'\d{4}', line) or re.search(r'\d+\.\d+', line):
                    continue
                # Skip education-related terms
                if any(term in line.lower() for term in ['university', 'college', 'bachelor', 'pharmacy', 'deemed', 'pursuing', 'expected']):
                    continue
                # Look for lines that look like names (1-4 words, all alphabetic)
                words = line.split()
                if 1 <= len(words) <= 4 and all(w.isalpha() for w in words):
                    # Check if it's likely a person's name (starts with capital letters)
                    if all(w[0].isupper() for w in words):
                        name_parts.extend(words)
                        # If we have collected enough name parts, break
                        if len(name_parts) >= 2:
                            break
            
            # Combine all name parts found
            if name_parts:
                full_name = ' '.join(w.capitalize() for w in name_parts)
                details['name'] = full_name
                print(f"DEBUG: details['name'] set to: {details['name']} (from education-contact section, combined parts)")
                # Return early to prevent other methods from overwriting this good result
                # --- Combined spaCy and keyword-based location extraction ---
                def extract_location(full_text, major_locations):
                    text_lower = full_text.lower()
                    
                    # 1. Prioritize Major Cities
                    first_major_city = None
                    first_major_pos = float('inf')

                    for city in major_locations:
                        try:
                            pos = re.search(r'\b' + re.escape(city.lower()) + r'\b', text_lower)
                            if pos and pos.start() < first_major_pos:
                                first_major_pos = pos.start()
                                first_major_city = city
                        except re.error:
                            continue
                        
                    if first_major_city:
                        return first_major_city

                    # 2. Fallback to smaller locations (GPEs from spaCy) if no major city is found
                    doc = nlp(full_text)
                    all_gpes = list(OrderedDict.fromkeys([ent.text.strip() for ent in doc.ents if ent.label_ == 'GPE']))
                    
                    # Filter out noise
                    noise = {'objective', 'summary', 'skills', 'education', 'experience', 'project', 'india'}
                    all_gpes = [loc for loc in all_gpes if loc.lower() not in noise and len(loc) > 2]

                    if not all_gpes:
                        return None

                    first_gpe = None
                    first_gpe_pos = float('inf')

                    for gpe in all_gpes:
                        try:
                            pos = re.search(r'\b' + re.escape(gpe.lower()) + r'\b', text_lower)
                            if pos and pos.start() < first_gpe_pos:
                                first_gpe_pos = pos.start()
                                first_gpe = gpe
                        except re.error:
                            continue
                        
                    return first_gpe
                
                details['location'] = extract_location(full_text, major_indian_locations)
                return details
    
    # Priority 2: Look for name patterns like "Name: John Doe" or lines with just a name and signature
    if not details.get('name'):
        for line in lines:
            line = line.strip()
            # Check for name near signature
            signature_match = re.search(r'signature:?\s*([A-Za-z\s\.]+)$', line, re.IGNORECASE)
            if signature_match:
                name_part = signature_match.group(1).strip()
                if len(name_part.split()) <= 4:  # Reasonable name length
                    details['name'] = ' '.join(w.capitalize() for w in name_part.split())
                    print(f"DEBUG: details['name'] set to: {details['name']} (from signature line)")
                    break
                    
            # Check for "Name:" pattern - enhanced to handle various formats
            name_label_patterns = [
                r'^name\s*:?\s*([A-Za-z\s\.]+)$',
                r'name\s*:\s*([A-Za-z\s\.]+)',
                r'^\s*:\s*([A-Za-z\s\.]+)$'  # For cases where "NAME" is on previous line
            ]
            
            for pattern in name_label_patterns:
                name_label_match = re.search(pattern, line, re.IGNORECASE)
                if name_label_match:
                    name_part = name_label_match.group(1).strip()
                    # Additional validation - should not be a section header
                    if not any(blacklist_term in name_part.lower() for blacklist_term in name_blacklist_keywords):
                        details['name'] = name_part
                        print(f"DEBUG: details['name'] set to: {details['name']} (from name label pattern)")
                        break
            
            if details.get('name'):
                break
                
            # Look for standalone names with initial patterns (like "R. Ramesh")
            initial_pattern_match = re.match(r'^([A-Z]\.\s+[A-Z][a-z]+)$', line) 
            if initial_pattern_match and len(line) < 30:
                details['name'] = line
                print(f"DEBUG: details['name'] set to: {details['name']} (from initial pattern)")
                break
    
    # Priority 3: Try to extract an all-caps name, as this is a common format in resumes and usually reliable
    if not details.get('name'):
        all_caps_name = extract_all_caps_name(lines)
        if all_caps_name:
            details['name'] = all_caps_name
            print(f"DEBUG: details['name'] set to: {details['name']} (from all_caps_name extraction)")
    
    # Second priority: Look for names with prefixes like "MR." in the first line
    if lines and not details.get('name'):
        first_line = lines[0].strip()
        # Check first line for prefixed names like "MR. JOHN DOE"
        prefix_match = re.match(r'^(MR\.|MS\.|DR\.|MISS|MRS\.|ER\.|PROF\.)\s+([A-Z\s\.]+)$', first_line, re.IGNORECASE)
        if prefix_match:
            prefix = prefix_match.group(1).capitalize()
            name_part = ' '.join(w.capitalize() for w in prefix_match.group(2).split())
            details['name'] = f"{prefix} {name_part}"
            print(f"DEBUG: details['name'] set to: {details['name']} (from prefix name extraction)")
    
    if details.get('name') and any(keyword in details['name'].lower() for keyword in name_blacklist_keywords):
        print(f"DEBUG: Potential section header detected as name: '{details['name']}'. Clearing and trying again.")
        details['name'] = None

    # Continue with other extraction methods if needed
    if use_robust_email_name and not details.get('name'):
        try:
            actual_name = search_for_actual_name_in_text(text)
            print(f"DEBUG: search_for_actual_name_in_text returned: {actual_name}")
            if actual_name:
                details['name'] = actual_name
                print(f"DEBUG: details['name'] set to: {details['name']} (from robust email logic)")
                # --- Location Extraction (always run before return) ---
                details['location'] = extract_location(full_text, major_indian_locations)
                return details  # Return early to prevent overwriting
        except Exception as e:
            print(f"DEBUG: search_for_actual_name_in_text failed: {e}")
    # First line priority: if first line looks like a name, use it (only if robust logic not used)
    if lines and not details.get('name'):
        first_line = lines[0]
        # Normalize first line by removing zero-width spaces and extra spaces
        normalized_first_line = re.sub(r'[\u200b\s]+', ' ', first_line).strip()
        normalized_first_line_lower = normalized_first_line.lower()
        words = normalized_first_line.split()
        section_headers_clean = set(h.strip().lower() for h in section_headers)
        # Additional check: exclude 'professional summary' explicitly if present
        if 'professional summary' in normalized_first_line_lower:
            print(f"DEBUG: Skipping first line as it is a professional summary header")
        elif (2 <= len(words) <= 4 and all(w.isalpha() for w in words) and
            normalized_first_line_lower not in section_headers_clean):
            # Special handling for names - apply proper capitalization
            if first_line.isupper():
                # Convert ALL CAPS to Title Case
                details['name'] = ' '.join(w.capitalize() for w in words)
                print(f"DEBUG: details['name'] set to: {details['name']} (from first line priority - converted ALL CAPS)")
            elif first_line.islower():
                # Convert lowercase to Title Case
                details['name'] = ' '.join(w.capitalize() for w in words)
                print(f"DEBUG: details['name'] set to: {details['name']} (from first line priority - converted lowercase)")
            else:
                # Keep original if it has mixed case (likely already formatted correctly)
                details['name'] = first_line
                print(f"DEBUG: details['name'] set to: {details['name']} (from first line priority)")
    # --- NEW: Try to extract name from first two lines if both look like names and no name found yet ---
    if not details.get('name'):
        non_empty_lines = [l for l in lines if l.strip()]
        if len(non_empty_lines) >= 2:
            first, second = non_empty_lines[0], non_empty_lines[1]
            section_headers_clean = set(h.strip().lower() for h in section_headers)
            first_clean = first.strip().lower()
            second_clean = second.strip().lower()
            # If first line is a single word and second is in blacklist, use only first
            if (first.isalpha() and 2 <= len(first) <= 20 and ' ' not in first and second_clean in section_headers_clean):
                # Handle all-caps first name
                if first.isupper():
                    details['name'] = first.capitalize()
                else:
                    details['name'] = first.title()
                print(f"DEBUG: details['name'] set to: {details['name']} (from single-word first line, second in blacklist)")
            elif (first.isalpha() and 2 <= len(first) <= 20 and first_clean not in section_headers_clean):
                if (second.isalpha() and 2 <= len(second) <= 20 and second_clean not in section_headers_clean):
                    # Handle all-caps name components
                    first_formatted = first.capitalize() if first.isupper() else first.title()
                    second_formatted = second.capitalize() if second.isupper() else second.title()
                    details['name'] = f'{first_formatted} {second_formatted}'
                    print(f"DEBUG: details['name'] set to: {details['name']} (from first two lines heuristic)")
                else:
                    # Handle all-caps first name
                    if first.isupper():
                        details['name'] = first.capitalize()
                    else:
                        details['name'] = first.title()
                    print(f"DEBUG: details['name'] set to: {details['name']} (from first line only, second line in blacklist)")
    # ... rest of the existing code ...

    # Special check for the exact case of "R. Ramesh" or similar names with initials
    if not details.get('name'):
        for line in lines:
            line = line.strip()
            # Match precisely patterns like "R. Ramesh" which have an initial followed by a name
            if re.match(r'^([A-Z]\.\s+[A-Z][a-z]+)$', line) or re.match(r'^([A-Z]\.)\s+([A-Z][a-z]+)$', line):
                details['name'] = line
                print(f"DEBUG: details['name'] set to: {details['name']} (from precise initial match)")
                break
    
    # Look for name near phone/email contact info
    if not details.get('name'):
        # Find lines with phone or email
        contact_lines = []
        for i, line in enumerate(lines):
            if details.get('phone') and details['phone'] in line:
                # Get surrounding lines for context
                start = max(0, i-2)
                end = min(len(lines), i+3)
                contact_lines.extend(lines[start:end])
            if details.get('email') and details['email'] in line:
                # Get surrounding lines for context
                start = max(0, i-2)
                end = min(len(lines), i+3)
                contact_lines.extend(lines[start:end])
        
        # Check these contact context lines for name patterns
        for line in contact_lines:
            line = line.strip()
            # Look for standalone short line that could be a name
            if (2 <= len(line.split()) <= 4 and 
                all(len(w) >= 2 and w[0].isupper() for w in line.split()) and
                not any(kw.lower() in line.lower() for kw in name_blacklist_keywords)):
                details['name'] = line
                print(f"DEBUG: details['name'] set to: {details['name']} (from contact context)")
                break
                
            # Look for standalone names that are 1-3 words with at least one capital letter
            words = line.split()
            if (1 <= len(words) <= 3 and 
                any(w[0].isupper() for w in words if w) and
                not any(kw.lower() in line.lower() for kw in name_blacklist_keywords) and
                len(line) <= 30):  # Not too long
                details['name'] = ' '.join(w[0].upper() + w[1:].lower() if w else '' for w in words)
                print(f"DEBUG: details['name'] set to: {details['name']} (from standalone name pattern)")
                break
    
    # If name_candidate is a single word, try to find a better multi-word name near contact info
    if name_candidate and len(name_candidate.split()) == 1:
        # Find indices of phone/email lines
        phone_idx = email_idx = None
        for idx, line in enumerate(lines):
            if details.get('phone') and details['phone'] in line:
                phone_idx = idx
            if details.get('email') and details['email'] in line:
                email_idx = idx
        candidate_indices = []
        if phone_idx is not None:
            candidate_indices.extend(range(max(0, phone_idx-5), min(len(lines), phone_idx+6)))
        if email_idx is not None:
            candidate_indices.extend(range(max(0, email_idx-5), min(len(lines), email_idx+6)))
        candidate_indices = sorted(set(candidate_indices))
        for idx in candidate_indices:
            line = lines[idx].strip()
            words = line.split()
            if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
                name_candidate = line.title()
                break
    # LinkedIn and GitHub extraction (improved)
    linkedin_match = re.findall(r'(?:https?://)?(?:www\.)?linkedin\.com/(?:in|pub)/([\w\d\-_/]+)', full_text, re.I)
    if linkedin_match:
        # Use the last part after the last '/'
        handle = linkedin_match[0].split('/')[-1]
        details['linkedin'] = f'linkedin.com/in/{handle}'
    else:
        # Try to find any linkedin.com/in/ handle in the text
        generic_linkedin = re.findall(r'linkedin\.com/([\w\d\-_/]+)', full_text, re.I)
        if generic_linkedin:
            handle = generic_linkedin[0].split('/')[-1]
            details['linkedin'] = f'linkedin.com/in/{handle}'
        else:
            details['linkedin'] = None
    print(f"DEBUG: LinkedIn handle for name extraction: {details['linkedin']}")
    github_match = re.findall(r'(?:https?://)?(?:www\.)?github\.com/[\w\d\-_/]+/?', full_text, re.I)
    details['github'] = github_match[0].rstrip('/') if github_match else None

    # --- Prioritized Indian address location extraction ---
    address_keywords = ['at/po-', 'ps-', 'district', 'pin-', 'city', 'village', 'taluka', 'mandal', 'tehsil']
    found_location = None
    for line in lines:
        line_lower = line.lower()
        # Check for address keywords
        if any(kw in line_lower for kw in address_keywords):
            match = re.search(r'at/po-\s*([a-zA-Z ]+)', line_lower)
            if match:
                found_location = match.group(1).strip().title()
            match = re.search(r'district\s*([a-zA-Z ]+)', line_lower)
            if match:
                found_location = match.group(1).strip().title()
            match = re.search(r'([a-zA-Z ]+),?\s*pin-', line_lower)
            if match:
                found_location = match.group(1).strip().title()
        # Check for known major cities/districts
        for city in major_indian_locations:
            if city.lower() in line_lower:
                found_location = city
    if found_location:
        details['location'] = found_location
    else:
        # --- Combined spaCy and keyword-based location extraction ---
        def extract_location_fallback(full_text, major_locations):
            text_lower = full_text.lower()
            
            # 1. Prioritize Major Cities
            first_major_city = None
            first_major_pos = float('inf')

            for city in major_locations:
                try:
                    pos = re.search(r'\b' + re.escape(city.lower()) + r'\b', text_lower)
                    if pos and pos.start() < first_major_pos:
                        first_major_pos = pos.start()
                        first_major_city = city
                except re.error:
                    continue
                
            if first_major_city:
                return first_major_city

            # 2. Fallback to smaller locations (GPEs from spaCy) if no major city is found
            doc = nlp(full_text)
            all_gpes = list(OrderedDict.fromkeys([ent.text.strip() for ent in doc.ents if ent.label_ == 'GPE']))
            
            # Filter out noise
            noise = {'objective', 'summary', 'skills', 'education', 'experience', 'project', 'india'}
            all_gpes = [loc for loc in all_gpes if loc.lower() not in noise and len(loc) > 2]

            if not all_gpes:
                return None

            first_gpe = None
            first_gpe_pos = float('inf')

            for gpe in all_gpes:
                try:
                    pos = re.search(r'\b' + re.escape(gpe.lower()) + r'\b', text_lower)
                    if pos and pos.start() < first_gpe_pos:
                        first_gpe_pos = pos.start()
                        first_gpe = gpe
                except re.error:
                    continue
                
            return first_gpe

        details['location'] = extract_location_fallback(full_text, major_indian_locations)

    # Debug: Print what name_candidate was found
    print(f"DEBUG: name_candidate found: {name_candidate}")

    def is_date_line(line):
        if not line:
            return False
        # Common date patterns: 'December 2024', '2024', '17 May 2024', 'May 2024', etc.
        date_patterns = [
            r'^(january|february|march|april|may|june|july|august|september|october|november|december) \d{4}$',
            r'^\d{4}$',
            r'^\d{1,2} (january|february|march|april|may|june|july|august|september|october|november|december) \d{4}$',
            r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec) \d{4}$',
            r'^\d{1,2} (jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec) \d{4}$',
            r'^(january|february|march|april|may|june|july|august|september|october|november|december)$',
        ]
        line_clean = line.strip().lower()
        for pat in date_patterns:
            if re.match(pat, line_clean):
                return True
        return False

    def split_email_prefix_to_name(email_prefix, lines, raw_text):
        email_prefix_clean = re.sub(r'\d+', '', email_prefix)
        # Try common separators
        for sep in ['.', '_', '-']:
            if sep in email_prefix_clean:
                parts = [p for p in email_prefix_clean.split(sep) if p.isalpha() and len(p) > 1]
                if 1 <= len(parts) <= 4:
                    candidate = ' '.join(part.capitalize() for part in parts)
                    pattern = re.compile(r'\b' + re.escape(candidate) + r'\b', re.IGNORECASE)
                    match = pattern.search(' '.join(raw_text.split()))
                    if match:
                        return match.group(0)
                    return candidate
        # If no separator, try all possible 2-way and 3-way splits and check all orders in raw text
        if email_prefix_clean.isalpha() and len(email_prefix_clean) > 5:
            best_candidate = None
            n = len(email_prefix_clean)
            # Try all 2-way splits (as before)
            for i in range(2, n-2):
                first, last = email_prefix_clean[:i], email_prefix_clean[i:]
                if len(first) > 1 and len(last) > 1:
                    for candidate in [f"{first.capitalize()} {last.capitalize()}", f"{last.capitalize()} {first.capitalize()}"]:
                        pattern = re.compile(r'\b' + re.escape(candidate) + r'\b', re.IGNORECASE)
                        match = pattern.search(' '.join(raw_text.split()))
                        if match:
                            return match.group(0)
                    if not best_candidate:
                        best_candidate = f"{first.capitalize()} {last.capitalize()}"
            # Try all 3-way splits
            for i in range(2, n-4):
                for j in range(i+1, n-2):
                    first, middle, last = email_prefix_clean[:i], email_prefix_clean[i:j], email_prefix_clean[j:]
                    if all(len(x) > 1 for x in [first, middle, last]):
                        # Try all 6 possible orders
                        orders = [
                            f"{first.capitalize()} {middle.capitalize()} {last.capitalize()}",
                            f"{first.capitalize()} {last.capitalize()} {middle.capitalize()}",
                            f"{middle.capitalize()} {first.capitalize()} {last.capitalize()}",
                            f"{middle.capitalize()} {last.capitalize()} {first.capitalize()}",
                            f"{last.capitalize()} {first.capitalize()} {middle.capitalize()}",
                            f"{last.capitalize()} {middle.capitalize()} {first.capitalize()}"
                        ]
                        for candidate in orders:
                            pattern = re.compile(r'\b' + re.escape(candidate) + r'\b', re.IGNORECASE)
                            match = pattern.search(' '.join(raw_text.split()))
                            if match:
                                return match.group(0)
                        if not best_candidate:
                            best_candidate = f"{first.capitalize()} {middle.capitalize()} {last.capitalize()}"
            if best_candidate:
                # ✅ Check if it actually appears in raw_text
                pattern = re.compile(r'\b' + re.escape(best_candidate) + r'\b', re.IGNORECASE)
                if pattern.search(' '.join(raw_text.split())):
                    return best_candidate
                else:
                    return None  # Don't return wrong guess if not in original resume
        # Fallback: just capitalize
        candidate = email_prefix_clean.capitalize()
        pattern = re.compile(r'\b' + re.escape(candidate) + r'\b', re.IGNORECASE)
        match = pattern.search(' '.join(raw_text.split()))
        if match:
            return match.group(0)
        return candidate

    def is_non_person_name(candidate, blacklist):
        if not candidate:
            return True
        candidate_clean = candidate.strip().lower()
        candidate_clean = re.sub(r'[^a-z ]', '', candidate_clean)
        candidate_clean = ' '.join(candidate_clean.split())
        return candidate_clean in section_header_names or candidate_clean in blacklist or candidate_clean in institution_keywords
    if (not name_candidate or is_non_person_name(name_candidate, blacklist)) and details['email']:
        email_prefix = details['email'].split('@')[0]
        possible_name = split_email_prefix_to_name(email_prefix, lines, text)
        if possible_name and not is_non_person_name(possible_name, blacklist):
            name_candidate = possible_name
        else:
            # Fallback: try to extract only the first name from raw data
            first_name = extract_first_name_from_raw(email_prefix, text)
            if first_name and not is_non_person_name(first_name, blacklist):
                name_candidate = first_name

    # If name_candidate is from email prefix and is a single word, try to match concatenated words in lines
    if name_candidate and details['email']:
        email_prefix = details['email'].split('@')[0].lower()
        if len(email_prefix) > 4 and len(name_candidate.split()) == 1:
            from difflib import SequenceMatcher
            for line in lines:
                line_words = [w.lower() for w in line.split()]
                if 2 <= len(line_words) <= 4 and all(w.isalpha() for w in line_words):
                    concat = ''.join(line_words)
                    # Substring check
                    if email_prefix in concat:
                        name_candidate = line.title()
                        break
                    # Fuzzy match: similarity > 0.7
                    if SequenceMatcher(None, email_prefix, concat).ratio() > 0.7:
                        name_candidate = line.title()
                        break
                    # Start/end match (Indian-style email prefix)
                    first, *middle, last = line_words
                    if email_prefix.startswith(first) and email_prefix.endswith(last):
                        name_candidate = line.title()
                        break

    # If name_candidate is from email prefix, try to find a matching line in the raw text
    if name_candidate and details['email']:
        email_prefix = details['email'].split('@')[0]
        # Split prefix by common separators and camel case
        parts = re.split(r'[._-]', email_prefix)
        if len(parts) == 1:
            # Try to split camel case
            parts = re.findall(r'[A-Z][a-z]*|[a-z]+', email_prefix)
        parts = [p.lower() for p in parts if len(p) > 1]
        # Scan lines for a line containing all parts as separate wordsSS
        for line in lines:
            line_words = [w.lower() for w in line.split()]
            if all(part in line_words for part in parts) and 2 <= len(line_words) <= 4 and all(w.isalpha() for w in line_words):
                name_candidate = line.title()
                break

    # name_candidate should take priority over LinkedIn/email extraction
    if name_candidate and not is_section_header_name(name_candidate) and not details.get('name'):
        details['name'] = name_candidate
        print(f"DEBUG: details['name'] set to: {details['name']} (from name_candidate)")

    if not details['name']:
        # Try LinkedIn handle (improved: always use last part after last /, allow 1-4 alpha parts)
        if details['linkedin']:
            handle = details['linkedin'].split('/')[-1]
            handle_clean = re.sub(r'\d+', '', handle)
            parts = re.split(r'[-_]', handle_clean)
            name_parts = [p.capitalize() for p in parts if p.isalpha() and len(p) > 1]
            possible_name = ' '.join(name_parts)
            if 1 <= len(name_parts) <= 4 and not is_section_header_name(possible_name):
                details['name'] = possible_name
                print(f"DEBUG: Name from LinkedIn: {details['name']}")
            elif len(name_parts) == 1:
                # Try to split single-word handle
                split_name = split_linkedin_handle(name_parts[0])
                if split_name and not is_section_header_name(split_name):
                    details['name'] = split_name
                    print(f"DEBUG: Name from LinkedIn split: {details['name']}")
        # Try email prefix (relaxed: allow 1-4 alphabetic parts)
        if not details['name'] and details['email']:
            email_prefix = details['email'].split('@')[0]
            possible_name = split_email_prefix_to_name(email_prefix, lines, text)
            if possible_name and not is_section_header_name(possible_name):
                name_candidate = possible_name
            else:
                # Fallback: try to extract only the first name from raw data
                first_name = extract_first_name_from_raw(email_prefix, text)
                if first_name and not is_section_header_name(first_name):
                    name_candidate = first_name

    # robust_name_end fallback
    if not details['name']:
        robust_name_end = robust_extract_name(lines[-20:])
        if robust_name_end and not is_section_header_name(robust_name_end):
            details['name'] = robust_name_end

    if not details['name']:
        profile_name = extract_name_from_profile_section(lines)
        if profile_name and not is_section_header_name(profile_name):
            details['name'] = profile_name

    if not details['name']:
        local_blacklist = set([
            'resume', 'curriculum vitae', 'cv', 'profile', 'summary', 'career objective', 'contact', 'email', 'mobile', 'phone',
            'address', 'education', 'academic', 'skills', 'experience', 'projects', 'certifications', 'languages', 'interests',
            'project', 'predictor', 'app', 'system', 'generator', 'website', 'application', 'report', 'analysis', 'certificate',
            'internship', 'achievement', 'interest', 'objective', 'training', 'course', 'workshop', 'seminar', 'conference',
            'award', 'publication', 'activity', 'volunteering', 'team', 'leadership', 'hobby', 'reference'
        ])
        # 1. Try after personal section
        section_name = extract_name_after_personal_section(lines, local_blacklist)
        if section_name and not is_section_header_name(section_name) and not is_date_line(section_name):
            details['name'] = section_name
        # 2. Universal fallback with location filtering
        if not details['name']:
            universal_name = universal_extract_name(lines, local_blacklist)
            if universal_name and not is_section_header_name(universal_name) and not is_date_line(universal_name):
                details['name'] = universal_name

    # Place this fallback at the very end, after all other extraction logic
    if not details['name']:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        # Search from bottom up for two-line names
        if len(lines) >= 2:
            last_line = lines[-1]
            second_last_line = lines[-2]
            if (is_valid_name_line(second_last_line + " " + last_line) and
                not is_section_header_name(second_last_line) and
                not is_section_header_name(last_line)):
                details['name'] = f"{second_last_line.title()} {last_line.title()}"

    if not details['name']:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        # Search from bottom up
        for line in reversed(lines):
            if is_valid_name_line(line):
                line_lower = line.strip().lower()
                # Skip lines containing misleading labels
                if any(keyword in line_lower for keyword in ['email', 'mobile', 'phone', 'address', 'date of birth', 'contact']):
                    continue
                # Skip known section headers
                if is_section_header_name(line):
                    continue
                # Skip all-uppercase blocks
                if line.isupper() and len(line) > 20:
                    continue
                # Safe final check: looks like a name
                words = line.split()
                if 2 <= len(words) <= 3 and all(w.isalpha() for w in words):
                    details['name'] = ' '.join(w.title() for w in words)
                    break

    # Final fallback: if name_candidate is a single word, scan for a better multi-word name
    if name_candidate and len(name_candidate.split()) == 1 and not details['name']:
        for line in lines:
            words = line.split()
            if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
                name_candidate = line.title()
                break

    # Set the name in details if not already set, but do additional checks
    if name_candidate and not details['name']:
        # Ensure name_candidate is not a section header or contains common section keywords
        section_keywords = ['summary', 'experience', 'education', 'skills', 'certificates', 'seminars', 
                           'workshops', 'training', 'digital', 'marketing', 'contact', 'hobby', 'languages']
        is_section = any(keyword.lower() in name_candidate.lower() for keyword in section_keywords)
        
        if not is_section:
            details['name'] = name_candidate
        else:
            print(f"DEBUG: Rejected name candidate '{name_candidate}' as it contains section keywords")
            
            # If we rejected a name candidate, try to get name from first line with special handling
            if lines and not details.get('name'):
                first_line = lines[0].strip()
                if first_line.isupper() and 2 <= len(first_line.split()) <= 4:
                    # Format properly if it's all uppercase
                    details['name'] = ' '.join(w.capitalize() for w in first_line.split())
                    print(f"DEBUG: Fallback to first line for name: {details['name']}")
            

    # Ensure location is a string for output
    if isinstance(details.get("location"), dict):
        loc = details["location"]
        # Join non-empty, non-duplicate values
        loc_str = ", ".join(
            [v for k, v in loc.items() if v and v.lower() not in [vv.lower() for kk, vv in loc.items() if kk != k]]
        )
        details["location"] = loc_str

    # Robust fallback: scan last 20 lines for a likely name if none found
    if not details.get('name'):
        for line in reversed(lines[-20:]):
            words = line.split()
            line_clean = ' '.join(line.lower().split())
            if (2 <= len(words) <= 4 and all(w.isalpha() for w in words)
                and line_clean not in section_headers):
                details['name'] = line.strip()
                print(f"DEBUG: details['name'] set to: {details['name']} (from end-of-resume fallback)")
                break

    # Robust fallback: scan last 30 lines for a likely name if none found
    name_blacklist = set(h.strip().lower() for h in section_headers)
    name_blacklist.update([
        'mobile', 'email', 'phone', 'contact', 'date', 'place', 'signature', 'major', 'minor',
        'languages known', 'local address', 'permanent address', 'address', 'linkedin', 'github'
    ])
    if not details.get('name'):
        for line in reversed(lines[-30:]):
            candidate = line.strip()
            candidate_clean = ' '.join(candidate.lower().split())
            words = candidate.split()
            if (2 <= len(words) <= 4 and all(w.isalpha() for w in words)
                and candidate_clean not in name_blacklist):
                details['name'] = candidate
                print(f"DEBUG: details['name'] set to: {details['name']} (from robust end-of-resume fallback)")
                break

       # Robust fallback: scan last 40 lines for a likely name if none found
    name_blacklist = set(h.strip().lower() for h in section_headers)
    name_blacklist.update([
        'mobile', 'email', 'phone', 'contact', 'date', 'place', 'signature', 'major', 'minor',
        'languages known', 'local address', 'permanent address', 'address', 'linkedin', 'github'
    ])
    if not details.get('name'):
        for line in reversed(lines[-40:]):
            candidate = line.strip()
            candidate_clean = ' '.join(candidate.lower().split())
            words = candidate.split()
            print(f"DEBUG: Fallback considering: '{candidate_clean}'")
            if candidate_clean in name_blacklist:
                print(f"DEBUG: Skipped (blacklist): '{candidate_clean}'")
                continue
            if (2 <= len(words) <= 4 and all(w.isalpha() for w in words)):
                details['name'] = candidate
                print(f"DEBUG: details['name'] set to: {details['name']} (from robust end-of-resume fallback)")
                break

    # Ultra-robust fallback: look near 'Mobile'/'E-mail' at the end
    if not details.get('name'):
        for i in range(len(lines)-1, -1, -1):
            line = lines[i]
            if re.search(r"Mobile", line, re.I) or re.search(r"E-?mail", line, re.I):
                # Look a few lines above for the name
                for j in range(i-1, max(i-5, -1), -1):
                    possible_name = lines[j].strip()
                    # Heuristic: not all uppercase, not a heading, no digits, 2-4 words, not empty
                    if (possible_name and
                        not possible_name.isupper() and
                        not re.search(r"\d", possible_name) and
                        2 <= len(possible_name.split()) <= 4 and
                        not any(word in possible_name.lower() for word in ["address", "date", "place", "declaration", "details"])):
                        details['name'] = possible_name
                        print(f"DEBUG: details['name'] set to: {details['name']} (from ultra robust mobile/email end fallback)")
                        break
                if details.get('name'):
                    break

        return details

    # Ultra-robust fallback: scan the entire resume from bottom to top for a likely name if none found
    def clean_candidate(line):
        # Remove non-printable, non-ASCII, and normalize spaces
        line = ''.join(c for c in line if c.isprintable() and ord(c) < 128)
        return ' '.join(line.strip().split())

    if not details.get('name'):
        name_blacklist = set(h.strip().lower() for h in section_headers)
        name_blacklist.update([
            'mobile', 'email', 'phone', 'contact', 'date', 'place', 'signature', 'major', 'minor',
            'languages known', 'local address', 'permanent address', 'address', 'linkedin', 'github'
        ])
        for line in reversed(lines):
            candidate = clean_candidate(line)
            candidate_clean = candidate.lower()
            words = candidate.split()
            print(f"DEBUG: Ultra fallback considering: '{candidate_clean}'")
            if candidate_clean in name_blacklist:
                print(f"DEBUG: Skipped (blacklist): '{candidate_clean}'")
                continue
            if (2 <= len(words) <= 4 and all(w.isalpha() for w in words)):
                details['name'] = candidate
                print(f"DEBUG: details['name'] set to: {details['name']} (from ultra robust fallback)")
                break

    # Heuristic: Name is often the line before the phone number
    if not details.get('name') and details.get('phone'):
        section_headers_clean = set(h.strip().lower() for h in section_headers)
        for idx, line in enumerate(lines):
            if details['phone'] in line:
                if idx > 0:
                    candidate = lines[idx - 1].strip()
                    candidate_clean = ' '.join(candidate.lower().split()).strip()
                    words = candidate.split()
                    if (2 <= len(words) <= 4 and all(w.isalpha() for w in words)
                        and candidate_clean not in section_headers_clean):
                        details['name'] = candidate
                        print(f"DEBUG: details['name'] set to: {details['name']} (from line before phone heuristic)")
                        break

    # Final post-processing: if extracted name is two words and second is in blacklist, use only the first word
    if details.get('name'):
        name_words = details['name'].split()
        if len(name_words) == 2:
            section_headers_clean = set(h.strip().lower() for h in section_headers)
            if name_words[1].strip().lower() in section_headers_clean:
                details['name'] = name_words[0].title()
                print(f"DEBUG: details['name'] set to: {details['name']} (final post-processing, removed blacklisted second word)")

    # --- NEW: Try to extract name from profile section (Name : ... pattern) if still not found ---
    if not details.get('name'):
        profile_name = extract_name_from_profile_section(lines)
        if profile_name and not is_section_header_name(profile_name) and not is_date_line(profile_name):
            details['name'] = profile_name
            print(f"DEBUG: details['name'] set to: {details['name']} (from profile section Name : ... pattern)")

    # --- NEW FINAL FALLBACK: Scan for all-uppercase or title-case name-like lines anywhere in the resume ---
    if not details.get('name'):
        for line in lines:
            words = line.strip().split()
            if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
                # Check if all words are uppercase or title case
                if (all(w.isupper() for w in words) or all(w.istitle() for w in words)):
                    candidate_clean = ' '.join(line.lower().split())
                    # Avoid section headers and known labels
                    if not is_section_header_name(candidate_clean) and not any(lbl in candidate_clean for lbl in ["objective","contact","education","skills","experience","projects","languages","interests","profile","summary","email","mobile","phone","address"]):
                        details['name'] = line.title()
                        print(f"DEBUG: details['name'] set to: {details['name']} (from new final fallback)")
                        break

    # --- NEW FINAL-FINAL FALLBACK: Two consecutive single-word lines, all alpha, all uppercase or title case ---
    if not details.get('name'):
        for i in range(len(lines) - 1):
            w1 = lines[i].strip()
            w2 = lines[i+1].strip()
            if (w1 and w2 and
                w1.isalpha() and w2.isalpha() and
                (w1.isupper() or w1.istitle()) and (w2.isupper() or w2.istitle()) and
                not is_section_header_name(w1.lower()) and not is_section_header_name(w2.lower()) and
                not any(lbl in w1.lower() for lbl in ["objective","contact","education","skills","experience","projects","languages","interests","profile","summary","email","mobile","phone","address"]) and
                not any(lbl in w2.lower() for lbl in ["objective","contact","education","skills","experience","projects","languages","interests","profile","summary","email","mobile","phone","address"])):
                details['name'] = f"{w1.title()} {w2.title()}"
                print(f"DEBUG: details['name'] set to: {details['name']} (from two-line fallback)")
                break

    # --- REFINED ROBUST FINAL-FINAL FALLBACK: Only use 2-3 consecutive single-word lines as name if between EDUCATION and CONTACT ---
    if details.get('name') and len(details['name'].split()) == 1:
        section_headers_priority = ["education", "contact"]
        section_headers_exclude = ["languages", "skills", "interests", "hobbies", "projects", "achievements", "certifications"]
        header_indices = {h: -1 for h in section_headers_priority + section_headers_exclude}
        for idx, line in enumerate(lines):
            lclean = line.strip().lower()
            for h in header_indices:
                if h in lclean:
                    header_indices[h] = idx
        edu_idx = header_indices["education"]
        contact_idx = header_indices["contact"] if header_indices["contact"] != -1 else len(lines)
        # Find all candidate groups strictly between EDUCATION and CONTACT
        candidates = []
        for i in range(edu_idx+1, contact_idx-1):
            w1 = lines[i].strip()
            w2 = lines[i+1].strip()
            if (w1 and w2 and
                w1.isalpha() and w2.isalpha() and
                (w1.isupper() or w1.istitle()) and (w2.isupper() or w2.istitle()) and
                not is_section_header_name(w1.lower()) and not is_section_header_name(w2.lower()) and
                not any(lbl in w1.lower() for lbl in ["objective","contact","education","skills","experience","projects","languages","interests","profile","summary","email","mobile","phone","address"]) and
                not any(lbl in w2.lower() for lbl in ["objective","contact","education","skills","experience","projects","languages","interests","profile","summary","email","mobile","phone","address"])):
                # Optionally check for a third line
                name_parts = [w1.title(), w2.title()]
                if i+2 < contact_idx:
                    w3 = lines[i+2].strip()
                    if (w3 and w3.isalpha() and (w3.isupper() or w3.istitle()) and not is_section_header_name(w3.lower()) and not any(lbl in w3.lower() for lbl in ["objective","contact","education","skills","experience","projects","languages","interests","profile","summary","email","mobile","phone","address"])):
                        name_parts.append(w3.title())
                # Exclude if after any exclude section header
                exclude_after = False
                for h in section_headers_exclude:
                    if header_indices[h] != -1 and i > header_indices[h]:
                        exclude_after = True
                        break
                if exclude_after:
                    continue
                candidates.append((i, " ".join(name_parts)))
        # Pick the first valid candidate
        if candidates:
            best = candidates[0]
            if len(best[1].split()) > 1:
                details['name'] = best[1]
                print(f"DEBUG: details['name'] set to: {details['name']} (from refined robust fallback between EDUCATION and CONTACT)")

    # --- FINAL REFINED ROBUST FALLBACK: Allow blank lines between candidate name lines between EDUCATION and CONTACT ---
    if details.get('name') and len(details['name'].split()) == 1:
        section_headers_priority = ["education", "contact"]
        section_headers_exclude = ["languages", "skills", "interests", "hobbies", "projects", "achievements", "certifications"]
        header_indices = {h: -1 for h in section_headers_priority + section_headers_exclude}
        for idx, line in enumerate(lines):
            lclean = line.strip().lower()
            for h in header_indices:
                if h in lclean:
                    header_indices[h] = idx
        edu_idx = header_indices["education"]
        contact_idx = header_indices["contact"] if header_indices["contact"] != -1 else len(lines)
        # Find all candidate groups strictly between EDUCATION and CONTACT, allowing blank lines
        candidates = []
        i = edu_idx + 1
        while i < contact_idx:
            # Skip blank lines
            while i < contact_idx and not lines[i].strip():
                i += 1
            name_parts = []
            j = i
            while j < contact_idx and len(name_parts) < 3:
                w = lines[j].strip()
                if w and w.isalpha() and (w.isupper() or w.istitle()) and not is_section_header_name(w.lower()) and not any(lbl in w.lower() for lbl in ["objective","contact","education","skills","experience","projects","languages","interests","profile","summary","email","mobile","phone","address"]):
                    name_parts.append(w.title())
                elif w:
                    break
                j += 1
            if len(name_parts) > 1:
                # Exclude if after any exclude section header
                exclude_after = False
                for h in section_headers_exclude:
                    if header_indices[h] != -1 and i > header_indices[h]:
                        exclude_after = True
                        break
                if not exclude_after:
                    candidates.append((i, " ".join(name_parts)))
            i = j + 1 if j > i else i + 1
        # Pick the first valid candidate
        if candidates:
            best = candidates[0]
            if len(best[1].split()) > 1:
                details['name'] = best[1]
                print(f"DEBUG: details['name'] set to: {details['name']} (from final robust fallback between EDUCATION and CONTACT, tolerant to blanks)")

    # --- ULTIMATE ROBUST FALLBACK: Always try to find multi-word name between education/contact (or reasonable defaults) ---
    if details.get('name') and len(details['name'].split()) == 1:
        # Normalize lines for header detection
        norm_lines = [re.sub(r'\s+', ' ', l.strip().lower()) for l in lines]
        def find_header_idx(header):
            header_norm = header.lower().strip()
            for idx, l in enumerate(norm_lines):
                if header_norm == l:
                    return idx
            for idx, l in enumerate(norm_lines):
                if header_norm in l:
                    return idx
            return -1
        section_headers_priority = ["education", "contact"]
        section_headers_exclude = ["languages", "skills", "interests", "hobbies", "projects", "achievements", "certifications"]
        header_indices = {h: find_header_idx(h) for h in section_headers_priority + section_headers_exclude}
        edu_idx = header_indices["education"] if header_indices["education"] != -1 else 0
        contact_idx = header_indices["contact"] if header_indices["contact"] != -1 else len(lines)
        # Find all candidate groups strictly between EDUCATION and CONTACT, allowing blank lines
        candidates = []
        i = edu_idx + 1
        while i < contact_idx:
            # Skip blank lines
            while i < contact_idx and not lines[i].strip():
                i += 1
            name_parts = []
            j = i
            while j < contact_idx and len(name_parts) < 3:
                w = lines[j].strip()
                if w and w.isalpha() and (w.isupper() or w.istitle()) and not is_section_header_name(w.lower()) and not any(lbl in w.lower() for lbl in ["objective","contact","education","skills","experience","projects","languages","interests","profile","summary","email","mobile","phone","address"]):
                    name_parts.append(w.title())
                elif w:
                    break
                j += 1
            if len(name_parts) > 1:
                # Exclude if after any exclude section header
                exclude_after = False
                for h in section_headers_exclude:
                    hidx = header_indices[h]
                    if hidx != -1 and i > hidx:
                        exclude_after = True
                        break
                if not exclude_after:
                    candidates.append((i, " ".join(name_parts)))
            i = j + 1 if j > i else i + 1
        # Pick the first valid candidate
        if candidates:
            best = candidates[0]
            if len(best[1].split()) > 1:
                details['name'] = best[1]
                print(f"DEBUG: details['name'] set to: {details['name']} (from ultimate robust fallback between EDUCATION and CONTACT, tolerant to blanks and header variants)")

    # --- TRULY ROBUST FALLBACK: Find best group of 1-3 lines between EDUCATION and CONTACT, concatenating words, 2-5 total words, not hardcoded ---
    if details.get('name') and len(details['name'].split()) == 1:
        norm_lines = [re.sub(r'\s+', ' ', l.strip().lower()) for l in lines]
        def find_header_idx(header):
            header_norm = header.lower().strip()
            for idx, l in enumerate(norm_lines):
                if header_norm == l:
                    return idx
            for idx, l in enumerate(norm_lines):
                if header_norm in l:
                    return idx
            return -1
        section_headers_priority = ["education", "contact"]
        section_headers_exclude = ["languages", "skills", "interests", "hobbies", "projects", "achievements", "certifications"]
        header_indices = {h: find_header_idx(h) for h in section_headers_priority + section_headers_exclude}
        edu_idx = header_indices["education"] if header_indices["education"] != -1 else 0
        contact_idx = header_indices["contact"] if header_indices["contact"] != -1 else len(lines)
        # Find all candidate groups strictly between EDUCATION and CONTACT, allowing blank lines
        best_group = None
        best_word_count = 0
        i = edu_idx + 1
        while i < contact_idx:
            # Skip blank lines
            while i < contact_idx and not lines[i].strip():
                i += 1
            group = []
            j = i
            while j < contact_idx and len(group) < 3:
                w = lines[j].strip()
                if w and not is_section_header_name(w.lower()) and all(x.isalpha() for x in w.replace(' ', '').split()):
                    # All words in line must be alpha, and line must be uppercase or title case
                    words = w.split()
                    if all(word.isupper() or word.istitle() for word in words):
                        group.append(w.title())
                    else:
                        break
                elif w:
                    break
                j += 1
            # Only consider groups with 2-5 total words
            total_words = sum(len(g.split()) for g in group)
            if 2 <= total_words <= 5:
                # Exclude if after any exclude section header
                exclude_after = False
                for h in section_headers_exclude:
                    hidx = header_indices[h]
                    if hidx != -1 and i > hidx:
                        exclude_after = True
                        break
                if not exclude_after and total_words > best_word_count:
                    best_group = group
                    best_word_count = total_words
            i = j + 1 if j > i else i + 1
        if best_group:
            details['name'] = ' '.join(best_group)
            print(f"DEBUG: details['name'] set to: {details['name']} (from truly robust fallback, best group between EDUCATION and CONTACT)")

    return details

def looks_like_name(line):
    words = line.strip().split()
    if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
        return True
    return False

def split_linkedin_handle(handle):
    # Remove digits and special chars
    handle = re.sub(r'[^a-zA-Z]', '', handle)
    if not handle or len(handle) < 4:
        return None
    # If camel case, split on uppercase
    if any(c.isupper() for c in handle[1:]):
        parts = re.findall(r'[A-Z][a-z]*', handle)
        if len(parts) >= 2:
            return ' '.join([p.capitalize() for p in parts])
    # Try to split in the middle (heuristic: first half/second half)
    # Use vowel-consonant transition for Indian names
    vowels = 'aeiou'
    best_split = None
    best_score = 0
    for i in range(2, len(handle)-2):
        left, right = handle[:i], handle[i:]
        # Prefer splits where both sides start with uppercase after split
        score = 0
        if left[0] in vowels or right[0] in vowels:
            score -= 1
        if left[-1] in vowels and right[0] not in vowels:
            score += 1
        if left[-1] not in vowels and right[0] in vowels:
            score += 1
        if 3 <= len(left) <= 8 and 3 <= len(right) <= 8:
            score += 2
        if score > best_score:
            best_score = score
            best_split = (left, right)
    if best_split:
        return f"{best_split[0].capitalize()} {best_split[1].capitalize()}"
    # Fallback: split into two halves
    mid = len(handle) // 2
    return f"{handle[:mid].capitalize()} {handle[mid:].capitalize()}"

# === Section Header Names and Helper ===
section_header_names = set([
    'extra curricular activities', 'skills', 'projects', 'internships', 'work experience', 'education', 'educational details',
    'languages', 'certifications', 'achievements', 'personal details', 'profile', 'summary', 'career objective',
    'objective', 'contact', 'email', 'mobile', 'phone', 'address', 'declaration', 'signature', 'place', 'date',
    'major', 'minor', 'courses', 'interests', 'statement', 'references', 'reference', 'hobbies', 'activities',
    'additional details', 'trainings', 'training', 'publications', 'awards', 'responsibilities', 'accomplishments',
    'statement of purpose', 'curriculum vitae', 'resume', 'biodata', 'web page designing', 'seeking a job role',
    'student', 'performance', 'predictor', 'technology', 'used', 'flutter', 'dart', 'firebase',
    'patents and publications', 'publications and patents', 'conferences', 'journal', 'thesis', 'in submission', 'patent','professional summary',
    'fresher', 'trainee', 'certificates', 'workshops', 'seminars', 'certificates workshops seminars', 'digital marketing training',
    'languages known', 'hobby', 'links', 'workshops & seminars', 'certificates, workshops & seminars'
])
def is_section_header_name(candidate):
    if not candidate:
        return False
    candidate_clean = candidate.strip().lower()
    # Remove punctuation and extra spaces
    candidate_clean = re.sub(r'[^a-z ]', '', candidate_clean)
    candidate_clean = ' '.join(candidate_clean.split())
    
    # Check for exact matches in section_header_names
    if candidate_clean in section_header_names:
        return True
    
    # Check for partial matches for common section headers
    common_headers = ['education', 'experience', 'skills', 'certifications', 'achievements', 
                      'contact', 'summary', 'profile', 'objective', 'training', 'seminars',
                      'workshops', 'digital marketing', 'details']
    
    # Specific check for "contact details" which is often misidentified as a name
    if candidate_clean == 'contact details':
        return True
                      
    for header in common_headers:
        if header in candidate_clean:
            return True
    
    # Additional check for common formats that indicate section headers
    if candidate_clean.endswith('details') or candidate_clean.startswith('contact'):
        return True
            
    return False

# === Education Checker ===
def check_education(text):
    lines = text.splitlines()
    text_lower = text.lower()

    def is_edu_context(line, idx):
        year_pat = r'(19|20)\d{2}'
        institution_keywords = [
            'university', 'college', 'school', 'institute', 'academy', 'faculty', 'polytechnic', 'junior college', 'public school', 'high school'
        ]
        for offset in range(-2, 3):
            i = idx + offset
            if 0 <= i < len(lines):
                l = lines[i].lower()
                if re.search(year_pat, l):
                    return True
                if any(inst in l for inst in institution_keywords):
                    return True
        for offset in range(-5, 1):
            i = idx + offset
            if 0 <= i < len(lines):
                l = lines[i].lower()
                if any(h in l for h in ['education', 'academic', 'qualification']):
                    return True
        return False

    def extract_education_section(lines):
        edu_start = None
        edu_end = None
        for i, line in enumerate(lines):
            if (
                'education' in line.lower() or
                'education history' in line.lower() or
                'academic' in line.lower() or
                'academic qualification' in line.lower()
            ):
                edu_start = i
                break
        if edu_start is not None:
            for j in range(edu_start + 1, len(lines)):
                if is_section_header_name(lines[j]):
                    edu_end = j
                    break
            if edu_end is None:
                edu_end = len(lines)
            return lines[edu_start:edu_end]
        return []

    def contains_any_pattern_in_context(patterns, skip_cert_course_for_ug=False, restrict_to_edu_section=False, ssc_mode=False):
        search_lines = lines
        if restrict_to_edu_section:
            edu_section = extract_education_section(lines)
            if edu_section:
                search_lines = edu_section
        for idx, line in enumerate(search_lines):
            line_lower = line.lower()
            if skip_cert_course_for_ug:
                skip_section_headers = ["certification", "certifications", "certificate", "course", "courses", "training"]
                skip_this_line = False
                for offset in range(-2, 2):
                    i = idx + offset
                    if 0 <= i < len(search_lines):
                        l = search_lines[i].lower()
                        if any(header in l for header in skip_section_headers):
                            skip_this_line = True
                            break
                if skip_this_line:
                    continue
            for pat in patterns:
                if ssc_mode:
                    # Only match 'x' and 'secondary' if not part of 'higher secondary', 'senior secondary', '10+2', etc.
                    if pat == r"x":
                        # Match if line contains 'x' or '10th' with education context
                        if re.search(r"\bx\b|\b10th\b|secondary \(x\)|\(x\)", line_lower.strip()):
                            if is_edu_context(line, idx):
                                return True
                        continue
                    if pat == r"secondary":
                        # Skip if line contains 'higher' or 'senior' or '10+2'
                        if ("higher" in line_lower or "senior" in line_lower or "10+2" in line_lower):
                            continue
                        # Only match if 'secondary' is not part of 'higher secondary', 'senior secondary', etc.
                        if re.search(r"\bsecondary\b", line_lower) or "secondary (x)" in line_lower:
                            if is_edu_context(line, idx):
                                return True
                        continue
                    # Special handling for explicit 10th standard or SSC mentions
                    if pat in [r"10th", r"ssc", r"matriculation"]:
                        if re.search(pat, line_lower):
                            return True
                # --- FIX: For SSC, skip 'secondary' if line contains 'higher' or 'senior' ---
                if pat == r"secondary" and ("higher" in line_lower or "senior" in line_lower):
                    continue
                if re.search(pat, line, re.IGNORECASE):
                    # Special handling for UG patterns to avoid false positives from PG diploma
                    if "diploma" in pat and any(pg_term in line_lower for pg_term in ["post graduate", "postgraduate", "pgdm", "pgd", "master"]):
                        continue  # Skip if it's a postgraduate diploma
                    if is_edu_context(line, idx):
                        return True
        return False

    # PG patterns (fixed)
    pg_patterns = [
        r"\b(m[.\s]*s[.\s]*c[.]?|msc|master of science)\b",
        r"\b(m[.\s]*b[.\s]*a[.]?|mba|master of business administration)\b",
        r"\b(m[.\s]*c[.\s]*a[.]?|mca|master of computer application[s]?)\b",
        r"\b(m[.\s]*t[.\s]*e[.\s]*c[.\s]*h[.]?|mtech|master of technology)\b",
        r"\b(m[.\s]*e[.]*|me|master of engineering)\b",
        r"\b(m[.\s]*a[.]*|ma|master of arts)\b",
        r"\b(m[.\s]*c[.\s]*o[.\s]*m[.]?|mcom|master of commerce)\b",
        r"\bpgdm\b", r"\bpgd\b", 
        r"\bpost[ -]?graduate\b", r"\bpost[ -]?graduation\b", r"\bpost graduate diploma\b", r"\bpost[ -]?grad\b",
        r"pursuing.*master", r"pursuing.*mba", r"pursuing.*postgraduate", r"pursuing.*pgdm", r"pursuing.*pgd"
    ]
    dr_patterns = [r"\bphd\b", r"\bdoctorate\b", r"\bdphil\b"]
    g_patterns = [r"b[.\s]*s[.\s]*c[.]*", r"bsc", r"b[.\s]*a[.]*", r"ba", r"b[.\s]*b[.\s]*a[.]*", r"bba", r"b[.\s]*c[.\s]*a[.]*", r"bca", r"b[.\s]*c[.\s]*o[.\s]*m[.]*", r"bcom", r"b[.\s]*e[.\s]*", r"be", r"bachelor", r"ug", r"undergraduate", r"b\.tech", r"btech", r"bachelor's degree"]
    ug_patterns = [
        r"\bdiploma\b",  # Add this to catch 'Diploma' in any context
        r"\bpolytechnic\b", 
        r"\bd\.pharma\b", 
        r"\bdiploma in pharmacy\b", 
        r"\biti\b", 
        r"\bindustrial training institute\b"
    ]
    hsc_patterns = [r"intermediate", r"hsc", r"12th", r"xii", r"higher secondary", r"10\+2", r"senior secondary"]
    ssc_patterns = [
        r"ssc", r"10th", r"matriculation", r"secondary", r"x", r"10th standard", 
        r"secondary school certificate", r"matric", r"secondary \(x\)", r"\(x\)",
        r"class 10", r"class x", r"std 10", r"std x", r"grade 10", r"grade x"
    ]

    found_g = contains_any_pattern_in_context(g_patterns)
    # PG: Only match in education section or with strong education context
    found_pg = contains_any_pattern_in_context(pg_patterns, restrict_to_edu_section=False)
    # --- SPECIAL: Loosen context for PGDM/PGD and all Master's degrees---
    if not found_pg:
        for idx, line in enumerate(lines):
            # Check for PGDM/PGD and M.Sc, MBA, etc. patterns
            if (re.search(r'\bpgdm\b', line, re.IGNORECASE) or 
                re.search(r'\bpgd\b', line, re.IGNORECASE) or
                re.search(r'\bm\.sc\b', line, re.IGNORECASE) or
                re.search(r'\bmsc\b', line, re.IGNORECASE) or
                re.search(r'\bm\.b\.a\b|\bmba\b', line, re.IGNORECASE) or
                re.search(r'\bm\.c\.a\b|\bmca\b', line, re.IGNORECASE) or
                re.search(r'\bm\.tech\b|\bmtech\b', line, re.IGNORECASE) or
                re.search(r'\bm\.e\b|\bme\b', line, re.IGNORECASE) or
                re.search(r'\bm\.a\b|\bma\b', line, re.IGNORECASE) or
                re.search(r'\bm\.com\b|\bmcom\b', line, re.IGNORECASE) or
                re.search(r'\bmaster', line, re.IGNORECASE)):
                found_pg = True
                print(f"[DEBUG] PG degree found in line: '{line}' (idx={idx})")
                break
    if not found_pg:
        # Fallback: allow PG match only if in strong education context (not job titles)
        for idx, line in enumerate(lines):
            for pat in pg_patterns:
                if re.search(pat, line, re.IGNORECASE):
                    if is_edu_context(line, idx):
                        print(f"[DEBUG] PG pattern matched: '{pat}' in line: '{line}' (idx={idx})")
                        found_pg = True
                        break
            if found_pg:
                break
    found_dr = contains_any_pattern_in_context(dr_patterns)
    # --- FIX: Loosen context for UG (Diploma/ITI) detection ---
    found_ug = False
    for idx, line in enumerate(lines):
        for pat in ug_patterns:
            if re.search(pat, line, re.IGNORECASE):
                # Avoid false positives from PG diplomas
                if "diploma" in pat and any(pg_term in line.lower() for pg_term in ["post graduate", "postgraduate", "pgdm", "pgd", "master"]):
                    continue
                found_ug = True
                break
        if found_ug:
            break
            
    # --- Direct check for SSC in text (since it's often missed) ---
    found_direct_ssc = False
    for line in lines:
        line_lower = line.lower()
        if "secondary (x)" in line_lower or "10th" in line_lower or "ssc" in line_lower or "matric" in line_lower:
            print(f"[DEBUG] SSC found in line: '{line}'")
            found_direct_ssc = True
            break
    found_hsc = contains_any_pattern_in_context(hsc_patterns)
    # --- FIX: Use ssc_mode for SSC patterns and enhanced detection ---
    found_ssc = contains_any_pattern_in_context(ssc_patterns, ssc_mode=True)
    
    # Additional check for SSC/10th if not found already
    if not found_ssc:
        for idx, line in enumerate(lines):
            line_lower = line.lower()
            # Check for keywords specific to SSC/10th
            if "secondary (x)" in line_lower or "10th standard" in line_lower or "matric" in line_lower:
                found_ssc = True
                break
            # Check for percentage/marks with class 10 context
            if "10" in line_lower and any(word in line_lower for word in ["percentage", "marks", "grade", "cgpa"]):
                found_ssc = True
                break
            # Check for school with year indication
            if "school" in line_lower and re.search(r'201\d|20\d{2}', line):
                found_ssc = True
                break

    return {
        'pg': found_pg,
        'dr': found_dr,
        'g': found_g,
        'ug': found_ug,
        'hsc': found_hsc,
        'ssc': found_ssc or found_direct_ssc  # Use either method to detect SSC
    }


# === Skill Extractor ===

def is_education_line(line):
    """Return True if a line looks like it refers to a school/college/institute/university."""
    EDU_KEYWORDS = [
        "school", "college", "university", "institute", "academy", "faculty", "polytechnic",
        "foundation", "b.sc", "b.e", "b.tech", "m.sc", "m.tech", "phd", "ssc", "hsc", "diploma",
        "junior college", "department"
    ]
    line_lower = line.lower()
    return any(keyword in line_lower for keyword in EDU_KEYWORDS)

def extract_skills_from_block(text, name=None):
    import re
    skill_headers = [
        r"skills?", r"software skills?", r"technical skills?", r"core skills?", r"it skills?", r"computer skills?",
        r"key skills?", r"professional skills?", r"relevant skills?", r"skill set?", r"competencies?",
        r"strengths?", r"core competencies?", r"expertise", r"technical proficiency", r"areas of expertise",
        r"highlights", r"key qualifications", r"abilities", r"summary of qualifications", r"Life Skills", r"KEY COMPETENCIES",
        # Handle spaced-out text like "S K I L L S"
        r"s\s*k\s*i\s*l\s*l\s*s?", r"s\s*k\s*i\s*l\s*l\s*s?\s*:?"
    ]
    # Pattern to detect section headers with underscores or dashes
    separator_pattern = r"^_{5,}$|^-{5,}$|^_{5,}\s*\w+|^\w+\s*_{5,}$|^-{5,}\s*\w+|^\w+\s*-{5,}$"
    # Exclude section headers with underscores/dashes from being considered skills
    underscored_skills_pattern = r"(?:_{3,}.*skills?|skills?.*_{3,})"
    header_regex = re.compile(r"^\s*(" + "|".join(skill_headers) + r")\s*:?$", re.I)
    underscored_header_regex = re.compile(underscored_skills_pattern, re.I)

    # Additional skill indicators that might appear in resume text
    skill_indicator_phrases = [
        r"proficient in\s+(.+)",
        r"knowledge of\s+(.+)",
        r"familiar with\s+(.+)",
        r"experienced in\s+(.+)",
        r"expertise in\s+(.+)",
        r"specializing in\s+(.+)",
        r"skilled in\s+(.+)",
        r"working knowledge of\s+(.+)",
        r"trained in\s+(.+)"
    ]

    skills_section = []
    lines = text.splitlines()
    capture = False
    
    # First pass: find the traditional skills section
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Check for normal skills header
        if header_regex.match(line_stripped):
            capture = True
            continue
        
        # Skip lines that are section separators or underscored headers
        if re.match(r"^[_\-=\.]{5,}$", line_stripped) or underscored_header_regex.match(line_stripped.lower()):
            continue
            
        if capture:
            # Stop at next section header or empty line
            known_sections = ['CONTACT', 'EDUCATION', 'EXPERIENCE', 'PROJECTS', 'CERTIFICATIONS', 
                            'LANGUAGES', 'VOLUNTEER', 'PROFILE', 'SUMMARY', 'OBJECTIVE', 
                            'WORK EXPERIENCE', 'ACADEMIC', 'ACHIEVEMENTS', 'AWARDS', 'HOBBIES',
                            'PERSONAL INFORMATION', 'CONTACT INFORMATION', 'PERSONAL PROFILE',
                            'E D U C A T I O N', 'P R O J E C T S', 'C E R T I F I C A T I O N S', 
                             'L A N G U A G E S', 'P R O F I L E', 'S U M M A R Y', 'DECLLERATION',
                             'DECLARATION', 'ACADMIC QUALIFICATION', 'CAREEROBJECTIVE', 'INTERESTS', 
                             'INTERNSHIP', 'PROJECTS', 'CERTIFICATIONS', 'DECLARATION']
            if line_stripped.upper() in known_sections or not line_stripped or re.match(r"^[_\-=\.]{5,}$", line_stripped):
                break
            # --- NEW FILTERS ---
            # Skip lines that are just numbers or contain a 6-digit number (pincode)
            if re.fullmatch(r'\d+', line_stripped):
                continue
            if re.search(r'\b\d{6}\b', line_stripped):
                continue
            # Skip lines that are mostly digits (e.g., phone numbers)
            if len(re.sub(r'\D', '', line_stripped)) > len(line_stripped) // 2:
                continue
            # Skip lines that are just section headers or contain only 'skills' words
            if line_stripped.lower() in ["skills", "designer skills", "technical skills", "core skills", "soft skills", "hard skills"]:
                continue
            skills_section.append(line_stripped)

    # Second pass: look for skill indicator phrases in the entire text when skills section is empty or small
    if len(skills_section) < 3:
        for line in lines:
            line_stripped = line.strip()
            for pattern in skill_indicator_phrases:
                match = re.search(pattern, line_stripped, re.IGNORECASE)
                if match:
                    skills_section.append(match.group(1))

    print(f"\n[DEBUG] extract_skills_from_block - Skills section captured: {skills_section}")

    # Enhanced blacklist for skills processing
    skill_blacklist = {
        "technical skills", "personal skills", "skills", "competencies", "strengths", "abilities",
        "father name", "mother name", "date of birth", "marital status", "gender", "nationality",
        "male", "female", "married", "un-married", "single", "indian", "muslim", "hindu", "christian",
        "address", "phone", "email", "contact", "mobile", "tel", "name", "details", "information",
        "profile", "summary", "objective", "qualification", "percentage", "marks", "grade", "score",
        "result", "passing", "year", "board", "college", "school", "university", "institute",
        "academy", "department", "faculty", "pursuing", "completed", "ongoing", "current",
        "previous", "past", "present", "declaration", "declare", "correct", "true", "knowledge",
        "signature", "place", "date", "languages known", "hobbies", "personal profile"
    }

    # Now, process each line to extract only the actual skills
    skills_flat = []
    for item in skills_section:
        print(f"[DEBUG] Processing item: {repr(item)}")
        # Remove bullet points and special characters
        item = re.sub(r"^[•\-*\u2022]\s*", "", item)  # Also handle unicode bullet \u2022
        # If there's a colon, only take the part after the colon
        if ':' in item:
            item = item.split(':', 1)[1].strip()
            print(f"[DEBUG] After colon split: {repr(item)}")
        # Replace ' and ' with ',' for easier splitting, but only if not inside parentheses
        item = re.sub(r'\s+and\s+', ',', item, flags=re.I)
        # Remove parentheses but keep content
        item = re.sub(r"[()]", "", item)
        # Split by commas and semicolons
        for skill in re.split(r",|;", item):
            skill = skill.strip()
            print(f"[DEBUG] Split skill: {repr(skill)}")
            # --- PINCODE/NUMBER FILTER ---
            # Skip if skill is a 6-digit number (pincode) or contains a 6-digit number
            if re.fullmatch(r'\d{6}', skill) or re.search(r'\b\d{6}\b', skill):
                print(f"[DEBUG] Skipped pincode skill: {skill}")
                continue
            # Skip if skill is a number followed by 'Skills' or 'Designer Skills', or contains a 6-digit number followed by 'Skills'
            if re.fullmatch(r'\d+\s*(Skills|Designer Skills)', skill, re.I) or re.search(r'\b\d{6}\b\s*(Skills|Designer Skills)', skill, re.I):
                print(f"[DEBUG] Skipped number+skills: {skill}")
                continue
            # Skip if the skill contains many underscores or separator characters (likely a section header)
            if skill.count('_') > 4 or skill.count('-') > 4 or skill.count('=') > 4 or skill.count('.') > 5:
                print(f"[DEBUG] Skipped separator line: {skill}")
                continue
            # Skip underscored section headers with "skills"
            if re.search(r'_{3,}.*\bskills?\b', skill.lower()) or re.search(r'\bskills?\b.*_{3,}', skill.lower()):
                print(f"[DEBUG] Skipped underscored skills header: {skill}")
                continue
            # Enhanced filtering
            if (skill and len(skill) > 1 and 
                not re.search(r'[\d@#$%^&*]', skill) and
                skill.lower() not in skill_blacklist and
                not re.search(r'^[A-Z\s]+$', skill) and  # Skip all caps section headers
                not re.search(r'[:\-]\s*[A-Z]', skill) and  # Skip lines with colons followed by caps
                not re.search(r'\d{1,2}[-/]\w+[-/]\d{2,4}', skill) and  # Skip date patterns
                not re.search(r'^\d{4}$', skill) and  # Skip year numbers only
                not any(header_word in skill.lower() for header_word in ["name", "date", "status", "nationality", "gender", "marital"])):
                skills_flat.append(skill)
                print(f"[DEBUG] Added skill: {skill}")
            else:
                print(f"[DEBUG] Skipped skill: {skill}")
    # Remove duplicates while preserving order
    seen = set()
    clean_skills = []
    for skill in skills_flat:
        if skill not in seen:
            seen.add(skill)
            clean_skills.append(skill)
    # --- FINAL PINCODE/NUMBER FILTER ---
    filtered_skills = []
    for skill in clean_skills:
        # Remove any skill that is a 6-digit number, contains a 6-digit number, or number+Skills
        if re.fullmatch(r'\d{6}', skill) or re.search(r'\b\d{6}\b', skill):
            continue
        if re.fullmatch(r'\d+\s*(Skills|Designer Skills)', skill, re.I) or re.search(r'\b\d{6}\b\s*(Skills|Designer Skills)', skill, re.I):
            continue
        filtered_skills.append(skill)
    print(f"[DEBUG] Final skills from block: {filtered_skills}")
    return filtered_skills

def extract_bullet_skills(text, name=None):
    import re
    # Collect all bullet points (•, -, *, >, ➢, ✓, ✔, ◆) in the document, even empty ones
    # Expanded to catch more bullet point styles and formats
    bullets = re.findall(r"(?:^[•\-\*>➢✓✔◆◼►▪▸][ \t]*.+|[\n\r][•\-\*>➢✓✔◆◼►▪▸][ \t]*.+)", text, re.MULTILINE)
    
    # Also look for numbered bullets (1., 2., i., ii., a., b.)
    numbered_bullets = re.findall(r"(?:^[\d]+\.[ \t]*.+|[\n\r][\d]+\.[ \t]*.+)", text, re.MULTILINE)
    bullets.extend(numbered_bullets)
    
    # Roman numeral bullets
    roman_bullets = re.findall(r"(?:^[ivxIVX]+\.[ \t]*.+|[\n\r][ivxIVX]+\.[ \t]*.+)", text, re.MULTILINE)
    bullets.extend(roman_bullets)
    
    # Alpha bullets
    alpha_bullets = re.findall(r"(?:^[a-zA-Z][\.)]+[ \t]*.+|[\n\r][a-zA-Z][\.)]+[ \t]*.+)", text, re.MULTILINE)
    bullets.extend(alpha_bullets)
    
    # Clean up all bullet items
    bullets = [b.lstrip("\n\r•-*>➢✓✔◆◼►▪▸0123456789ivxIVXabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.) \t") for b in bullets]
    
    candidate_skills = set()
    
    # Enhanced blacklist to filter out personal information and non-skills
    blacklist = {
        "resume", "cv", "profile", "summary", "career objectives", "objectives",
        "project", "internship", "education", "percentage", "system", "industry", "crusher",
        "accomplishments", "certifications", "languages", "professional summary",
        # Personal information
        "father name", "mother name", "date of birth", "marital status", "gender", "nationality",
        "male", "female", "married", "un-married", "single", "indian", "muslim", "hindu", "christian",
        "address", "phone", "email", "contact", "mobile", "tel",
        # Dates and numbers
        "2020", "2018", "2017", "2021", "2022", "2023", "2024", "2025",
        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        # Section headers
        "personal profile", "languages known", "hobbies", "declaration", "academic qualification",
        "career objective", "qualification", "institution", "university", "board", "year of passing",
        # Common non-skill words
        "name", "details", "information", "profile", "summary", "objective", "qualification",
        "percentage", "marks", "grade", "score", "result", "passing", "year", "board",
        "college", "school", "university", "institute", "academy", "department", "faculty",
        "pursuing", "completed", "ongoing", "current", "previous", "past", "present",
        "declaration", "declare", "correct", "true", "knowledge", "signature", "place", "date"
    }

    skill_whitelist = {
        "project management", "teamwork", "time management", "life skills", "effective communication", 
        "quick learner", "problem solving", "leadership", "communication", "team work", "self confident",
        "self-confident", "adaptability", "creativity", "critical thinking", "decision making",
        "organization", "planning", "research", "analysis", "presentation", "negotiation",
        "customer service", "sales", "marketing", "finance", "accounting", "human resources",
        "quality assurance", "testing", "development", "programming", "coding", "design",
        "pharmacy", "dispensing", "clinical pharmacy", "patient counseling", "inventory management",
        "pharmacy law", "prescription", "medication", "drug", "healthcare", "pharmaceutical",
        "active listening", "listening skills", "good communication", "communication skills",
        "hard working", "hardworking", "learn new things", "learning", "focused", "confident",
        "positive attitude", "handling experience", "dissolution apparatus", "ultra-violet spectroscopy",
        "sonicator", "spectroscopy", "ms office", "microsoft office", "technical skills",
        "personal skills", "soft skills", "interpersonal skills", "listening music", "travelling",
        "traveling", "music", "reading", "writing", "speaking", "english", "hindi", "urdu", "telugu",
        "manipuri", "playing football", "playing basketball", "reading books", "sincere at work",
        "responsible", "ability to work in a team", "achieve goal", "problem solving skills",
        "punctuality", "flexible to work", "basic computer skills", "ms office", "ms-off",
        "football", "basketball", "books", "sports", "team sports", "reading", "literature"
    }

    if name:
        for part in name.lower().split():
            blacklist.add(part)
    
    for b in bullets:
        skill = b.lower().strip(" .,:;")
        if not skill or len(skill) < 2:
            continue
            
        # Skip if it's in blacklist
        if skill in blacklist:
            continue
            
        # Skip if it contains date patterns
        if re.search(r'\d{1,2}[-/]\w+[-/]\d{2,4}', skill) or re.search(r'\d{4}', skill):
            continue
            
        # Skip if it's a section header or contains typical header words
        if any(header_word in skill for header_word in ["name", "date", "status", "nationality", "gender", "marital"]):
            continue
            
        # Always include if in skill_whitelist
        if skill in skill_whitelist:
            candidate_skills.add(skill)
            continue
            
        # Only keep short lines (max 5 words), ignore obviously non-skill bullets
        if (skill and skill not in blacklist and 1 < len(skill.split()) <= 5
                and not skill.replace('.', '').isdigit()
                and not re.search(r'^[A-Z\s]+$', skill)  # Skip all caps section headers
                and not re.search(r'[:\-]\s*[A-Z]', skill)):  # Skip lines with colons followed by caps
            candidate_skills.add(skill)
    
    return candidate_skills

def extract_short_skills_from_text(text, name=None):
    """
    Extract short skills (1-3 words) from paragraph-style skills text using NLP.
    """
    import re
    
    # Define common skill keywords that we want to extract
    skill_keywords = {
        # Communication skills
        "communication", "communication skills", "verbal communication", "written communication",
        "interpersonal skills", "presentation skills", "public speaking", "listening skills",
        
        # Technical skills
        "ms office", "microsoft office", "excel", "word", "powerpoint", "computer skills",
        "basic computer skills", "typing", "data entry", "spreadsheets", "database",
        
        # Soft skills
        "teamwork", "team work", "team player", "leadership", "problem solving", "problem-solving",
        "critical thinking", "decision making", "time management", "organization", "planning",
        "adaptability", "flexibility", "creativity", "innovation", "attention to detail",
        "punctuality", "reliability", "responsibility", "sincerity", "hard working", "hardworking",
        "quick learner", "fast learner", "learning ability", "self motivated", "self-motivated",
        
        # Work-related skills
        "customer service", "sales", "marketing", "negotiation", "project management",
        "quality assurance", "quality control", "research", "analysis", "reporting",
        
        # Pharmacy-specific skills
        "pharmacy", "dispensing", "prescription", "medication", "drug knowledge",
        "patient counseling", "inventory management", "clinical pharmacy", "pharmaceutical",
        "dissolution apparatus", "spectroscopy", "ultra-violet", "sonicator",
        
        # Languages
        "english", "hindi", "urdu", "telugu", "manipuri", "spanish", "french", "german",
        
        # Interests that can be skills
        "football", "basketball", "sports", "reading", "writing", "music", "traveling",
        "travelling", "books", "literature", "team sports"
    }
    
    # Blacklist for non-skills
    skill_blacklist = {
        "my skills", "skills", "skill", "com skills", "technical skills", "personal skills",
        "soft skills", "hard skills", "key skills", "core skills", "professional skills",
        "work", "work and", "work basic", "work in", "work on", "work with", "work to",
        "at work", "to work", "of work", "work experience", "work environment",
        "goal", "achieve goal", "team achieve", "team achieve goal",
        "responsible", "sincere at", "sincere at work", "flexible to", "flexible to work",
        "ability to", "ability to work", "ability to work in", "ability to work in a team",
        "good communication skills and", "problem solving skills punctuality and",
        "basic computer skills on", "on ms-off", "ms-off"
    }
    
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    
    # Split text into potential skill phrases
    # Split by common separators and conjunctions
    phrases = re.split(r'[,;]|\s+and\s+|\s+or\s+|\s+with\s+|\s+including\s+|\s+such\s+as\s+', text)
    
    extracted_skills = set()
    
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase or len(phrase) < 2:
            continue
            
        # Remove percentage indicators and numbers
        phrase = re.sub(r'\d+%', '', phrase)
        phrase = re.sub(r'\d+', '', phrase)
        phrase = re.sub(r'\s+', ' ', phrase).strip()
        
        # Skip if too long (more than 4 words)
        if len(phrase.split()) > 4:
            continue
            
        # Skip if in blacklist
        if phrase in skill_blacklist:
            continue
            
        # Check if phrase contains any skill keywords
        for skill_keyword in skill_keywords:
            if skill_keyword in phrase:
                # Extract the skill keyword, not the entire phrase
                extracted_skills.add(skill_keyword)
                break
        else:
            # If no keyword found, check if the phrase itself is a skill
            if phrase in skill_keywords:
                extracted_skills.add(phrase)
    
    # Additional processing: look for specific patterns
    # Look for "skills" patterns
    skill_patterns = [
        r'(\w+\s+skills?)',  # e.g., "communication skills", "problem solving skills"
        r'(good\s+\w+)',     # e.g., "good communication"
        r'(basic\s+\w+)',    # e.g., "basic computer"
        r'(\w+\s+ability)',  # e.g., "learning ability"
        r'(\w+\s+management)', # e.g., "time management"
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            match = match.strip()
            if 1 <= len(match.split()) <= 3 and match not in skill_blacklist:
                extracted_skills.add(match)
    
    # Clean up and return
    cleaned_skills = []
    for skill in extracted_skills:
        skill = skill.strip()
        if skill and len(skill) > 1 and skill not in skill_blacklist:
            # Capitalize first letter of each word
            skill = ' '.join(word.capitalize() for word in skill.split())
            cleaned_skills.append(skill)
    
    return sorted(list(set(cleaned_skills)))

def extract_advanced_skills(text, name=None):
    """
    Enhanced skills extraction function that uses multiple techniques:
    1. Looks for skill indicator phrases like "proficient in X"
    2. Uses better bullet point detection
    3. Handles numbered lists and other formatting
    """
    import re
    
    # Skill indicator phrases that might indicate what follows is a skill
    skill_indicators = [
        r"proficient in\s+(.+?)(\.|\n|$|,|;)",
        r"skilled in\s+(.+?)(\.|\n|$|,|;)",
        r"knowledge of\s+(.+?)(\.|\n|$|,|;)", 
        r"familiar with\s+(.+?)(\.|\n|$|,|;)",
        r"experience (?:in|with)\s+(.+?)(\.|\n|$|,|;)",
        r"expertise in\s+(.+?)(\.|\n|$|,|;)",
        r"specialized in\s+(.+?)(\.|\n|$|,|;)",
        r"trained in\s+(.+?)(\.|\n|$|,|;)",
        r"certified in\s+(.+?)(\.|\n|$|,|;)",
        r"qualified in\s+(.+?)(\.|\n|$|,|;)",
        r"competent in\s+(.+?)(\.|\n|$|,|;)",
        r"worked (?:on|with)\s+(.+?)(\.|\n|$|,|;)",
        r"using\s+(.+?)(\.|\n|$|,|;)"
    ]
    
    # Collection of different bullet point styles
    bullet_patterns = [
        r"(?:^|\n)[ \t]*[•\-\*>➢✓✔◆◼►▪▸][ \t]*(.+?)(?:\n|$)",  # Unicode bullets
        r"(?:^|\n)[ \t]*[\d]+\.[ \t]*(.+?)(?:\n|$)",           # Numbered bullets
        r"(?:^|\n)[ \t]*[ivxIVX]+\.[ \t]*(.+?)(?:\n|$)",       # Roman numerals
        r"(?:^|\n)[ \t]*[a-zA-Z][\.)]+[ \t]*(.+?)(?:\n|$)"     # Alphabetic bullets
    ]
    
    # Common technical and professional skills
    technical_skills_patterns = [
        r'\b(?:python|java|javascript|html|css|sql|php|c\+\+|c#|typescript|ruby|swift|kotlin)\b',
        r'\b(?:react|angular|vue|node\.js|express|django|flask|spring|laravel|ruby on rails)\b',
        r'\b(?:git|github|gitlab|bitbucket|svn|mercurial|jira|confluence|trello|asana)\b',
        r'\b(?:aws|azure|google cloud|firebase|docker|kubernetes|jenkins|terraform|ansible)\b',
        r'\b(?:machine learning|artificial intelligence|data science|nlp|computer vision|deep learning)\b',
        r'\b(?:excel|word|powerpoint|ms office|access|outlook|microsoft 365|google workspace)\b',
        r'\b(?:networking|cybersecurity|penetration testing|ethical hacking|firewall|vpn|ssl|encryption)\b'
    ]
    
    candidate_skills = set()
    
    # Extract skills from indicator phrases
    for pattern in skill_indicators:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # The first group contains the skill
            skill_text = match[0].strip().lower()
            # Further split by common separators
            for skill in re.split(r',|;|and', skill_text):
                skill = skill.strip()
                if skill and 1 <= len(skill.split()) <= 5:
                    candidate_skills.add(skill)
    
    # Extract skills from bullet points
    for pattern in bullet_patterns:
        bullet_matches = re.findall(pattern, text, re.MULTILINE)
        for bullet in bullet_matches:
            bullet = bullet.strip().lower()
            # Skip if too long or too short
            if not bullet or len(bullet) < 2 or len(bullet.split()) > 6:
                continue
            # Skip if contains date patterns
            if re.search(r'\d{1,2}[-/]\w+[-/]\d{2,4}', bullet) or re.search(r'\b\d{4}\b', bullet):
                continue
            # Skip obvious section headers
            if re.search(r'^[A-Z\s]+$', bullet) or any(header_word in bullet.lower() for header_word in ["name", "date", "status", "nationality", "gender", "marital"]):
                continue
            # Only add if it's a potential skill
            if 1 < len(bullet.split()) <= 5 and not bullet.replace('.', '').isdigit():
                candidate_skills.add(bullet)
    
    # Extract technical skills using specific patterns
    for pattern in technical_skills_patterns:
        tech_matches = re.findall(pattern, text, re.IGNORECASE)
        for match in tech_matches:
            candidate_skills.add(match.lower())
    
    return candidate_skills
    
    # Split text into potential skill phrases
    # Split by common separators and conjunctions
    phrases = re.split(r'[,;]|\s+and\s+|\s+or\s+|\s+with\s+|\s+including\s+|\s+such\s+as\s+', text)
    
    extracted_skills = set()
    
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase or len(phrase) < 2:
            continue
            
        # Remove percentage indicators and numbers
        phrase = re.sub(r'\d+%', '', phrase)
        phrase = re.sub(r'\d+', '', phrase)
        phrase = re.sub(r'\s+', ' ', phrase).strip()
        
        # Skip if too long (more than 4 words)
        if len(phrase.split()) > 4:
            continue
            
        # Skip if in blacklist
        if phrase in skill_blacklist:
            continue
            
        # Check if phrase contains any skill keywords
        for skill_keyword in skill_keywords:
            if skill_keyword in phrase:
                # Extract the skill keyword, not the entire phrase
                extracted_skills.add(skill_keyword)
                break
        else:
            # If no keyword found, check if the phrase itself is a skill
            if phrase in skill_keywords:
                extracted_skills.add(phrase)
    
    # Additional processing: look for specific patterns
    # Look for "skills" patterns
    skill_patterns = [
        r'(\w+\s+skills?)',  # e.g., "communication skills", "problem solving skills"
        r'(good\s+\w+)',     # e.g., "good communication"
        r'(basic\s+\w+)',    # e.g., "basic computer"
        r'(\w+\s+ability)',  # e.g., "learning ability"
        r'(\w+\s+management)', # e.g., "time management"
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            match = match.strip()
            if 1 <= len(match.split()) <= 3 and match not in skill_blacklist:
                extracted_skills.add(match)
    
    # Clean up and return
    cleaned_skills = []
    for skill in extracted_skills:
        skill = skill.strip()
        if skill and len(skill) > 1 and skill not in skill_blacklist:
            # Capitalize first letter of each word
            skill = ' '.join(word.capitalize() for word in skill.split())
            cleaned_skills.append(skill)
    
    return sorted(list(set(cleaned_skills)))

def load_skill_synonyms():
    """
    Load skill synonyms from CSV file to enhance skill recognition.
    Returns a dictionary mapping synonyms to their canonical skill names.
    """
    import csv
    import os
    
    synonym_map = {}
    canonical_skills = {}
    
    try:
        synonyms_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skill_synonyms.csv")
        with open(synonyms_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    canonical_skill = row[0].strip().lower()
                    canonical_skills[canonical_skill] = canonical_skill
                    synonym_list = row[1].split(';')
                    for syn in synonym_list:
                        syn = syn.strip().lower()
                        if syn:
                            synonym_map[syn] = canonical_skill
    except Exception as e:
        print(f"[WARNING] Could not load skill synonyms: {e}")
    
    return synonym_map, canonical_skills

def extract_skills(text, name=None):
    """
    Master function for extracting skills from resume text.
    Uses multiple methods and combines them for comprehensive skill extraction.
    """
    # Load skill synonyms for enhanced recognition
    synonym_map, canonical_skills = load_skill_synonyms()
    
    # Get skills from all extraction methods
    block_skills = set(extract_skills_from_block(text, name))
    bullet_skills = set(extract_bullet_skills(text, name))
    short_skills = set(extract_short_skills_from_text(text, name))
    advanced_skills = set(extract_advanced_skills(text, name))
    
    # Combine all skills sources with priority
    if block_skills:
        # If we found block skills, prioritize them but add unique skills from other methods
        all_skills = block_skills.union(bullet_skills).union(advanced_skills)
    else:
        # Otherwise use all extraction methods
        all_skills = short_skills.union(bullet_skills).union(advanced_skills)
    
    print(f"[DEBUG] Skills before cleaning - Block: {len(block_skills)}, Bullet: {len(bullet_skills)}, " + 
          f"Short: {len(short_skills)}, Advanced: {len(advanced_skills)}, Total: {len(all_skills)}")
    
    # Final cleanup: remove Unicode characters and other artifacts
    cleaned_skills = []
    for skill in all_skills:
        # Remove Unicode characters and normalize
        cleaned_skill = unicodedata.normalize('NFKD', skill).encode('ASCII', 'ignore').decode('ASCII')
        cleaned_skill = cleaned_skill.strip()
        # Skip if it's just punctuation or too short
        if cleaned_skill and len(cleaned_skill) > 1 and not cleaned_skill.isspace():
            # Skip if it's just a single character or punctuation
            if len(cleaned_skill) > 1 and not cleaned_skill in ['•', '-', '*', ':', ';', ',', '.']:
                cleaned_skills.append(cleaned_skill)
    
    # --- GLOBAL FILTERS ---
    final_skills = []
    import re
    
    # Define location blacklist and other non-skill terms to exclude
    location_blacklist = [
        # States
        'maharashtra', 'gujarat', 'karnataka', 'tamil nadu', 'andhra pradesh', 'telangana', 
        'uttar pradesh', 'bihar', 'west bengal', 'rajasthan', 'odisha', 'kerala', 'jharkhand',
        'assam', 'punjab', 'haryana', 'uttarakhand', 'himachal pradesh', 'goa', 'arunachal pradesh',
        'meghalaya', 'manipur', 'nagaland', 'tripura', 'sikkim', 'mizoram',
        'jammu', 'kashmir', 'delhi', 'chandigarh', 'andaman', 'nicobar', 'puducherry',
        'dadra', 'nagar haveli', 'daman', 'diu', 'lakshadweep', 'ladakh',
        # Major cities
        'mumbai', 'bangalore', 'bengaluru', 'hyderabad', 'chennai', 'kolkata', 'delhi',
        'pune', 'ahmedabad', 'jaipur', 'surat', 'lucknow', 'kanpur', 'nagpur',
        'indore', 'thane', 'bhopal', 'visakhapatnam', 'patna', 'vadodara', 'ludhiana',
        'kochi', 'agra', 'madurai', 'nashik', 'varanasi', 'amritsar', 'allahabad', 'prayagraj',
        'ranchi', 'guwahati', 'chandigarh', 'gwalior', 'vijayawada', 'jodhpur', 'raipur',
        'kota', 'mysore', 'bareilly', 'tiruppur', 'gurgaon', 'gurugram', 'jalandhar', 'jamnagar',
        'dehradun', 'mangalore', 'mangaluru', 'jammu', 'trivandrum', 'thiruvananthapuram',
        # Specific locations mentioned
        'talasari', 'palghar', 'arvi', 'ahemedabad', 'anantapur', 'ananthapuramu', 'amarapur',
        # Location-related terms
        'village', 'district', 'state', 'city', 'town', 'pincode', 'postal code', 'zip code', 
        'address', 'locality', 'region', 'zone', 'area', 'sector', 'block', 'street', 'road',
        'highway', 'lane', 'avenue', 'colony', 'society', 'apartment', 'flat', 'house', 'residence',
        'location', 'headquarter', 'branch', 'office', 'pin', 'pin-code', 'near', 'opposite'
    ]
    
    # Common English words that should not be considered skills
    common_english_words = [
        # Articles and prepositions
        'with', 'and', 'the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'to', 'from', 'of', 
        'as', 'or', 'but', 'if', 'then', 'than', 'about', 'above', 'across', 'after', 'against',
        'along', 'among', 'around', 'before', 'behind', 'below', 'beneath', 'beside', 'besides',
        'between', 'beyond', 'during', 'except', 'inside', 'into', 'like', 'near', 'off',
        'onto', 'outside', 'over', 'past', 'since', 'through', 'throughout', 'till', 'toward',
        'under', 'underneath', 'until', 'unto', 'upon', 'up', 'via', 'within', 'without',
        
        # Auxiliary verbs and common verbs
        'have', 'has', 'had', 'having', 'be', 'being', 'been', 'am', 'is', 'are', 'was', 'were',
        'do', 'does', 'did', 'done', 'doing', 'can', 'could', 'will', 'would', 'shall', 'should',
        'may', 'might', 'must', 'get', 'got', 'getting', 'go', 'goes', 'going', 'gone', 'went',
        'come', 'comes', 'coming', 'came', 'take', 'takes', 'taking', 'took', 'taken',
        'make', 'makes', 'making', 'made', 'give', 'gives', 'giving', 'gave', 'given',
        
        # Pronouns and determiners
        'this', 'that', 'these', 'those', 'it', 'its', 'his', 'her', 'their', 'our', 'my', 'your',
        'i', 'we', 'they', 'you', 'he', 'she', 'them', 'us', 'me', 'him', 'one', 'oneself', 
        'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves',
        'each', 'every', 'some', 'any', 'all', 'most', 'more', 'much', 'many', 'little', 'few',
        'less', 'least', 'several', 'enough', 'both', 'either', 'neither', 'whose', 'which', 'what',
        
        # Adverbs and modifiers
        'very', 'extremely', 'quite', 'rather', 'too', 'so', 'just', 'only', 'even', 'still',
        'almost', 'also', 'always', 'often', 'sometimes', 'rarely', 'never', 'ever', 'soon',
        'now', 'then', 'when', 'where', 'how', 'why', 'well', 'quickly', 'slowly', 'really',
        'actually', 'basically', 'certainly', 'definitely', 'eventually', 'exactly', 'finally',
        'hopefully', 'literally', 'naturally', 'obviously', 'possibly', 'probably', 'simply',
        
        # Common phrases
        'with a', 'in a', 'by a', 'for a', 'on a', 'of a', 'to a', 'as a', 'at a',
        'with the', 'in the', 'by the', 'for the', 'on the', 'of the', 'to the', 'as the', 'at the'
    ]
    
    # General terms that are not skills but might get extracted
    non_skill_terms = [
        # Educational institutions
        'college', 'university', 'school', 'institute', 'academy', 'department', 'campus',
        'polytechnic', 'community college', 'technical institute', 'board of', 'faculty',
        # Months and time periods
        'january', 'february', 'march', 'april', 'may', 'june', 'july', 
        'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'semester', 'year', 'session', 'academic year', 'quarter', 'term',
        # Academic fields (not specific skills)
        'commerce', 'arts', 'science', 'engineering', 'management', 'humanities',
        'economics', 'mathematics', 'history', 'geography', 'biology', 'chemistry', 'physics',
        'social science', 'political science', 'environmental science',
        # Time-related
        'present', 'current', 'ongoing', 'today', 'till date', 'till now',
        'duration', 'period', 'timeline', 'from', 'to', 'since', 'until',
        # Education levels and qualifications
        'bachelor', 'master', 'phd', 'doctor', 'diploma', 'degree', 'certificate',
        'certification', 'graduate', 'postgraduate', 'undergraduate', 'doctoral',
        # Resume sections
        'experience', 'education', 'certification', 'achievements', 'projects', 'skills',
        'interests', 'hobbies', 'languages', 'personal details', 'profile', 'summary',
        'objective', 'professional summary', 'career objective', 'references', 'declaration',
        # Common resume phrases
        'student', 'graduate', 'undergraduate', 'completed', 'achievement', 'accomplishment',
        'responsibility', 'team', 'environment', 'problem', 'solving', 'scholar', 'alumnus', 'alumni',
        'oriented', 'attitude', 'exposure', 'foundation', 'career', 'professional', 'resume', 'cv',
        'curriculum vitae', 'background', 'information', 'detail', 'qualification', 'portfolio',
        'major', 'minor', 'specialization', 'concentration', 'stream', 'branch',
        # Education metrics and markers
        'cgpa', 'percentage', 'percentile', 'rank', 'grade', 'score', 'gpa', 'marks', 'aggregate',
        '10th', '12th', 'ssc', 'hsc', 'bachelors', 'masters', 'doctorate', 'intermediate',
        'bachelor of', 'master of', 'pursuing', 'passed', 'completed', 'appeared',
        'graduated', 'class of', 'batch', 'batch of', 'year of passing', 'expected graduation',
        # Address and location fragments that might be parsed as skills
        'p.o', 'p.s', 'po', 'ps', 'district', 'state', 'pin', 'pincode', 'postal',
        'lilong', 'manipur', 'karong', 'suptu', 'ushoipokpi', 'tamyai',
        # Common address abbreviations
        'o -', 's -', 'p -', 'district -', 'state -', 'pin -'
    ]
    
    for skill in cleaned_skills:
        # Filter out numeric strings and pincodes
        if re.fullmatch(r'\d{6}', skill) or re.search(r'\b\d{6}\b', skill):
            continue
        if re.fullmatch(r'\d+\s*(Skills|Designer Skills)', skill, re.I) or re.search(r'\b\d{6}\b\s*(Skills|Designer Skills)', skill, re.I):
            continue
        # Filter out strings that are too long (likely sentences, not skills)
        if len(skill.split()) > 6:
            continue
        # Filter out strings that are likely dates
        if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', skill) or re.search(r'\b20\d{2}\b', skill):
            continue
        # Filter out strings with odd special characters (likely parsing artifacts)
        if re.search(r'[\\\/\[\]\{\}\(\)\<\>]', skill):
            continue
        # Filter out section separators with underscores, dashes, etc.
        if re.search(r'^[_\-=\.]{5,}', skill) or re.search(r'[_\-=\.]{5,}$', skill):
            continue
        # Filter out strings that contain many underscores or dashes (likely section separators)
        if skill.count('_') > 4 or skill.count('-') > 4:
            continue
        # Filter out section headings that end with "Skills" and have underscores/dashes
        if re.search(r'_{3,}.*\bskills?\b', skill.lower()) or re.search(r'\bskills?\b.*_{3,}', skill.lower()):
            continue
        
        # Filter out locations and other non-skill terms
        skill_lower = skill.lower()
        
        # Skip if skill is a single common English word or phrase
        if skill_lower in common_english_words:
            continue
            
        # Skip if the skill contains location names
        if any(location in skill_lower for location in location_blacklist):
            continue
            
        # Skip if the skill contains non-skill terms
        if any(term in skill_lower for term in non_skill_terms):
            continue
            
        # Additional filtering for address components that might slip through
        if re.match(r'^[a-z]\s*-\s*[a-z]+$', skill_lower):  # Matches "O - Lilong", "S - Lilong"
            continue
            
        # Skip field names that are too generic (like "pharmacy fields")
        if 'fields' in skill_lower and len(skill_lower.split()) <= 2:
            continue
            
        # Skip single letters or very short abbreviations
        if len(skill_lower) <= 2 and not skill_lower.isalpha():
            continue
            
        # Skip very short skills (likely not valid)
        if len(skill_lower) <= 2:
            continue
            
        # Skip skills that are just articles, prepositions or common words
        if len(skill_lower.split()) == 1 and skill_lower in common_english_words:
            continue
            
        # Skip skills that start with prepositions or articles
        if skill_lower.split()[0] in ['with', 'the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'to', 'as']:
            continue
            
        # Skip education metrics that include numbers (like "71 cgpa")
        if re.search(r'\d+(\.\d+)?\s*(cgpa|gpa|percentage|percentile|rank|grade|score|marks)', skill_lower):
            continue
            
        # Skip any string that starts with a number and contains grade-related terms
        if re.match(r'^\d+', skill_lower) and any(term in skill_lower for term in ['cgpa', 'gpa', 'percentage', 'grade']):
            continue
            
        # Skip standalone numbers or very short number-based strings (likely scores, not skills)
        if re.match(r'^\d+(\.\d+)?$', skill_lower) or re.match(r'^\d+[a-z]$', skill_lower):
            continue
        
        # Skip strings that look like ratings (e.g., "5/10")
        if re.match(r'^\d+\s*/\s*\d+$', skill_lower):
            continue
            
        # Skip terms that are likely dates or time periods
        if re.search(r'^\d{4}\s*-\s*\d{4}$|^\d{4}\s*-\s*(present|current|now)$', skill_lower):
            continue
            
        # Skip if the skill contains common honorifics or personal identifiers
        if any(title in skill_lower for title in ['mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'sri', 'smt', 'shri']):
            continue
            
        # Skip if the skill appears to be a person's name (first name followed by last name)
        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', skill):
            continue
            
        # Skip terms that look like identifiers or registration numbers
        if re.search(r'id:|reg:|no:|number:|registration:|enrollment:', skill_lower):
            continue
            
        final_skills.append(skill)
    
    print(f"[DEBUG] Skills after cleaning: {len(final_skills)}")
    
    # Normalize skills using synonyms
    normalized_skills = set()
    
    # Final filter of common words and non-skills
    common_stopwords = ['with', 'and', 'the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'to', 'from', 'of']
    
    for skill in final_skills:
        skill_lower = skill.lower()
        
        # Skip standalone common words
        if skill_lower in common_stopwords:
            continue
            
        # Skip phrases that are just common words combined
        if all(word in common_stopwords for word in skill_lower.split()):
            continue
        
        # Skip phrases like "with a" or "in the" which are just preposition + article
        if len(skill_lower.split()) == 2 and skill_lower.split()[0] in common_stopwords and skill_lower.split()[1] in common_stopwords:
            continue
        
        # Check if this is a known canonical skill or synonym
        if skill_lower in canonical_skills:
            # Use title case for consistency
            normalized_skills.add(skill_lower.title())
        elif skill_lower in synonym_map:
            # If it's a synonym, use the canonical form
            normalized_skills.add(synonym_map[skill_lower].title())
        else:
            # Otherwise keep the skill as is (with title case)
            normalized_skills.add(' '.join(word.capitalize() for word in skill_lower.split()))
    
    print(f"[DEBUG] Final normalized skills: {len(normalized_skills)}")
    return sorted(list(normalized_skills))

def is_valid_resume(text, is_ocr=False):
    """
    Heuristically check if the text appears to be from a resume.
    Stricter validation for OCR text to reject non-resume documents like grade cards and certificates.
    """
    resume_keywords = [
        "objective", "summary", "skills", "education", "experience", "internship",
        "project", "certification", "language", "profile", "career", "responsibilities",
        "work", "training", "qualification", "achievement", "hobbies", "personal"
    ]
    non_resume_keywords = [
        "grade card", "marksheet", "result", "seat number", "subject code", "examination",
        "internal assessment", "total marks", "controller of examination", "semester", "credit",
        "certificate of training", "certificate no", "date of certification", "for certificate authentication"
    ]
    certificate_indicators = [
        "certificate of", "training on", "scored % marks", "final assessment", "top performer"
    ]

    lower_text = text.lower().strip()
    resume_score = sum(1 for word in resume_keywords if word in lower_text)
    non_resume_score = sum(2 for word in non_resume_keywords if word in lower_text)  # Increased weight
    certificate_score = sum(1 for indicator in certificate_indicators if indicator in lower_text)

    print(f"DEBUG: Resume score: {resume_score}, Non-resume score: {non_resume_score}, "
          f"Certificate score: {certificate_score}, OCR used: {is_ocr}, Word count: {len(lower_text.split())}")

    # For OCR text, require at least two resume keywords and fewer non-resume/certificate indicators
    if is_ocr:
        return (resume_score >= 2 and resume_score > non_resume_score and 
                resume_score > certificate_score and len(lower_text.split()) > 50)
    # For non-OCR, require at least two resume keywords and dominance over non-resume keywords
    else:
        return resume_score >= 2 and resume_score > non_resume_score

# === Project & Internship Checker ===
def check_projects_and_internships(text):
    text = text.lower()
    return {
        'projects': "project" in text,
        'internships': "internship" in text or "intern" in text
    }

# === Work Experience Checker ===
def check_work_experience(text):
    return "experience" in text.lower() or "worked at" in text.lower()

# === Language Checker ===
def check_languages(text):
    return "language" in text.lower() or "languages known" in text.lower()

# === Certification Checker ===
def check_certification(text):
    return "certificate" in text.lower() or "certification" in text.lower()

# === Resume Rating ===
def calculate_rating(personal, edu, proj_int, certification, work_exp, skills, langs):
    score = 0
    if all([personal['name'], personal['email'], personal['phone']]):
        score += RATING_WEIGHTS['personal_info']
    if edu['pg'] or edu['dr']: score += 10
    if edu['g']: score += 10
    if edu['ug'] or edu['hsc']: score += 10
    if edu['ssc']: score += 10
    if proj_int['projects']: score += RATING_WEIGHTS['projects']
    if proj_int['internships']: score += RATING_WEIGHTS['internships']
    if work_exp: score += RATING_WEIGHTS['work_experience']
    if skills: score += RATING_WEIGHTS['skills']
    if langs: score += RATING_WEIGHTS['languages']
    return round(score, 2)

# === Education Summary ===
def education_summary(edu):
    summary_parts = []
    if edu['pg']:
        summary_parts.append("Post Graduate: Yes")
    if edu['dr']:
        summary_parts.append("Doctorate: Yes")
    if not edu['pg'] and not edu['dr']:
        summary_parts.append("Post Graduate: No")
        summary_parts.append("Doctorate: No")
    summary_parts.append(f"Graduate (Bachelor's): {'Yes' if edu['g'] else 'No'}")
    summary_parts.append(f"UG (Diploma/ITI): {'Yes' if edu['ug'] else 'No'}")
    summary_parts.append(f"HSC: {'Yes' if edu['hsc'] else 'No'}")
    summary_parts.append(f"SSC: {'Yes' if edu['ssc'] else 'No'}")
    return ', '.join(summary_parts)

# === Improvement Suggestion Generator ===
def generate_improvement_suggestions(personal, edu, proj_int, certification, work_exp, skills, langs):
    suggestions = []
    if not all([personal['name'], personal['email'], personal['phone']]):
        missing = []
        if not personal['name']: missing.append("Name")
        if not personal['email']: missing.append("Email")
        if not personal['phone']: missing.append("Phone")
        suggestions.append(f"Include missing personal information: {', '.join(missing)}.")
    if not (edu['pg'] or edu['dr']):
        suggestions.append("Consider adding or pursuing Postgraduate (Master's, PGD) or Doctorate education.")
    if not edu['g']:
        suggestions.append("Add your Bachelor's degree qualification.")
    if not (edu['ug'] or edu['hsc']):
        suggestions.append("Include Diploma/ITI or Higher Secondary (12th) education.")
    if not edu['ssc']:
        suggestions.append("Mention your Secondary School (10th) education.")
    if not proj_int['projects']:
        suggestions.append("Add academic or personal projects.")
    if not proj_int['internships']:
        suggestions.append("Mention any internships or trainee roles you've done.")
    if not certification:
        suggestions.append("List certifications or training programs.")
    if not work_exp:
        suggestions.append("Include work experience or volunteering.")
    if not skills:
        suggestions.append("List your technical or domain-specific skills.")
    if not langs:
        suggestions.append("Mention the languages you're fluent in.")
    return suggestions

# === AI Skill Gap Suggestion ===
def generate_ai_skill_suggestions(job_title, detected_skills):
    """Generates skill suggestions by comparing detected skills to job requirements in MongoDB."""
    expected_skills = set()
    if jobs_collection is None: # <-- FIX
        return []  
    try:
        job = jobs_collection.find_one({"job_title": job_title}, {"keywords": 1, "_id": 0})
        if job and job.get("keywords"):
            expected_skills.update([kw.strip().lower() for kw in job["keywords"].split(",") if kw.strip()])
    except Exception as e:
        print(f"❌ Error fetching AI skill suggestions from MongoDB: {e}")

    # Ensure detected_skills are all lowercase for accurate comparison
    detected_skills_lower = {skill.lower() for skill in detected_skills}
    return list(expected_skills - detected_skills_lower) 

# === Resume Score Chart ===
def display_resume_score(rating):
    pass  # Disabled as per original update

# === Main Analyzer ===
major_indian_locations = {
    "Thane", "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad", "Chennai", "Kolkata", "Surat",
    "Pune", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Visakhapatnam", "Boisar","Hanamkonda","Anantapur","Wagholi"
}

# Institution keywords for name/section filtering
institution_keywords = [
    'college', 'university', 'polytechnic', 'foundation', 'school', 'institute', 'academy', 'faculty',
    'company', 'organization', 'department', 'junior college'
]

# Blacklist for name/section filtering
blacklist = set([
    'resume', 'curriculum vitae', 'cv', 'profile', 'summary', 'career objective', 'contact', 'email', 'mobile', 'phone',
    'address', 'education', 'academic', 'skills', 'experience', 'projects', 'certifications', 'languages', 'interests',
    'project', 'predictor', 'app', 'system', 'generator', 'website', 'application', 'report', 'analysis', 'certificate',
    'internship', 'achievement', 'interest', 'objective', 'training', 'course', 'workshop', 'seminar', 'conference',
    'award', 'publication', 'activity', 'volunteering', 'team', 'leadership', 'hobby', 'reference'
])

def predict_job_location(personal_location, resume_text):
    # Extract locations mentioned in education section or resume text excluding personal_location
    edu_locations = []
    for loc in major_indian_locations:
        if loc.lower() in resume_text.lower() and loc.lower() != (personal_location or "").lower():
            edu_locations.append(loc)
    # Determine the most relevant location based on education locations only
    if edu_locations:
        # Choose the most frequent or first found location in education locations
        predicted_location = edu_locations[0]
    else:
        predicted_location = None
    return predicted_location

def is_paragraph_skills_section(text):
    """
    Detect if the SKILLS section is a paragraph/long format (not bullet/line format).
    Returns True if the SKILLS section is likely a paragraph.
    """
    import re
    lines = text.splitlines()
    skill_section_lines = []
    capture = False
    for line in lines:
        if re.match(r"^\s*skills?\s*$", line.strip(), re.I):
            capture = True
            continue
        if capture:
            line_stripped = line.strip()
            # Stop at next section header or empty line
            if not line_stripped or line_stripped.upper() in [
                'CONTACT', 'EDUCATION', 'EXPERIENCE', 'PROJECTS', 'CERTIFICATIONS', 'LANGUAGES',
                'VOLUNTEER', 'PROFILE', 'SUMMARY', 'OBJECTIVE', 'WORK EXPERIENCE', 'ACADEMIC',
                'ACHIEVEMENTS', 'AWARDS', 'HOBBIES', 'PERSONAL INFORMATION', 'CONTACT INFORMATION',
                'PERSONAL PROFILE', 'E D U C A T I O N', 'P R O J E C T S', 'C E R T I F I C A T I O N S',
                'L A N G U A G E S', 'P R O F I L E', 'S U M M A R Y', 'DECLLERATION', 'DECLARATION',
                 'ACADMIC QUALIFICATION', 'CAREEROBJECTIVE', 'INTERESTS', 'INTERNSHIP', 'PROJECTS', 'CERTIFICATIONS', 'DECLARATION']:
                break
            skill_section_lines.append(line_stripped)
    
    print("\n[DEBUG] SKILLS SECTION LINES CAPTURED:")
    for idx, l in enumerate(skill_section_lines):
        print(f"  {idx+1}: {repr(l)}")
    
    if not skill_section_lines:
        print("DEBUG: No skills section lines found")
        return False
    
    for line in skill_section_lines:
        if (':' in line or '|' in line or line.startswith('•') or line.startswith('-') or line.startswith('*') or len(line) < 30):
            print(f"DEBUG: Detected well-formatted line: {line}")
            print("DEBUG: Detected as well-formatted skills section")
            return False
    print("DEBUG: Detected as paragraph-style skills section")
    return True

def analyze_resume(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    resume_text = ""
    is_ocr_used = False
    try:
        if file_path.endswith(".pdf"):
            resume_text = extract_text_from_pdf(file_path)
            if not resume_text.strip():
                print("⚠️ PDF appears to be image-based. Trying OCR...")
                resume_text = extract_text_from_image_pdf(file_path)
                is_ocr_used = True
                if isinstance(resume_text, str) and resume_text.startswith("Error"):
                    print(f"❌ {resume_text}")
                    return None
        elif file_path.endswith(".docx"):
            resume_text = extract_text_from_docx(file_path)
        else:
            print("❌ Unsupported file format. Only .pdf and .docx are supported.")
            return None
    except Exception as e:
        print(f"❗ Text extraction error: {e}")
        return None

    if not resume_text.strip():
        print("❌ No text extracted. Check file, Tesseract, and poppler-utils.")
        return None

    print("📄 Extracted text sample:", resume_text[:200])

    if not is_valid_resume(resume_text, is_ocr=is_ocr_used):
        print("❌ This does not appear to be a valid resume. It may be a grade card or certificate.")
        return None

    personal_info = extract_personal_details(resume_text)
    print(f"DEBUG: Resume text length: {len(resume_text)}")
    print(f"DEBUG: Resume text sample (first 500 chars): {resume_text[:500]}")
    print(f"DEBUG: Looking for 'S K I L L S' in text: {'S K I L L S' in resume_text}")

    # --- Adaptive skills extraction ---
    if is_paragraph_skills_section(resume_text):
        print("[INFO] Detected paragraph-style skills section. Using only short skills extraction.")
        skills = extract_short_skills_from_text(resume_text, name=personal_info.get("name"))
    else:
        skills = extract_skills(resume_text, name=personal_info.get("name"))
    print(f"DEBUG: Extracted skills: {skills}")
    print(f"DEBUG: Number of skills: {len(skills)}")
    print(f"DEBUG: Skills type: {type(skills)}")

    edu = check_education(resume_text)
    proj_int = check_projects_and_internships(resume_text)
    work_exp = check_work_experience(resume_text)
    langs = check_languages(resume_text)
    certification = check_certification(resume_text)

    job_field_matches = predict_job_field(resume_text)
    # --- Technical job evidence filter ---
    job_field_evidence = {
        "android developer": ["android", "java", "kotlin", "android studio", "xml", "sdk", "app development"],
        "java developer": ["java", "spring", "hibernate", "jvm"],
        "python developer": ["python", "django", "flask"],
        "web developer": ["html", "css", "javascript", "react", "angular", "vue", "web development"],
        "software developer": ["software development", "programming", "coding", "c++", "c#", "python", "java", "software engineer"],
        "full stack developer": ["full stack", "frontend", "backend", "node.js", "react", "angular", "vue", "express", "mongodb", "sql"],
        "data scientist": ["data science", "machine learning", "deep learning", "python", "r", "pandas", "numpy", "scikit-learn", "tensorflow", "keras"],
        "machine learning engineer": ["machine learning", "deep learning", "python", "tensorflow", "keras", "pytorch", "scikit-learn"],
        # Add more technical roles as needed
    }
    # Only use actual extracted skills (not ai_skill_suggestions) and resume_text for evidence
    actual_skills = skills  # Make explicit for clarity
    def has_evidence(job_title, resume_text, actual_skills):
        keywords = job_field_evidence.get(job_title.lower())
        if not keywords:
            return True  # No evidence required for non-technical jobs
        text = resume_text.lower()
        skill_text = " ".join(actual_skills).lower() if actual_skills else ""
        for kw in keywords:
            if kw in text or kw in skill_text:
                return True
        return False
    # Filter job_field_matches for technical jobs
    filtered_matches = []
    for entry in job_field_matches:
        title = entry.get("title", "").lower()
        if has_evidence(title, resume_text, actual_skills):
            filtered_matches.append(entry)
    if filtered_matches:
        job_field_matches = filtered_matches
        top_job_entry = job_field_matches[0]
        job_field = top_job_entry["title"]
    else:
        # If all technical jobs are filtered out, select the first non-technical job from the original predictions
        for entry in job_field_matches:
            title = entry.get("title", "").lower()
            if title not in job_field_evidence:
                job_field_matches = [entry] + [e for e in job_field_matches if e != entry]
                top_job_entry = entry
                job_field = top_job_entry["title"]
                break
        else:
            # If all are technical, fallback to the original top job
            top_job_entry = job_field_matches[0]
            job_field = top_job_entry["title"]

    if job_field_matches:
        top_job_entry = job_field_matches[0]
        job_field = top_job_entry["title"]
        # Build recommended_courses as a dict for top 3 job fields
        recommended_courses_dict = {}
        for entry in job_field_matches[:3]:
            title = entry.get("title", "Unknown")
            recommended_courses_dict[title] = get_recommended_courses(title)
    else:
        top_job_entry = {"title": "No such job field found on server.", "confidence": 0.0}
        job_field = "No such job field found on server."
        recommended_courses_dict = {}
    rating = calculate_rating(personal_info, edu, proj_int, certification, work_exp, bool(skills), langs)
    suggestions = generate_improvement_suggestions(personal_info, edu, proj_int, certification, work_exp, skills, langs)
    ai_skill_suggestions = generate_ai_skill_suggestions(job_field, skills)
    preferred_location = personal_info.get('location')

    # Use job_field_matches for output (after filtering)
    # Filter skills again for job field prediction to ensure no non-skill items are included
    import re
    filtered_job_skills = []
    
    # Common English phrases, prepositions, articles, conjunctions to filter out
    common_english_words = [
        r'^(with|and|the|a|an|in|on|at|by|for|to|from|of|as|or|but|if|then|than)$',
        r'^(with a|in a|by a|for a|to a|from a|of a|as a)$',
        r'^(have|has|had|having|be|being|been|am|is|are|was|were)$',
        r'^(this|that|these|those|it|its|his|her|their|our|my|your)$',
        r'^(can|could|will|would|should|may|might|must)$'
    ]
    
    non_skill_patterns = [
        # Educational institutions
        r'^.*\b(college|university|institute|school)\b.*$',
        # Months
        r'^.*\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b.*$',
        # Locations
        r'^.*\b(maharashtra|gujarat|mumbai|delhi|bangalore|pune|hyderabad|chennai)\b.*$',
        r'^.*\b(talasari|palghar|district|state|country|region|city|town|village)\b.*$',
        # Years and dates
        r'^.*\b(20\d{2})\b.*$',  
        # Academic subjects (not technical skills)
        r'^.*\b(commerce|arts|science)\b.*$',
        # Common phrases from resumes that aren't skills
        r'^.*\b(completed|achievement|responsibility|team|environment|problem|solving)\b.*$',
        r'^.*\b(student|graduate|undergraduate|certification|course|training)\b.*$',
        # Education metrics and grades
        r'^.*\b(cgpa|percentage|percentile|rank|grade|score|gpa)\b.*$',
        r'^.*\b(\d+\.\d+\s*cgpa|\d+\.\d+\s*gpa|\d+\s*%)\b.*$',
        r'^\d+(\.\d+)?\s*(cgpa|gpa)$',
        # Education institutions & qualifiers
        r'^.*\b(bachelors?|masters?|phd|doctorate|diploma|certificate)\b.*$'
    ]
    
    for skill in skills:
        # Skip if the skill is a common English word/phrase
        if any(re.match(pattern, skill.lower()) for pattern in common_english_words):
            continue
            
        # Skip if matches any non-skill pattern
        if any(re.search(pattern, skill.lower()) for pattern in non_skill_patterns):
            continue
            
        # Skip if it's very short (likely not a skill)
        if len(skill) <= 2:
            continue
            
        # Skip anything that looks like a grade/score with numerical pattern
        if re.search(r'^\d+(\.\d+)?\s*', skill):
            # Check if it contains education metrics
            if re.search(r'(cgpa|gpa|percentage|percentile|score|grade|rank)', skill.lower()):
                continue
                
        # Skip entries that are just numbers or contain digits with educational terms
        if re.search(r'^\d+(\.\d+)?$', skill) or re.search(r'^\d+(\.\d+)?\s*(cgpa|gpa|percentage|%)', skill.lower()):
            continue
            
        filtered_job_skills.append(skill)
    
    output_data = OrderedDict([
        ("raw_text", resume_text),
        ("personal_info", personal_info),
        ("resume_score", rating),
        ("education_summary", education_summary(edu)),
        ("improvement_suggestions", suggestions),
        ("ai_skill_suggestions", ai_skill_suggestions),
        ("predicted_job_field", OrderedDict([
            ("top_matches", job_field_matches[:3]),  # Use filtered list
            ("job_title", job_field),
            ("skills", filtered_job_skills),  # Use filtered skills list
            ("preferred_location", preferred_location),
            ("recommended_courses", recommended_courses_dict)
        ]))
    ])

    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/resume_analysis_report_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print("\n✅ Resume analyzed successfully!")
    print("Predicted Job Field:", job_field)
    print("Resume Score:", rating, "/ 100")
    print("Output saved to:", output_path)
    return output_data

# === Entry Point ===
if __name__ == "__main__":
    print("=== Resume Analyzer ===")
    while True:
        file_path = input("Enter resume file path (.pdf/.docx) or 'exit': ").strip()
        if file_path.lower() == 'exit':
            print("Exiting...")
            break
        result = analyze_resume(file_path)
        if result is None:
            print("❌ Analysis failed. Try again with a valid file.\n")
        else:
            print("\n📊 Ready to analyze another resume?\n")
    print("=== Resume Analyzer ===")

def extract_first_name_from_raw(email_prefix, raw_text):
    email_prefix_clean = re.sub(r'\d+', '', email_prefix)
    # Try to get the first part (before any separator)
    for sep in ['.', '_', '-']:
        if sep in email_prefix_clean:
            first_part = email_prefix_clean.split(sep)[0]
            break
    else:
        first_part = email_prefix_clean[:8]
    # Search for a word in the raw text that starts with this part (case-insensitive)
    words = re.findall(r'\b\w+\b', raw_text)
    for word in words:
        if word.lower().startswith(first_part.lower()) and len(word) > 1:
            return re.sub(r'\d+', '', word).capitalize()
    return re.sub(r'\d+', '', first_part).capitalize()

def name_in_raw_text(candidate, raw_text):
    if not candidate:
        return False
    # Remove extra spaces and compare case-insensitive
    candidate_clean = ' '.join(candidate.split()).lower()
    raw_text_clean = ' '.join(raw_text.split()).lower()
    # Look for the candidate as a whole word sequence in the raw text
    pattern = re.compile(r'\b' + re.escape(candidate_clean) + r'\b', re.IGNORECASE)
    return bool(pattern.search(raw_text_clean))

def is_valid_name_line(line):
    line = line.strip()
    # Skip lines with email, numbers, or special characters
    if '@' in line or any(char.isdigit() for char in line):
        return False
    # Skip lines with phone-like patterns
    if re.search(r'\b\d{10}\b', line) or re.search(r'\+\d{1,3}', line):
        return False
    # At least 2 words, all alphabetic
    words = line.split()
    if len(words) >= 2 and all(w.isalpha() for w in words):
        return True
    return False

def extract_name_first_line_priority(raw_text):
    """
    For resumes where the first line looks like a name (2+ words, all alphabetic), always use the first line as the name.
    Returns the name if found, else None.
    """
    lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
    if lines:
        first_line = lines[0]
        words = first_line.split()
        # Section headers to avoid as names
        section_headers = [
            "local address", "date of birth", "mobile", "email", "phone", "major", "minor", "address",
            "declaration", "signature", "place", "date", "nationality", "achievements", "skills", "education",
            "projects", "internship", "languages", "courses", "interests", "profile", "summary", "professional summary",
            "fresher", "intern", "trainee", "student"
        ]
        if len(words) >= 2 and all(w.isalpha() for w in words):
            # Check if the first line is a section header (case-insensitive, ignore extra spaces)
            first_line_clean = ' '.join(first_line.lower().split()).strip()
            if first_line_clean not in section_headers:
                return first_line
    return None

# Example usage for resumes like 'Mayur Patil' (not in main pipeline):
# raw_text = ...  # The raw resume text
# name = extract_name_first_line_priority(raw_text)
# if name:
#     print(f"First-line-priority extracted name: {name}")
# else:
#     # Fallback to your main extraction logic
#     ...

def search_for_actual_name_in_text(text):
    """
    Search for actual name patterns in the resume text.
    Looks for lines that contain name-like patterns, with robust email prefix verification.
    """
    import re

    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    # Blacklist of non-name words (expand as needed)
    non_name_words = {
        'objective', 'summary', 'profile', 'resume', 'curriculum', 'vitae', 'cv', 'biodata',
        'education', 'experience', 'skills', 'projects', 'internships', 'certifications',
        'achievements', 'personal', 'details', 'contact', 'email', 'phone', 'mobile',
        'address', 'date', 'birth', 'nationality', 'languages', 'interests', 'hobbies',
        'references', 'declaration', 'signature', 'place', 'major', 'minor', 'courses',
        'statement', 'purpose', 'career', 'goal', 'ambition', 'aspiration', 'target',
        # Add more as needed
    }

    # Extract email and email prefix
    email_match = re.search(r'[^\s]+@[\w\-]+(?:\.[\w\-]+)+', text)
    email_prefix = None
    email_parts = []
    if email_match:
        email = email_match.group()
        email_prefix = re.sub(r'\d+', '', email.split('@')[0]).lower()
        # Try to extract likely first and last name from prefix
        # If prefix is a single word, try to find all substrings of length >= 4
        substrings = set()
        if len(email_prefix) > 7 and ('.' not in email_prefix and '_' not in email_prefix and '-' not in email_prefix):
            # Heuristic: try all substrings of length 4-8
            for l in range(4, min(9, len(email_prefix))):
                for i in range(len(email_prefix)-l+1):
                    substrings.add(email_prefix[i:i+l])
            email_parts = list(substrings)
        else:
            # Use split by common separators
            email_parts = [p for p in re.split(r'[._-]', email_prefix) if len(p) > 1]

    # 1. Strict: Look for 2–4 word lines, all words start with uppercase, all alphabetic, not in blacklist
    for line in lines:
        words = line.split()
        if not (2 <= len(words) <= 4):
            continue
        if any(word.lower() in non_name_words for word in words):
            continue
        if any(char.isdigit() for char in line) or '@' in line:
            continue
        if not all(word.isalpha() and word[0].isupper() for word in words):
            continue

        candidate = ' '.join(word.title() for word in words)
        candidate_lower = candidate.lower()
        print(f"DEBUG: Considering candidate: {candidate}")

        # If email prefix is a concatenated name, require all likely parts to be present in candidate
        if email_prefix and len(email_prefix) > 7 and email_parts:
            # For 'shivanichavan', check if both 'shivani' and 'chavan' are in the candidate
            likely_first = email_prefix[:len(email_prefix)//2]
            likely_last = email_prefix[len(email_prefix)//2:]
            if (likely_first in candidate_lower and likely_last in candidate_lower):
                print(f"DEBUG: Matched both likely_first '{likely_first}' and likely_last '{likely_last}' in candidate.")
                return candidate
            # Or, check if at least two different substrings are present
            found_parts = [part for part in email_parts if part in candidate_lower]
            if len(found_parts) >= 2:
                print(f"DEBUG: Matched substrings {found_parts} in candidate.")
                return candidate
        else:
            return candidate

    # 2. Fallback: More flexible, but still checks for 2–4 words, all alphabetic, not in blacklist
    for line in lines:
        words = line.split()
        if not (2 <= len(words) <= 4):
            continue
        if any(word.lower() in non_name_words for word in words):
            continue
        if any(char.isdigit() for char in line) or '@' in line:
            continue
        if not all(word.isalpha() for word in words):
            continue

        candidate = ' '.join(word.title() for word in words)
        candidate_lower = candidate.lower()
        print(f"DEBUG: Fallback considering candidate: {candidate}")

        if email_prefix and len(email_prefix) > 7 and email_parts:
            likely_first = email_prefix[:len(email_prefix)//2]
            likely_last = email_prefix[len(email_prefix)//2:]
            if (likely_first in candidate_lower and likely_last in candidate_lower):
                print(f"DEBUG: Fallback matched both likely_first '{likely_first}' and likely_last '{likely_last}' in candidate.")
                return candidate
            found_parts = [part for part in email_parts if part in candidate_lower]
            if len(found_parts) >= 2:
                print(f"DEBUG: Fallback matched substrings {found_parts} in candidate.")
                return candidate
        else:
            return candidate

    return None

    # --- Only consider lines in the education section for PG detection ---
    def extract_education_section(lines):
        edu_start = None
        edu_end = None
        for i, line in enumerate(lines):
            if 'education' in line.lower():
                edu_start = i
                break
        if edu_start is not None:
            for j in range(edu_start + 1, len(lines)):
                if is_section_header_name(lines[j]):
                    edu_end = j
                    break
            if edu_end is None:
                edu_end = len(lines)
            return lines[edu_start:edu_end]
        return []

    education_section_lines = extract_education_section(lines)

    found_pg = False
    for pat in pg_patterns:
        if re.search(pat, line, re.IGNORECASE):
            if is_edu_context(line, idx):
                # ❌ Avoid triggering PG if "currently studying" or "student" with UG
                if "third-year" in line.lower() or "bachelor" in line.lower():
                    continue
            found_pg = True
            break
    if found_pg:
        return True
    return False