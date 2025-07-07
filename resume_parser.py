import re
import os
import csv
import json
import spacy
from datetime import datetime
from collections import OrderedDict
from job_predictor import extract_text_from_pdf, extract_text_from_docx, predict_job_field, get_recommended_courses
from image_pdf_text_extractor import extract_text_from_image_pdf
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import unicodedata

nlp = spacy.load("en_core_web_trf")

# Explicitly set Tesseract path (optional, remove if PATH is working)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Ensure poppler-utils is in PATH
os.environ["PATH"] += os.pathsep + r"C:\\poppler\\poppler-24.08.0\\Library\\bin"

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
    r"post[- ]?graduate", r"post[- ]?graduation", r"post graduate diploma", r"post[- ]?grad"
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
    ner_pipe_edu = pipeline("ner", model=model_edu, tokenizer=tokenizer_edu, aggregation_strategy="simple")
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
    'date of birth', 'major', 'minor'
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
    for line in lines:
        # Remove zero-width spaces and normalize whitespace
        clean_line = re.sub(r'[\u200b\s]+', ' ', line).strip()
        # Match lines like "Name : BHUTHALE ARAVIND" or "Name: BHUTHALE ARAVIND"
        match = re.match(r'^name\s*[:\-]?\s*([A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){1,2})', clean_line, re.IGNORECASE)
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
    email_match = re.search(r'[^\s]+@[\w\-]+(?:\.[\w\-]+)+', full_text)
    details['email'] = email_match.group() if email_match else None
    phone = None
    phone_candidates = []
    incomplete_phone = None
    contact_keywords = ['contact', 'phone', 'mobile', 'tel', 'cell']
    for idx, line in enumerate(lines):
        if any(kw in line.lower() for kw in contact_keywords):
            found = re.findall(r'(?:\+91[\-\s]*)?(?:\(\+91\)[\-\s]*)?([6-9][0-9]{4}[\s-]?[0-9]{5})', line)
            found = [re.sub(r'[^0-9]', '', num) for num in found]
            phone_candidates.extend(found)
            for offset in [1, 2]:
                if idx + offset < len(lines):
                    next_line = lines[idx + offset]
                    found_next = re.findall(r'(?:\+91[\-\s]*)?(?:\(\+91\)[\-\s]*)?([6-9][0-9]{4}[\s-]?[0-9]{5})', next_line)
                    found_next = [re.sub(r'[^0-9]', '', num) for num in found_next]
                    phone_candidates.extend(found_next)
    if not phone_candidates:
        for line in lines:
            if re.search(r'\d{1,2}\s*[a-zA-Z]{3,9}\s*\d{2,4}', line) or '%' in line:
                continue
            found = re.findall(r'(?:\+91[\-\s]*)?(?:\(\+91\)[\-\s]*)?([6-9][0-9]{4}[\s-]?[0-9]{5})', line)
            found = [re.sub(r'[^0-9]', '', num) for num in found]
            phone_candidates.extend(found)
    if not phone_candidates:
        fallback_found = re.findall(r'(?:\+91[\-\s]*)?(?:\(\+91\)[\-\s]*)?([6-9][0-9]{4}[\s-]?[0-9]{5})', full_text)
        fallback_found = [re.sub(r'[^0-9]', '', num) for num in fallback_found]
        phone_candidates.extend(fallback_found)
    if phone_candidates:
        phone = next((num for num in phone_candidates if len(num) == 10), None)
        if not phone:
            incomplete_phone = next((num for num in phone_candidates if 8 <= len(num) < 10), None)
    if phone:
        details['phone'] = phone
    elif incomplete_phone:
        details['phone'] = f"{incomplete_phone} (possibly incomplete)"
    else:
        match_10 = re.search(r'(\d{10})', full_text)
        if match_10:
            details['phone'] = match_10.group(1)
        else:
            match_12 = re.search(r'(\d{2})[\s-](\d{10})', full_text)
            if match_12:
                details['phone'] = f"{match_12.group(1)} {match_12.group(2)}"
            else:
                details['phone'] = None

    # --- Name Extraction ---
    name_candidate = None
    # Conditional: Only use robust email-prefix-based extraction for likely concatenated email names
    email_match = re.search(r'[^\s]+@[\w\-]+(?:\.[\w\-]+)+', text)
    use_robust_email_name = False
    if email_match:
        email_prefix = re.sub(r'\d+', '', email_match.group().split('@')[0]).lower()
        if len(email_prefix) > 8 and all(sep not in email_prefix for sep in ['.', '_', '-']):
            use_robust_email_name = True

    if use_robust_email_name:
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
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines and not details.get('name'):
        first_line = lines[0]
        words = first_line.split()
        if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
            details['name'] = first_line
            print(f"DEBUG: details['name'] set to: {details['name']} (from first line priority)")
    # ... rest of the existing code ...

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
    if name_candidate and len(name_candidate.split()) == 1:
        for line in lines:
            words = line.split()
            if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
                name_candidate = line.title()
                break

    # Set the name in details if not already set
    if name_candidate and not details['name']:
        details['name'] = name_candidate

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
    'extra curricular activities', 'skills', 'projects', 'internships', 'work experience', 'education',
    'languages', 'certifications', 'achievements', 'personal details', 'profile', 'summary', 'career objective',
    'objective', 'contact', 'email', 'mobile', 'phone', 'address', 'declaration', 'signature', 'place', 'date',
    'major', 'minor', 'courses', 'interests', 'statement', 'references', 'reference', 'hobbies', 'activities',
    'additional details', 'trainings', 'training', 'publications', 'awards', 'responsibilities', 'accomplishments',
    'statement of purpose', 'curriculum vitae', 'resume', 'biodata', 'web page designing', 'seeking a job role',
    'student', 'performance', 'predictor', 'technology', 'used', 'flutter', 'dart', 'firebase',
    'patents and publications', 'publications and patents', 'conferences', 'journal', 'thesis', 'in submission', 'patent','professional summary',
])
def is_section_header_name(candidate):
    if not candidate:
        return False
    candidate_clean = candidate.strip().lower()
    # Remove punctuation and extra spaces
    candidate_clean = re.sub(r'[^a-z ]', '', candidate_clean)
    candidate_clean = ' '.join(candidate_clean.split())
    return candidate_clean in section_header_names

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

    def contains_any_pattern_in_context(patterns, skip_cert_course_for_ug=False):
        for idx, line in enumerate(lines):
            line_lower = line.lower()
            if skip_cert_course_for_ug:
                skip_section_headers = ["certification", "certifications", "certificate", "course", "courses", "training"]
                # If the current line or nearby lines are section headers for cert/course, skip this line
                skip_this_line = False
                for offset in range(-2, 2):
                    i = idx + offset
                    if 0 <= i < len(lines):
                        l = lines[i].lower()
                        if any(header in l for header in skip_section_headers):
                            skip_this_line = True
                            break
                if skip_this_line:
                    continue  # skip this line if in cert/course section
            for pat in patterns:
                if re.search(pat, line, re.IGNORECASE):
                    if is_edu_context(line, idx):
                        return True
        return False

    # Expanded PG patterns: abbreviations and full names
    pg_patterns = [
        r"\b(m[.\s]*s[.\s]*c[.]?|msc|master of science)\b",
        r"\b(m[.\s]*b[.\s]*a[.]?|mba|master of business administration)\b",
        r"\b(m[.\s]*c[.\s]*a[.]?|mca|master of computer application[s]?)\b",
        r"\b(m[.\s]*t[.\s]*e[.\s]*c[.\s]*h[.]?|mtech|master of technology)\b",
        r"\b(m[.\s]*e[.]?|me|master of engineering)\b",
        r"\b(m[.\s]*a[.]?|ma|master of arts)\b",
        r"\b(m[.\s]*c[.\s]*o[.\s]*m[.]?|mcom|master of commerce)\b",
        r"\bpg\b", r"\bpgd\b", r"\bpgdm\b", r"post[- ]?graduate", r"post[- ]?graduation", r"post graduate diploma", r"post[- ]?grad"
    ]
    dr_patterns = [r"\bphd\b", r"\bdoctorate\b", r"\bdphil\b"]
    g_patterns = [r"b[.\s]*s[.\s]*c[.]*", r"bsc", r"b[.\s]*a[.]*", r"ba", r"b[.\s]*b[.\s]*a[.]*", r"bba", r"b[.\s]*c[.\s]*a[.]*", r"bca", r"b[.\s]*c[.\s]*o[.\s]*m[.]*", r"bcom", r"b[.\s]*e[.\s]*", r"be", r"bachelor", r"ug", r"undergraduate", r"b\.tech", r"btech", r"bachelor's degree"]
    # Use word boundaries for UG patterns to avoid substring matches
    ug_patterns = [r"\bdiploma\b", r"\bpolytechnic\b", r"\bd\.pharma\b", r"\bdiploma in pharmacy\b", r"\biti\b", r"\bindustrial training institute\b"]
    hsc_patterns = [r"intermediate", r"hsc", r"12th", r"xii", r"higher secondary", r"10\+2", r"senior secondary"]
    ssc_patterns = [r"ssc", r"10th", r"matriculation", r"secondary", r"x", r"10th standard", r"secondary school certificate", r"matric"]

    found_pg = contains_any_pattern_in_context(pg_patterns)
    found_dr = contains_any_pattern_in_context(dr_patterns)
    found_g = contains_any_pattern_in_context(g_patterns)
    found_ug = contains_any_pattern_in_context(ug_patterns, skip_cert_course_for_ug=True)
    found_hsc = contains_any_pattern_in_context(hsc_patterns)
    found_ssc = contains_any_pattern_in_context(ssc_patterns)

    return {
        'pg': found_pg,
        'dr': found_dr,
        'g': found_g,
        'ug': found_ug,
        'hsc': found_hsc,
        'ssc': found_ssc
    }

# === Skill Extractor ===
def extract_skills(text):
    text = text.lower()
    skills_found = []
    for kw in SKILL_KEYWORDS:
        # Use word boundary matching to avoid partial word matches
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            skills_found.append(kw)
    return list(set(skills_found))

def is_skill(line):
    line_lower = line.lower()
    for skill in SKILL_KEYWORDS:
        # Use word boundary matching to avoid partial word matches
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, line_lower, re.IGNORECASE):
            return True
    return False

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
    expected_skills = set()
    try:
        with open("job_knowledge_base.csv", newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['job_title'].strip().lower() == job_title.strip().lower():
                    expected_skills.update([kw.strip().lower() for kw in row['keywords'].split(",") if kw.strip()])
    except FileNotFoundError:
        print("❌ job_knowledge_base.csv not found. AI skill suggestions unavailable.")
    return list(expected_skills - set(detected_skills))

# === Resume Score Chart ===
def display_resume_score(rating):
    pass  # Disabled as per original update

# === Main Analyzer ===
major_indian_locations = {
    "Thane", "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad", "Chennai", "Kolkata", "Surat",
    "Pune", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Visakhapatnam", "Boisar"
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
    skills = extract_skills(resume_text)
    edu = check_education(resume_text)
    proj_int = check_projects_and_internships(resume_text)
    work_exp = check_work_experience(resume_text)
    langs = check_languages(resume_text)
    certification = check_certification(resume_text)

    job_field_matches = predict_job_field(resume_text)
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

    output_data = OrderedDict([
        ("raw_text", resume_text),
        ("personal_info", personal_info),
        ("resume_score", rating),
        ("education_summary", education_summary(edu)),
        ("improvement_suggestions", suggestions),
        ("ai_skill_suggestions", ai_skill_suggestions),
        ("predicted_job_field", OrderedDict([
            ("top_matches", job_field_matches),
            ("job_title", job_field),
            ("skills", skills),
            ("preferred_location", preferred_location),
            ("recommended_courses", recommended_courses_dict)
        ]))
    ])

    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/resume_analysis_report_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
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
            "projects", "internship", "languages", "courses", "interests", "profile", "summary", "professional summary"
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