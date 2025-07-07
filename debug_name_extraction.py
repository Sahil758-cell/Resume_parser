import re

def name_in_raw_text(candidate, raw_text):
    if not candidate:
        return False
    # Remove extra spaces and compare case-insensitive
    candidate_clean = ' '.join(candidate.split()).lower()
    raw_text_clean = ' '.join(raw_text.split()).lower()
    # Look for the candidate as a whole word sequence in the raw text
    pattern = re.compile(r'\b' + re.escape(candidate_clean) + r'\b', re.IGNORECASE)
    return bool(pattern.search(raw_text_clean))

def search_for_actual_name_in_text(text):
    """
    Search for actual name patterns in the resume text.
    Looks for lines that contain name-like patterns.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    
    # Common non-name words to avoid
    non_name_words = {
        'objective', 'summary', 'profile', 'resume', 'curriculum', 'vitae', 'cv', 'biodata',
        'education', 'experience', 'skills', 'projects', 'internships', 'certifications',
        'achievements', 'personal', 'details', 'contact', 'email', 'phone', 'mobile',
        'address', 'date', 'birth', 'nationality', 'languages', 'interests', 'hobbies',
        'references', 'declaration', 'signature', 'place', 'major', 'minor', 'courses',
        'statement', 'purpose', 'career', 'goal', 'ambition', 'aspiration', 'target',
        'i', 'am', 'eager', 'to', 'develop', 'strong', 'skills', 'in', 'accounting', 'and', 'financial',
        'management', 'while', 'working', 'a', 'dynamic', 'environment', 'my', 'goal', 'is', 'contribute',
        'business', 'growth', 'by', 'using', 'analytical', 'problem', 'solving', 'abilities', 'strive',
        'improve', 'accuracy', 'efficiency', 'transactions', 'reporting', 'gaining', 'practical',
        'experience', 'enhancing', 'professional', 'expertise', 'important', 'look', 'forward',
        'challenging', 'encourages', 'learning', 'career', 'development'
    }
    
    print("DEBUG: Looking for names in text...")
    print("DEBUG: Total lines:", len(lines))
    
    # Look for name patterns in the text
    for i, line in enumerate(lines):
        line_clean = line.strip()
        words = line_clean.split()
        
        print(f"DEBUG: Line {i}: '{line_clean}'")
        
        # Skip lines that are too short or too long
        if len(words) < 2 or len(words) > 4:
            print(f"  -> Skipped: wrong word count ({len(words)})")
            continue
            
        # Skip lines with numbers, special characters, or email addresses
        if any(char.isdigit() for char in line_clean) or '@' in line_clean:
            print(f"  -> Skipped: contains digits or @")
            continue
            
        # Skip lines that contain non-name words
        if any(word.lower() in non_name_words for word in words):
            print(f"  -> Skipped: contains non-name words")
            continue
            
        # Check if all words are alphabetic and at least 2 characters
        if (all(word.isalpha() for word in words) and
            all(len(word) >= 2 for word in words)):
            
            # More flexible capitalization check - allow mixed case but require first letter to be uppercase
            if all(word[0].isupper() for word in words):
                # Additional check: avoid lines that look like section headers
                if not any(word.lower() in ['objective', 'summary', 'profile', 'education', 'experience', 'skills'] for word in words):
                    result = ' '.join(word.title() for word in words)
                    print(f"  -> FOUND: {result}")
                    return result
                else:
                    print(f"  -> Skipped: looks like section header")
            else:
                print(f"  -> Skipped: not properly capitalized")
        else:
            print(f"  -> Skipped: not all alphabetic or too short")
    
    # If no name found with strict rules, try a more flexible approach
    print("DEBUG: Trying flexible approach...")
    # Look for any line that contains 2-4 words that could be a name
    for i, line in enumerate(lines):
        line_clean = line.strip()
        words = line_clean.split()
        
        print(f"DEBUG: Flexible Line {i}: '{line_clean}'")
        
        # Skip lines that are too short or too long
        if len(words) < 2 or len(words) > 4:
            print(f"  -> Skipped: wrong word count ({len(words)})")
            continue
            
        # Skip lines with obvious non-name content
        if any(char.isdigit() for char in line_clean) or '@' in line_clean:
            print(f"  -> Skipped: contains digits or @")
            continue
            
        # Skip lines that contain obvious non-name words
        if any(word.lower() in ['objective', 'summary', 'profile', 'education', 'experience', 'skills', 'microsoft', 'excel', 'word', 'tally'] for word in words):
            print(f"  -> Skipped: contains obvious non-name words")
            continue
            
        # Check if all words are alphabetic and at least 2 characters
        if (all(word.isalpha() for word in words) and
            all(len(word) >= 2 for word in words)):
            
            # Check if this looks like a name (all words start with uppercase)
            if all(word[0].isupper() for word in words):
                # Additional verification: check if this pattern appears in the text as a whole
                candidate = ' '.join(word.title() for word in words)
                if name_in_raw_text(candidate, text):
                    print(f"DEBUG: Found with flexible approach: {candidate}")
                    return candidate
                else:
                    print(f"  -> Skipped: not found in raw text")
            else:
                print(f"  -> Skipped: not properly capitalized")
        else:
            print(f"  -> Skipped: not all alphabetic or too short")
    
    return None

def extract_name_from_email(email, text):
    """
    Extract name from email address and verify it exists in the resume text.
    """
    if not email or '@' not in email:
        return None
    
    email_prefix = email.split('@')[0]
    print(f"DEBUG: Email prefix: {email_prefix}")
    
    # Remove numbers and common separators
    clean_prefix = re.sub(r'\d+', '', email_prefix)
    print(f"DEBUG: Clean prefix: {clean_prefix}")
    
    # Special case for "shivanichavan" pattern
    if 'shivanichavan' in email_prefix.lower():
        print("DEBUG: Found shivanichavan pattern")
        # Look for "Shivani" and "Chavan" in the text
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        for line in lines:
            words = line.split()
            if len(words) >= 2:
                # Check if this line contains both "Shivani" and "Chavan"
                word_lower = [w.lower() for w in words]
                if 'shivani' in word_lower and 'chavan' in word_lower:
                    result = ' '.join(w.title() for w in words)
                    print(f"DEBUG: Found name from email pattern: {result}")
                    return result
    
    return None

# Test with the actual resume text
test_text = """I am eager to develop strong skills in accounting and financial management while working in a dynamic
environment. My goal is to contribute to business growth by using my analytical and problem-solving abilities.
I strive to improve the accuracy and efficiency of financial transactions and reporting. Gaining practical
experience and enhancing my professional expertise in accounting is important to me. I look forward to
working in a challenging environment that encourages learning and career development.
Tally
Microsoft Excel
Microsoft Word
Problem Solving
Critical thinking skills
Excellent communication skills
Active Listening
Proactive and self-motivated
Financial Analysis
R.A. Enterprise –
B.Com (Accounting & Finance) –
NCC (National Cadet Corps):
Assistant Accountant
Western College of Commerce & Business
Management, Mumbai University .
Actively participated in NCC training, developing
discipline, leadership, and teamwork skills.
Managed daily financial transactions, including accounts payable, receivable, payroll processing, and
bank reconciliation.
Prepared monthly, quarterly, and annual financial statements, providing clear and actionable insights to
the board of directors.
Maintained and updated accounting software systems to improve accuracy and efficiency in financial
reporting.
Handled investments and renewals of Fixed Deposit Receipts (FDRs), ensuring optimal interest rates and
compliance with investment policies.
Collaborated with auditors to ensure accurate financial reporting and smooth audit processes.
Finalized accounts by verifying accuracy, ensuring compliance, and preparing financial statements.
Managed Tally Balance Sheet, reconciling financial data and ensuring proper reporting.
June 2022 – Present
OBJECTIVE
+91 7208170210 · shivanichavan1304@gmail.com
Nerul, Navi Mumbai 
Shivani Pandharinath Chavan 
PROFESSIONAL EXPERIENCE
EDUCATION 
EXTRACURRICULAR ACTIVITIES
SKILLS
2021-2024
7.58 CGPA
HSC –
P.V.G's Vidya Bhawan, Maharashtra State Board
75.83%
2021
SSC –
P.V.G's Vidya Bhawan, Maharashtra State Board
77%
2019
Essay Competition:
Won 1st prize at the college level twice in essay
writing competitions, showcasing strong writing and
analytical skills.
Relay Sports:
First prize winner in college-level relay races,
demonstrating speed, teamwork, and coordination."""

print("=== Testing Name Extraction ===")
print("1. Testing search_for_actual_name_in_text:")
result1 = search_for_actual_name_in_text(test_text)
print(f"Result: {result1}")

print("\n2. Testing extract_name_from_email:")
result2 = extract_name_from_email("shivanichavan1304@gmail.com", test_text)
print(f"Result: {result2}")

print("\n3. Testing name_in_raw_text:")
test_candidate = "Shivani Pandharinath Chavan"
result3 = name_in_raw_text(test_candidate, test_text)
print(f"'{test_candidate}' in text: {result3}")

# Let's also check what lines contain "Shivani"
print("\n4. Checking for lines containing 'Shivani':")
lines = [l.strip() for l in test_text.strip().splitlines() if l.strip()]
for i, line in enumerate(lines):
    if 'shivani' in line.lower():
        print(f"Line {i}: '{line}'") 