import resume_parser

# Create a test text similar to Akhila's resume skills section
test_text = """
AKHILA K
Student
Skills:
• Leadership
• Communication
• Time management 
• Self-Confident

Education:
Bachelor of Engineering
"""

# Test skill extraction
print("Testing Akhila-like resume...")
skills = resume_parser.extract_skills(test_text, name="AKHILA K")
print(f"Extracted skills: {skills}")

# Also test the block extraction specifically
block_skills = resume_parser.extract_skills_from_block(test_text, name="AKHILA K")
print(f"Block skills: {block_skills}")

# Test bullet extraction
bullet_skills = resume_parser.extract_bullet_skills(test_text, name="AKHILA K")
print(f"Bullet skills: {bullet_skills}")

# Test without name to see if name is causing issues
skills_no_name = resume_parser.extract_skills(test_text)
print(f"Skills without name: {skills_no_name}")
