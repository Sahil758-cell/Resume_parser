from resume_parser import is_paragraph_skills_section

# Test 1: Well-formatted skills section (should return False)
well_formatted_text = """SKILLS
Programming Languages:  C | C++ | JAVA | Python | HTML | CSS 
Soft Skills: Communication | Teamwork | Decision Making | Planning
Database Management: SQL | MySQL"""

# Test 2: Paragraph-style skills section (should return True)
paragraph_text = """SKILLS
Sincere at work and responsible 80%
ability to work in a team achieve goal
good communication skills and
problem solving skills punctuality and
flexible to work basic computer skills
on Ms-off"""

print("=== TEST 1: Well-formatted skills section ===")
is_paragraph1 = is_paragraph_skills_section(well_formatted_text)
print(f"Result: {is_paragraph1} (should be False)")

print("\n=== TEST 2: Paragraph-style skills section ===")
is_paragraph2 = is_paragraph_skills_section(paragraph_text)
print(f"Result: {is_paragraph2} (should be True)") 