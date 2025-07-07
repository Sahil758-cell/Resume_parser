from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os
import tempfile
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image_pdf(file_path):
    """
    Extract text from an image-based PDF using OCR with proper file handle management.
    """
    try:
        pages = convert_from_path(file_path, dpi=300, fmt='jpeg')
        text = ""
        for i, page in enumerate(pages):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_filename = temp_file.name
                page.save(temp_filename, 'JPEG')
                with Image.open(temp_filename) as img:
                    page_text = pytesseract.image_to_string(
                        img,
                        lang='eng',
                        config='--psm 6'
                    )
                    text += page_text + "\n"
            
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    os.remove(temp_filename)
                    break
                except PermissionError as e:
                    if attempt == max_attempts - 1:
                        print(f"⚠️ Failed to delete {temp_filename}: {e}")
                    time.sleep(0.1)

        if not text.strip():
            raise Exception("OCR extracted no text. Check PDF quality or Tesseract installation.")
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading image-based PDF: {str(e)}")