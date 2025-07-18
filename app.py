from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename
import os
from resume_parser import analyze_resume, is_valid_resume
from job_predictor import extract_text_from_pdf, extract_text_from_docx
from image_pdf_text_extractor import extract_text_from_image_pdf
import json
import secrets
from functools import wraps
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === API Key Setup (Fixed) ===
API_KEY = os.getenv("API_KEY") 


def require_api_key(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        key = request.headers.get('Authorization')
        if key != API_KEY:
            return jsonify({'error': 'Unauthorized: Invalid or missing API key'}), 401
        return view_function(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return ("🚀 Resume Analyzer Flask API is running! good")

@app.route('/analyze', methods=['POST'])
@require_api_key
def analyze():
    file = request.files.get('file')
    pdf_url = request.form.get('pdf_url', '').strip()

    # === Case 1: File is present and has a valid filename ===
    if file and file.filename and file.filename.endswith(('.pdf', '.docx')):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            return process_resume_file(filepath)
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"❌ File upload error: {str(e)}")
            return jsonify({"error": f"File upload failed: {str(e)}"}), 500

    # === Case 2: No file, but pdf_url is provided ===
    elif pdf_url:
        # Handle Google Drive links
        if "drive.google.com" in pdf_url:
            parsed = urlparse(pdf_url)
            if "/file/d/" in parsed.path:
                try:
                    file_id = parsed.path.split("/file/d/")[1].split("/")[0]
                    pdf_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                    print(f"[INFO] Transformed Google Drive link: {pdf_url}")
                except Exception as e:
                    return jsonify({'error': 'Unable to extract file ID from Google Drive URL.'}), 400
            else:
                return jsonify({'error': 'Unsupported Google Drive URL format.'}), 400

        if not pdf_url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format. Must start with http:// or https://'}), 400

        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not response.content.startswith(b'%PDF'):
                return jsonify({'error': 'The provided link does not contain a PDF file.'}), 400

            filename = f"downloaded_pdf_{secrets.token_hex(8)}.pdf"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)

            return process_resume_file(filepath)

        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if "403" in error_msg or "Forbidden" in error_msg:
                return jsonify({'error': 'PDF is restricted or requires authentication. Cannot access the file.'}), 403
            elif "404" in error_msg or "Not Found" in error_msg:
                return jsonify({'error': 'PDF file not found at the provided URL.'}), 404
            elif "timeout" in error_msg.lower():
                return jsonify({'error': 'Request timeout. The PDF file is too large or the server is slow.'}), 408
            else:
                return jsonify({'error': f'Cannot access PDF from URL: {error_msg}'}), 400
        except Exception as e:
            return jsonify({'error': f'Error downloading PDF: {str(e)}'}), 500

    # === Case 3: Neither file nor URL provided ===
    else:
        return jsonify({'error': 'Either a file or pdf_url must be provided'}), 400


def process_resume_file(filepath):
    """Process the resume file and return analysis results"""
    try:
        resume_text = ""
        is_ocr_used = False
        if filepath.endswith('.pdf'):
            resume_text = extract_text_from_pdf(filepath)
            if not resume_text.strip():
                print("⚠️ PDF appears to be image-based. Trying OCR...")
                resume_text = extract_text_from_image_pdf(filepath)
                is_ocr_used = True
                if isinstance(resume_text, str) and resume_text.startswith("Error"):
                    raise Exception(resume_text)
        elif filepath.endswith('.docx'):
            resume_text = extract_text_from_docx(filepath)

        if not resume_text.strip():
            raise Exception("No text extracted from file")

        print("📄 Extracted text sample:", resume_text[:200])

        if not is_valid_resume(resume_text, is_ocr=is_ocr_used):
            raise Exception("Invalid resume format. Ensure the file is a valid resume, not a grade card or certificate.")

        result = analyze_resume(filepath)
        os.remove(filepath)
        
        if result is None:
            return jsonify({"error" : "Failed to analyze resume. Check file format or server logs."}) , 500
        
        return Response(json.dumps(result, indent=4), mimetype='application/json')
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        print(f"❌ API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)