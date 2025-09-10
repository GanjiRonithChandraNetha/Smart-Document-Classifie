import os

def create_requirements_txt():
    return """streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
spacy>=3.6.0
PyPDF2>=3.0.0
python-docx>=0.8.11
joblib>=1.3.0
"""

def create_setup_instructions():
    return """# Smart Document Classifier - Setup Instructions
1. Install dependencies:
   pip install -r requirements.txt
2. Download spaCy model:
   python -m spacy download en_core_web_sm
3. Run the app:
   streamlit run app.py
"""

class DocumentValidator:
    MAX_FILE_SIZE = 10*1024*1024
    ALLOWED_EXT = ['.pdf','.docx','.doc']

    @classmethod
    def validate_file(cls, path, size):
        if size > cls.MAX_FILE_SIZE:
            return False,"File too large"
        if os.path.splitext(path)[1].lower() not in cls.ALLOWED_EXT:
            return False,"Unsupported file type"
        return True,"OK"
