import docx
import pdfplumber
import pytesseract
from PIL import Image
from langdetect import detect
from googletrans import Translator, LANGUAGES


def extract_text_from_file(file_path):
    """Extract text from different file formats (PDF, DOCX, Images, TXT)"""
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == 'docx':
        return extract_text_from_docx(file_path)
    elif file_extension in ['jpg', 'jpeg', 'png']:
        return extract_text_from_image(file_path)
    elif file_extension == 'txt':
        return extract_text_from_txt(file_path)  # Add this line for TXT support
    else:
        raise ValueError("Unsupported file format: {}".format(file_extension))

def extract_text_from_txt(txt_path):
    """Extract text from a TXT file"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"Error reading TXT file: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfplumber"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Check if the page has text
                    text += page_text
            return text
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from DOCX using python-docx"""
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return ""

def extract_text_from_image(image_path):
    """Extract text from an image using OCR (Tesseract)"""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error processing image: {e}")
        return ""

# Function to translate text to English
def translate_text_to_english(text):
    """Translate non-English text to English using googletrans"""
    try:
        lang = detect(text)  # Detect the language
        if lang != 'en':
            translator = Translator()
            translated_text = translator.translate(text, src=lang, dest='en')  # Translate to English
            print(f"Detected language: {LANGUAGES.get(lang, 'Unknown')}")  # Print detected language
            return translated_text.text
        else:
            return text  # Return text as-is if it's already in English
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if an error occurs



