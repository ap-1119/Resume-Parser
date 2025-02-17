import sqlite3
import nltk
import spacy
import torch
import logging
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel

# ✅ Download NLTK stopwords
nltk.download('stopwords')

# ✅ Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ✅ Load resources
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")  # Load spaCy NLP model
sentence_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')  # Pretrained transformer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# ✅ Load the trained model
model = joblib.load('final_trained_model.pkl')  # Adjust the path to your saved model file

# ✅ Preprocessing: Clean text
def clean_text(text):
    """Remove stopwords and convert text to lowercase."""
    tokens = text.lower().split()
    cleaned_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(cleaned_tokens)

# ✅ Extract common words between resume and job description
def get_common_words(text1, text2):
    words1 = set(text1.split())
    words2 = set(text2.split())
    return words1.intersection(words2)

# ✅ BERT Embeddings
def get_bert_embeddings(text):
    """Get BERT embeddings for a given text."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Average pooling

# ✅ Sentence Transformer Embeddings (More Efficient)
def get_advanced_embeddings(text):
    """Get embeddings using a better SentenceTransformer model."""
    return sentence_model.encode([text], convert_to_tensor=True)

# ✅ Cosine Similarity Calculation
def calculate_cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    return cosine_similarity(embedding1.cpu().numpy(), embedding2.cpu().numpy())[0][0]

# ✅ Extract Skills using Named Entity Recognition (NER)
def extract_skills(text):
    """Extract skills using spaCy NLP."""
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "SKILL"]]
    return skills

# ✅ Extract Experience (Years) using NER
def extract_experience(text):
    """Extract years of experience using spaCy."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "DATE" and "year" in ent.text.lower():
            return ent.text  # Return experience duration
    return "0 years"

# ✅ Hybrid Match Score (TF-IDF + BERT)
def calculate_match_score(resume_text, job_description, alpha=0.7):
    """
    Compute match score using:
    - TF-IDF (weighted alpha)
    - BERT/Sentence Transformer (weighted 1-alpha)
    """
    # ✅ Clean Resume and Job Description
    cleaned_resume = clean_text(resume_text)
    cleaned_job_desc = clean_text(job_description)

    # ✅ Find common words
    common_words = get_common_words(cleaned_resume, cleaned_job_desc)

    logging.info(f"📌 Cleaned Job Description: {cleaned_job_desc}")
    logging.info(f"📌 Cleaned Resume: {cleaned_resume}")
    logging.info(f"🔹 Common Words: {common_words}")

    # ✅ If there are no common words, force score to 0
    if not common_words:
        logging.warning("⚠️ No common words found. Forcing match score to 0%.")
        return 0.0

    # ✅ TF-IDF Score
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_job_desc])
    tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # ✅ BERT Score (or Sentence Transformer)
    resume_embedding = get_advanced_embeddings(cleaned_resume)
    job_embedding = get_advanced_embeddings(cleaned_job_desc)
    bert_score = calculate_cosine_similarity(resume_embedding, job_embedding)

    # ✅ Final Hybrid Score
    hybrid_score = (alpha * tfidf_score) + ((1 - alpha) * bert_score)

    logging.info(f"🔹 TF-IDF Score: {tfidf_score:.4f}")
    logging.info(f"🔹 BERT Score: {bert_score:.4f}")
    logging.info(f"🔥 Final Hybrid Score: {hybrid_score * 100:.2f}%")

    return hybrid_score * 100  # Convert to percentage

# ✅ Weighted Match Score (Skills, Experience, General)
def calculate_weighted_match_score(resume_text, job_description):
    """
    Compute weighted match score:
    - Skills: 50%
    - Experience: 30%
    - General: 20%
    """
    # Extract weighted sections
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)
    
    resume_experience = extract_experience(resume_text)
    job_experience = extract_experience(job_description)

    # Compute section-wise similarity
    skill_score = calculate_match_score(" ".join(resume_skills), " ".join(job_skills))
    experience_score = calculate_match_score(resume_experience, job_experience)
    general_score = calculate_match_score(resume_text, job_description)

    # Apply weights
    final_score = (0.5 * skill_score) + (0.3 * experience_score) + (0.2 * general_score)

    logging.info(f"✅ Weighted Scores:")
    logging.info(f"🔹 Skill Score: {skill_score:.2f}%")
    logging.info(f"🔹 Experience Score: {experience_score:.2f}%")
    logging.info(f"🔹 General Match Score: {general_score:.2f}%")
    logging.info(f"🔥 Final Weighted Match Score: {final_score:.2f}%")

    return final_score

# ✅ Fetch Job Descriptions from SQLite Database
def get_job_descriptions_from_db(db_path):
    """Fetch job descriptions from the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, description FROM jobs")  # Adjust the query as needed
        jobs = cursor.fetchall()
        conn.close()
        return jobs
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return []

# ✅ Get Best Matching Jobs (With Filtering)
def get_top_matching_jobs(resume_text, db_path, threshold=50):
    """Find top jobs that match a resume with a minimum threshold."""
    jobs = get_job_descriptions_from_db(db_path)
    matches = []

    for job in jobs:
        job_id, title, description = job
        job_text = f"{title} {description}"
        score = calculate_weighted_match_score(resume_text, job_text)

        if score >= threshold:  # ✅ Filter out low matches
            matches.append((job_id, title, score))

    return sorted(matches, key=lambda x: x[2], reverse=True)  # Sort by highest score

# ✅ Example Usage
if __name__ == "__main__":
    db_path = 'resume_parser.db'  # Adjust path to your SQLite database
    resume_text = """
    Dr. Rajesh Ingle, with expertise in Computer Engineering, has worked extensively in distributed systems, cloud computing, and related research areas. 
    He has also worked on IoT, big data, and resource provisioning techniques.
    """

    top_jobs = get_top_matching_jobs(resume_text, db_path)
    
    logging.info("\n🎯 Top Matching Jobs:")
    for job in top_jobs:
        logging.info(f"📝 Job ID: {job[0]} | Title: {job[1]} | Score: {job[2]:.2f}%")
