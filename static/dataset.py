import multiprocessing

# ✅ Fix for Windows multiprocessing issues
multiprocessing.set_start_method('spawn', force=True)

import json
import csv
import re
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Download necessary NLTK resources
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# ✅ Function to clean text
stop_words = set(nltk.corpus.stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# ✅ Function to compute match score (workers load model inside function)
def process_resume_match_batch(args):
    """Processes a batch of resumes and finds best matches."""
    batch_data, job_texts, vectorizer, tfidf_matrix = args  # ✅ Unpack arguments
    sentence_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')  # ✅ Load inside worker
    batch_results = []
    
    print(f"Processing {len(batch_data)} resumes...")  # ✅ Debugging: Track how many resumes are processed

    for resume_text, resume_idx in batch_data:
        scores = []
        
        vectorized_resume = vectorizer.transform([resume_text])
        resume_embedding = sentence_model.encode(resume_text).reshape(1, -1)
        
        for job_idx, job_text in enumerate(job_texts):
            tfidf_score = cosine_similarity(vectorized_resume, tfidf_matrix[job_idx]).flatten()[0]
            job_embedding = sentence_model.encode(job_text).reshape(1, -1)
            bert_score = cosine_similarity(resume_embedding, job_embedding).flatten()[0]
            
            hybrid_score = (0.7 * tfidf_score) + (0.3 * bert_score)
            scores.append((job_idx, round(max(hybrid_score * 100, 10), 2)))

        scores.sort(key=lambda x: x[1], reverse=True)
        batch_results.append((resume_idx, scores[:3]))  # Top 3 matches

    print(f"Processed {len(batch_data)} resumes.")  # ✅ Debugging: Track completion of batch processing
    return batch_results

# ✅ Main script execution (Prevents infinite loops in multiprocessing)
if __name__ == '__main__':
    multiprocessing.freeze_support()  # ✅ Required for Windows multiprocessing

    # ✅ Load datasets
    print("🔄 Loading resume and job description datasets...")
    with open("static/resume_dataset.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        resumes = [row for row in reader]

    with open("static/job_description_dataset.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        jobs = [row for row in reader][5000:9000]  # ✅ Limit to 5000 jobs

    print(f"✅ Loaded {len(resumes)} resumes and {len(jobs)} job descriptions.")

    # ✅ Preprocess job descriptions
    print("🔄 Performing TF-IDF vectorization...")
    job_texts = [
        job["Job Title"] + " " + job.get("Key Skills", "") + " " + job.get("Role Category", "") +" " + job.get("Job Salary", "") +" " + job.get("Functional Area", "") +" " + job.get("Industry", "")
        for job in jobs
    ]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(job_texts)

    print("🔄 Cleaning resumes and job descriptions...")
    cleaned_resumes = [(clean_text(resume["Resume"]), idx) for idx, resume in enumerate(resumes)]
    cleaned_job_texts = [clean_text(job_text) for job_text in job_texts]

    # ✅ Parallel Processing Using Multiprocessing Pool
    print("🚀 Starting dataset generation with multiprocessing.Pool...")
    batch_size = 5  # ✅ Reduced batch size to prevent memory overload
    resume_batches = [cleaned_resumes[i:i+batch_size] for i in range(0, len(cleaned_resumes), batch_size)]
    
    matched_data = []
    with multiprocessing.Pool(processes=4) as pool:  # ✅ Use multiprocessing Pool
        results = list(pool.imap_unordered(process_resume_match_batch, [(batch, cleaned_job_texts, vectorizer, tfidf_matrix) for batch in resume_batches]))

    for resume_batch_result in results:
        for resume_idx, top_matches in resume_batch_result:
            for job_idx, match_score in top_matches:
                matched_data.append({
                    "Resume": resumes[resume_idx]["Resume"],  
                    "Job Description": jobs[job_idx]["Job Title"] + " " + jobs[job_idx].get("Key Skills", ""),
                    "Match Score": match_score
                })

    # ✅ Save dataset
    print(f"✅ Dataset generation completed! Total Matches: {len(matched_data)}")
    with open("matched_dataset.json", "w", encoding="utf-8") as file:
        json.dump(matched_data, file, indent=4)

    print("🎉 Successfully saved `matched_dataset.json` with matched resume-job pairs!")



# # ✅ Model Training & Evaluation

# def train_model(X_train, y_train):
#     model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05)
#     model.fit(X_train, y_train)
#     return model

# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = mse ** 0.5
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     print(f"RMSE: {rmse:.2f}")
#     print(f"MAE: {mae:.2f}")
#     print(f"R^2 Score: {r2:.2f}")

# # Load augmented dataset
# data = pd.read_json('matched_dataset.json')

# # Prepare features and labels
# X = data[['Resume', 'Job Description']]
# y = data['Match Score']

# # Clean the text (apply the cleaning function)
# X['Resume'] = X['Resume'].apply(clean_text)
# X['Job Description'] = X['Job Description'].apply(clean_text)

# # Combine resume and job description for model training/testing
# X_combined = X['Resume'] + " " + X['Job Description']

# # Get TF-IDF features for both train and test
# vectorizer = TfidfVectorizer(stop_words='english')
# X_tfidf = vectorizer.fit_transform(X_combined)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# # Train the model
# model = train_model(X_train, y_train)

# # Evaluate the model
# evaluate_model(model, X_test, y_test)

# # Save the trained model
# with open('resume_match_model.pkl', 'wb') as f:
#     joblib.dump(model, f)

# print("Model saved successfully!")
