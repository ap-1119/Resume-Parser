import os
import asyncio
import sqlite3

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory, get_flashed_messages
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

# ✅ Language Processing & Translation
from langdetect import detect, DetectorFactory
from googletrans import Translator

# ✅ NLP Libraries for Resume Parsing & Matching
import nltk
import spacy
import torch
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel

# ✅ Resume Parsing & Ranking (Updated)
from utils.parser import extract_text_from_file, translate_text_to_english
from utils.ranking import get_top_matching_jobs, calculate_match_score

# ✅ Download necessary NLP resources
nltk.download('stopwords')
spacy_nlp = spacy.load("en_core_web_sm")  # Load spaCy model

# ✅ Initialize Sentence Transformer (Better for Job Matching)
sentence_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# ✅ Load BERT Tokenizer & Model (For Text Embeddings)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# ✅ Stopwords for Cleaning
stop_words = set(stopwords.words('english'))

# # ✅ Flask App Configuration
# app = Flask(__name__)
# app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')  # Use environment variable for security

# # ✅ Set Upload Folder for Resumes
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# # ✅ Ensure upload folder exists
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# # ✅ Check File Extension
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # ✅ Database Connection
# def get_db():
#     conn = sqlite3.connect('resume_parser.db')
#     conn.row_factory = sqlite3.Row
#     return conn


DetectorFactory.seed = 0

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.secret_key = 'your_secret_key'
DATABASE = 'resume_parser.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login_signup', methods=['GET', 'POST'])
def login_signup():
    if request.method == 'POST':
        email = request.form.get('email').strip()
        password = request.form.get('password').strip()

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM candidates WHERE email = ?", (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['user_type'] = 'candidate'
            return redirect(url_for('view_jobs'))

        cursor.execute("SELECT * FROM recruiters WHERE email = ?", (email,))
        recruiter = cursor.fetchone()

        if recruiter and check_password_hash(recruiter['password'], password):
            session['user_id'] = recruiter['id']
            session['user_type'] = 'recruiter'
            return redirect(url_for('recruiter_dashboard'))

        flash('Invalid email or password. Please try again.', 'danger')
        return redirect(url_for('login_signup'))  # Redirect after flashing the message

    return render_template('login_signup.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name').strip()
        email = request.form.get('email').strip()
        password = request.form.get('password').strip()
        profession = request.form.get('profession').strip()

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM candidates WHERE email = ?", (email,))
        existing_candidate = cursor.fetchone()

        if existing_candidate:
            flash('Email already exists. Please use a different email.')
            return redirect(url_for('login_signup'))

        cursor.execute("SELECT * FROM recruiters WHERE email = ?", (email,))
        existing_recruiter = cursor.fetchone()

        if existing_recruiter:
            flash('Email already exists. Please use a different email.')
            return redirect(url_for('login_signup'))

        hashed_password = generate_password_hash(password)
        if profession == 'candidate':
            cursor.execute("INSERT INTO candidates (name, email, password, profession) VALUES (?, ?, ?, ?)", 
                           (name, email, hashed_password, profession))
        elif profession == 'recruiter':
            cursor.execute("INSERT INTO recruiters (name, email, password, profession) VALUES (?, ?, ?, ?)", 
                           (name, email, hashed_password, profession))
        
        conn.commit()
        flash('Account created successfully! You can now log in.')
        return redirect(url_for('login_signup'))

    return render_template('register.html')

# @app.route('/view_jobs')
# def view_jobs():
#     conn = get_db()
#     cursor = conn.cursor()

#     # Fetch all jobs and convert them to dictionaries
#     cursor.execute("SELECT * FROM jobs")
#     jobs = [dict(row) for row in cursor.fetchall()]  # Convert jobs to dictionaries

#     jobs_with_applications = []
#     for job in jobs:
#         job_id = job['id']

#         # Fetch applications for each job and convert to dictionaries
#         cursor.execute("""
#             SELECT applications.id as application_id, applications.candidate_id, candidates.name 
#             FROM applications 
#             JOIN candidates ON applications.candidate_id = candidates.id 
#             WHERE applications.job_id = ?
#         """, (job_id,))
        
#         applications = [dict(row) for row in cursor.fetchall()]  # Convert rows to dictionaries

#         jobs_with_applications.append({
#             'job': job,  # Now `job` is a proper dictionary
#             'applications': applications
#         })

#     flash_messages = get_flashed_messages(with_categories=True)  # Get flashed messages with categories
#     return render_template('view_jobs.html', jobs=jobs_with_applications, flash_messages=flash_messages)


@app.route('/view_jobs')
def view_jobs():
    conn = get_db()
    cursor = conn.cursor()

    search_query = request.args.get('search', '').strip()  # Get search query from request

    # Base SQL query
    sql_query = "SELECT * FROM jobs"
    params = []

    # Modify SQL query if search query exists
    if search_query:
        sql_query += " WHERE title LIKE ? OR company_name LIKE ? OR location LIKE ?"
        search_pattern = f"%{search_query}%"
        params.extend([search_pattern, search_pattern, search_pattern])

    cursor.execute(sql_query, params)
    jobs = [dict(row) for row in cursor.fetchall()]  # Convert jobs to dictionaries

    jobs_with_applications = []
    for job in jobs:
        job_id = job['id']

        # Fetch applications for each job
        cursor.execute("""
            SELECT applications.id as application_id, applications.candidate_id, candidates.name 
            FROM applications 
            JOIN candidates ON applications.candidate_id = candidates.id 
            WHERE applications.job_id = ?
        """, (job_id,))
        
        applications = [dict(row) for row in cursor.fetchall()]  # Convert to dictionaries

        jobs_with_applications.append({
            'job': job,  # Job as a dictionary
            'applications': applications
        })

    flash_messages = get_flashed_messages(with_categories=True)  # Get flashed messages
    return render_template('view_jobs.html', jobs=jobs_with_applications, flash_messages=flash_messages)


@app.route('/recruiter_dashboard')
def recruiter_dashboard():
    recruiter_id = session.get('user_id')
    if not recruiter_id:
        flash('You need to log in as a recruiter.')
        return redirect(url_for('login_signup'))

    conn = get_db()
    cursor = conn.cursor()

    # Fetch jobs posted by recruiter
    cursor.execute("SELECT * FROM jobs WHERE recruiter_id = ?", (recruiter_id,))
    jobs = cursor.fetchall()

    jobs_with_candidates = []

    for job in jobs:
        job_id = job['id']

        # Fetch candidates who applied for the job, only the latest resume
        cursor.execute('''SELECT 
            a.candidate_id, 
            c.name, 
            c.email, 
            r.file_path,
            r.id as resume_id
        FROM applications a
        JOIN candidates c ON a.candidate_id = c.id
        LEFT JOIN (
            SELECT 
                candidate_id, 
                MAX(uploaded_at) as latest_uploaded_at 
            FROM resumes 
            WHERE job_id = ? 
            GROUP BY candidate_id
        ) latest_resumes ON a.candidate_id = latest_resumes.candidate_id
        LEFT JOIN resumes r ON r.candidate_id = a.candidate_id AND r.uploaded_at = latest_resumes.latest_uploaded_at
        WHERE a.job_id = ?''', (job_id, job_id))
        
        applications = cursor.fetchall()
        
        ranked_candidates = []

        for application in applications:
            candidate_name = application['name']
            resume_file_path = application['file_path']  # Use the resume file path if available

            if not resume_file_path:
                print(f"Skipping {candidate_name} due to missing resume.")
                continue

            # Extract and process resume text
            resume_text = extract_text_from_file(resume_file_path)
            resume_text = translate_text_to_english(resume_text)

            # Calculate match score
            job_description = get_job_description(job_id)  # Ensure this function is defined
            final_score = calculate_match_score(resume_text, job_description)

            print(f"Candidate: {candidate_name} | Job Title: {job['title']} | Match Score: {final_score}%%")

            ranked_candidates.append({
                'candidate': {
                    'name': candidate_name,
                    'email': application['email'],
                },
                'resume': {
                    'file_path': resume_file_path,
                    'resume_id': application['resume_id']  # Include the resume ID
                },
                'ranking_score': round(final_score, 2)
            })

        # Sort candidates by ranking score in descending order
        ranked_candidates.sort(key=lambda x: x['ranking_score'], reverse=True)

        jobs_with_candidates.append({
            'job': job,
            'candidates': ranked_candidates
        })

    return render_template('recruiter_dashboard.html', jobs_with_candidates=jobs_with_candidates)

@app.route('/post_job', methods=['GET', 'POST'])
def post_job():
    if request.method == 'POST':
        title = request.form.get('jobTitle', '').strip()
        company_name = request.form.get('companyName', '').strip()
        location = request.form.get('location', '').strip()
        description = request.form.get('jobDescription', '').strip()
        responsibilities = request.form.get('responsibilities', '').strip()
        salary = request.form.get('salary', '').strip()
        experience = request.form.get('experiences', '').strip()
        recruiter_id = session.get('user_id')

        if not title or not description or not salary or not experience or not recruiter_id:
            flash('Please fill in all required fields.', 'danger')
            return render_template('post_job.html')

        conn = get_db()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO jobs (title, description, salary, experience, company_name, location, responsibilities, recruiter_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                           (title, description, salary, experience, company_name, location, responsibilities, recruiter_id))
            conn.commit()
            flash('Job posted successfully!')
        except Exception as e:
            flash('Error posting job. Please try again.', 'danger')

        return redirect(url_for('recruiter_dashboard'))

    return render_template('post_job.html')
@app.route('/apply_job/<int:job_id>', methods=['POST'])
def apply_job(job_id):
    # Check if the user is logged in
    if 'user_id' not in session:
        flash('You need to be logged in to apply for a job.')
        return redirect(url_for('login_signup'))

    user_id = session['user_id']
    conn = get_db()
    cursor = conn.cursor()

    # Check if the user has a default resume
    cursor.execute("SELECT default_resume_path FROM candidates WHERE id = ?", (user_id,))
    resume = cursor.fetchone()

    if not resume or not resume['default_resume_path']:
        flash('You need to upload a resume before applying for a job.')
        return redirect(url_for('upload_default_resume'))

    # Attempt to insert application into the applications table
    try:
        cursor.execute("INSERT INTO applications (job_id, candidate_id) VALUES (?, ?)", (job_id, user_id))
        conn.commit()

        # Insert the resume into the resumes table with the current timestamp
        cursor.execute("INSERT INTO resumes (job_id, candidate_id, file_path, is_default, uploaded_at) VALUES (?, ?, ?, ?, ?)", 
                       (job_id, user_id, resume['default_resume_path'], 1, sqlite3.datetime.datetime.now()))  # Assuming it's a default resume
        conn.commit()

        flash('Successfully applied for the job!')
    except sqlite3.IntegrityError:
        flash('You have already applied for this job.')
    except Exception as e:
        flash(f'An error occurred: {e}')
    finally:
        conn.close()

    return redirect(url_for('view_jobs', _anchor='flash_messages'))  # Add _anchor parameter

@app.route('/view_resume/<int:resume_id>')
def view_resume(resume_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM resumes WHERE id = ?", (resume_id,))
    resume = cursor.fetchone()

    if resume:
        cursor.execute("SELECT * FROM candidates WHERE id = ?", (resume['candidate_id'],))
        candidate = cursor.fetchone()

        if candidate:
            ranking_score = resume['ranking_score'] if 'ranking_score' in resume else None
            return render_template('view_resume.html', resume=resume, candidate=candidate, ranking_score=ranking_score)
        else:
            flash("Candidate not found.", 'danger')
            return redirect(url_for('recruiter_dashboard'))
    else:
        flash("Resume not found.", 'danger')
        return redirect(url_for('recruiter_dashboard'))

@app.route('/download_resume/<int:resume_id>')
def download_resume(resume_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM resumes WHERE id = ?", (resume_id,))
    resume = cursor.fetchone()

    if resume:
        file_path = resume['file_path']
        if os.path.exists(file_path):
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            return send_from_directory(directory, filename, as_attachment=True)
        else:
            flash("File not found on the server.", 'danger')
            return redirect(url_for('recruiter_dashboard'))
    else:
        flash("Resume not found.", 'danger')
        return redirect(url_for('recruiter_dashboard'))

@app.route('/remove_application/<int:application_id>', methods=['POST'])
def remove_application(application_id):
    conn = get_db()
    cursor = conn.cursor()

    # Ensure only the logged-in candidate can remove their application
    user_id = session['user_id']
    cursor.execute("DELETE FROM applications WHERE id = ? AND candidate_id = ?", (application_id, user_id))
    conn.commit()

    flash('Job application removed successfully!', 'success')

    # Redirect back to the view_jobs page
    return redirect(url_for('view_jobs'))

@app.route('/upload_default_resume', methods=['GET', 'POST'])
def upload_default_resume():
    if 'user_id' not in session:
        flash('You need to be logged in to upload a resume.', 'danger')
        return redirect(url_for('login_signup'))

    user_id = session['user_id']
    
    conn = get_db()
    cursor = conn.cursor()

    if request.method == 'POST':
        if 'resume' not in request.files:
            flash('No file uploaded.', 'danger')
            return redirect(request.url)

        file = request.files['resume']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Update the default resume path in the database and set uploaded_at to now
            cursor.execute("UPDATE candidates SET default_resume_path = ? WHERE id = ?", (file_path, user_id))
            cursor.execute("INSERT INTO resumes (candidate_id, file_path, uploaded_at) VALUES (?, ?, ?)", 
                           (user_id, file_path, sqlite3.datetime.datetime.now()))  # Insert the resume with the current timestamp
            conn.commit()
            flash('Resume uploaded successfully!', 'success')
        except Exception as e:
            flash(f'Error updating resume: {e}', 'danger')
        finally:
            conn.close()

        return redirect(url_for('view_jobs'))

    return render_template('upload_default_resume.html')

@app.route('/edit_job/<int:job_id>', methods=['GET', 'POST'])
def edit_job(job_id):
    conn = get_db()
    cursor = conn.cursor()

    if request.method == 'POST':
        title = request.form.get('jobTitle').strip()
        description = request.form.get('jobDescription').strip()
        salary = request.form.get('salary').strip()
        experience = request.form.get('experience').strip()
        company_name = request.form.get('companyName').strip()
        location = request.form.get('location').strip()

        cursor.execute("""
            UPDATE jobs
            SET title = ?, description = ?, salary = ?, experience = ?, company_name = ?, location = ?
            WHERE id = ?
        """, (title, description, salary, experience, company_name, location, job_id))
        conn.commit()

        flash('Job updated successfully!', 'success')
        return redirect(url_for('recruiter_dashboard'))

    cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    job = cursor.fetchone()

    return render_template('edit_job.html', job=job)

def get_job_description(job_id):
    """Fetch job description from the database"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT description FROM jobs WHERE id = ?", (job_id,))
    job = cursor.fetchone()
    if job:
        return job['description']
    return ""

@app.route('/api/jobs/<int:job_id>', methods=['DELETE'])
def delete_job(job_id):
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Check if job exists by matching job ID with the correct column name 'id'
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        job = cursor.fetchone()
        
        if job:
            cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.commit()
            flash(f'Job with ID {job_id} deleted successfully!', 'success')
            return jsonify({"message": "Job deleted successfully."}), 200
        else:
            return jsonify({"error": "Job not found."}), 404
    except Exception as e:
        print(f"Error deleting job: {e}")
        return jsonify({"error": "Error deleting job."}), 500

if __name__ == '__main__':
    app.run(debug=True)