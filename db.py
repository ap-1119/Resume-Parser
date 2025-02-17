import sqlite3

def init_db():
    conn = sqlite3.connect('resume_parser.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''CREATE TABLE IF NOT EXISTS candidates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        email TEXT UNIQUE,
                        password TEXT,
                        profession TEXT,
                        default_resume_path TEXT)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS recruiters (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        email TEXT UNIQUE,
                        password TEXT,
                        profession TEXT)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS jobs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT,
                        description TEXT,
                        salary TEXT,
                        experience TEXT,
                        company_name TEXT,
                        location TEXT,
                        responsibilities TEXT,
                        recruiter_id INTEGER,
                        FOREIGN KEY (recruiter_id) REFERENCES recruiters(id));''')

    # In db.py, update the resumes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            candidate_id INTEGER,
            file_path TEXT,
            is_default BOOLEAN DEFAULT 0,
            ranking_score REAL,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs(id),
            FOREIGN KEY (candidate_id) REFERENCES candidates(id)
        );
    ''')
                    
    

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER,
        candidate_id INTEGER,
        application_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_id) REFERENCES jobs(id),
        FOREIGN KEY (candidate_id) REFERENCES candidates(id),
        UNIQUE (job_id, candidate_id)  
);
    ''')

    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()