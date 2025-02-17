# import random
# import joblib
# import xgboost as xgb
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import logging
# import torch

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")
# from sentence_transformers import SentenceTransformer

# # Load from local directory
# model_path = "C:/Users/comp/OneDrive/Desktop/paraphrase-MiniLM-L6-v2" 
# sentence_model = SentenceTransformer(model_path)


# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Load NLP models
# sentence_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# # ✅ Load the augmented dataset
# data = pd.read_json('matched_dataset.json')

# # ✅ Preprocess the text: combining the resume and job description text
# def combine_text(resume, job_description):
#     return resume + " " + job_description

# data['Combined'] = data.apply(lambda row: combine_text(row['Resume'], row['Job Description']), axis=1)

# # ✅ Vectorize the combined text using TF-IDF
# vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Adjust max_features based on data
# X_tfidf = vectorizer.fit_transform(data['Combined'])

# # ✅ Set the target variable (match score)
# y = data['Match Score']

# # ✅ Train-Test Split (80% for training, 20% for testing)
# X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# # ✅ Feature Scaling (standardize the features)
# scaler = StandardScaler(with_mean=False)  # We need to avoid centering for sparse matrices like TF-IDF
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # ✅ Hyperparameter Tuning using GridSearchCV for XGBoost
# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0],
#     'min_child_weight': [1, 2, 3]  # Added min_child_weight for further tuning
# }

# # Create the XGBoost model
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# # GridSearchCV to find the best parameters
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# grid_search.fit(X_train_scaled, y_train)

# # Best parameters from GridSearchCV
# logging.info(f"Best parameters found: {grid_search.best_params_}")

# # ✅ Train the model using the best parameters
# best_model = grid_search.best_estimator_

# # ✅ Evaluate the model
# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
    
#     # Calculate performance metrics
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = mse ** 0.5
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
    
#     logging.info(f"RMSE: {rmse:.2f}")
#     logging.info(f"MAE: {mae:.2f}")
#     logging.info(f"R^2 Score: {r2:.2f}")

# # Evaluate on the test set
# evaluate_model(best_model, X_test_scaled, y_test)

# # ✅ Save the trained model
# joblib.dump(best_model, 'final_trained_model.pkl')
# logging.info("Model saved successfully!")



import joblib
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the saved model
model_path = 'final_trained_model.pkl'
model = joblib.load(model_path)
logging.info("Model loaded successfully!")

# Load the dataset
data = pd.read_json('matched_dataset.json')

def combine_text(resume, job_description):
    return resume + " " + job_description

data['Combined'] = data.apply(lambda row: combine_text(row['Resume'], row['Job Description']), axis=1)

# Vectorize using the same TF-IDF setup
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(data['Combined'])

# Standardize the features
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X_tfidf)

# Target variable
y = data['Match Score']

# Evaluate the model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    logging.info(f"RMSE: {rmse:.2f}")
    logging.info(f"MAE: {mae:.2f}")
    logging.info(f"R^2 Score: {r2:.2f}")

# Run evaluation
evaluate_model(model, X_scaled, y)
