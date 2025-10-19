🛒 *E-COMMERCE-PRODUCT-REVIEW-SENTIMENT-ANALYZER*

Classifying customer product reviews into Positive or Negative sentiments using Natural Language Processing (NLP) and Machine Learning techniques.

📖 Project Overview

This project focuses on developing an automated sentiment analysis system for e-commerce product reviews using Natural Language Processing (NLP) and Machine Learning (ML).
The goal is to help online retailers and businesses extract actionable insights from customer feedback to improve user experience, identify areas of dissatisfaction, and make data-driven marketing decisions.

The system processes raw text reviews, cleans and tokenizes the data, extracts features using TF-IDF and Bag-of-Words (BoW), and classifies sentiments using multiple machine learning algorithms.
A user-friendly Streamlit web app was developed to allow interactive sentiment predictions in real time.

🎯 Objectives

Automate sentiment classification of e-commerce product reviews.

Evaluate and compare rule-based and ML-based sentiment models.

Deploy the best-performing model on a web interface for end-user interaction.

Provide insights that can guide businesses in improving customer satisfaction and product quality.

⚙️ Tools, Libraries & Tech Stack

Programming Language: Python

Libraries & Frameworks:

Pandas, NumPy, Scikit-learn

NLTK (for text cleaning and tokenization)

TF-IDF Vectorizer & Bag-of-Words

VADER Sentiment Analysis

Support Vector Machine (SVM), Naive Bayes

GridSearchCV (for hyperparameter tuning)

Streamlit (for deployment)

Environment:

Jupyter Notebook

VS Code / Anaconda

GitHub

🧩 Workflow / Steps to Execute
1️⃣ Data Collection

The dataset consisted of customer product reviews with associated metadata (rating, date, and location).

2️⃣ Data Cleaning & Preprocessing

Removed HTML tags, punctuation, URLs, and unnecessary symbols.

Tokenized the reviews using NLTK.

Removed stopwords and lemmatized text to reduce inflectional forms.

Replaced missing values in reviewer and country columns.

3️⃣ Feature Engineering

Applied TF-IDF to weigh the importance of words across documents.

Implemented Bag-of-Words (BoW) for baseline comparison.

4️⃣ Exploratory Data Analysis (EDA)

Visualized review distribution across ratings and years.

Identified that the majority of reviews were 1-star and 5-star, showing polarized opinions.

Found most reviews originated from the US, UK, and Canada.

5️⃣ Model Building

Trained multiple models:

Naive Bayes (BoW) — 93% accuracy

SVM (TF-IDF) — best performance with 94% accuracy and 0.94 F1-score

VADER (rule-based) — 76% accuracy for baseline comparison

6️⃣ Model Evaluation

Metrics used: Accuracy, Precision, Recall, F1-Score
Hyperparameter tuning with GridSearchCV improved SVM’s performance on edge-case sentiments.

7️⃣ Deployment

Deployed the final SVM model using Streamlit.
Users can type or paste product reviews and instantly see whether they are classified as Positive or Negative.

📊 Results & Key Insights
Model	Technique	Accuracy	F1-Score	Notes
Naive Bayes	Bag-of-Words	93%	0.92	Solid baseline model
SVM	TF-IDF	94%	0.94	Best performing model
VADER	Lexicon-based	76%	0.70	Struggled with nuanced reviews

Key Findings:

The majority of reviews were polarized between 1-star and 5-star.

SVM with TF-IDF captured subtle contextual patterns better than rule-based models.

The web app allows non-technical users to analyze sentiment interactively.

💡 Recommendations

Integrate deep learning models like BERT or LSTM for context-aware multilingual sentiment detection.

Expand dataset to include more regions for global sentiment coverage.

Incorporate real-time monitoring to track customer feedback trends.

🧠 Business Impact

Helped e-commerce teams prioritize negative feedback for quick resolution.

Provided sentiment-driven insights for marketing, product improvement, and customer retention.

Enabled automated sentiment scoring without manual labeling.

🧰 How to Run Locally
# Clone this repository
git clone https://github.com/<your-username>/ecommerce-sentiment-analyzer.git
cd ecommerce-sentiment-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py


The app will open in your browser at:
👉 http://localhost:8501

📁 Folder Structure
Ecommerce-Sentiment-Analyzer/
│
├── data/
│   └── reviews_dataset.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│
├── app.py
├── sentiment_model.pkl
├── requirements.txt
└── README.md

📅 Project Timeline

Start Date: May 2025

End Date: July 2025

Status: ✅ Completed

🏁 Outcome Summary

This project successfully developed and deployed a Machine Learning-based sentiment analysis system for e-commerce reviews, achieving a 94% accuracy rate.
It demonstrates proficiency in NLP, feature engineering, model evaluation, and deployment — key competencies for data science and ML engineering roles.
