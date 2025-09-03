# SMS-Spam-Classifier

📧 Email/SMS Spam Classifier

A machine learning-based web app built with Streamlit that classifies messages as Spam or Not Spam. This project uses NLP techniques like tokenization, stemming, and TF-IDF vectorization, along with multiple machine learning algorithms to make accurate predictions.
eatures

Preprocessing pipeline: Cleans and processes text using tokenization, stopword removal, and stemming.

TF-IDF Vectorization: Converts text into numerical vectors for model training.

Multiple ML Algorithms Tested:
Logistic Regression
Support Vector Machine (SVC)
Multinomial Naive Bayes
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Random Forest Classifier
AdaBoost Classifier
Bagging Classifier
Extra Trees Classifier
Gradient Boosting Classifier
XGBoost Classifier

Streamlit UI: Simple and interactive interface to test message

spam-classifier/
│
├── vectorizer.pkl          # Saved TF-IDF vectorizer
├── model.pkl               # Trained machine learning model
├── app.py                  # Main Streamlit application
├── requirements.txt        # Required dependencies
└── README.md               # Project documentation

Workflow
1. Data Preprocessing
Convert text to lowercase
Tokenize text using nltk.word_tokenize
Remove stopwords and punctuation
Apply stemming with Porter Stemmer

2. Feature Extraction
Use TF-IDF (Term Frequency-Inverse Document Frequency) to transform the cleaned text into a sparse matrix.

3. Model Training
Multiple models were trained and evaluated.
The best-performing model was selected and saved as model.pkl using Pickle.

4. Streamlit Integration
Users can input a message.
The app preprocesses, vectorizes, and classifies the message in real time.
