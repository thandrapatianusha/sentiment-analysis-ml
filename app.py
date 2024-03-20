from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd  # Import Pandas

app = Flask(__name__)

# Load the trained model and vectorizer
with open('model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
def preprocess_text(text):
    if pd.isnull(text):  # Check for NaN values using Pandas
        return ''
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review_text = request.form['review']
        cleaned_text = preprocess_text(review_text)
        vectorized_text = tfidf_vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        if prediction:
            sentiment = "Positive"
            sentiment_class = "positive"
        else:
            sentiment = "Negative"
            sentiment_class = "negative"
        return render_template('result.html', review=review_text, sentiment=sentiment, sentiment_class=sentiment_class)

if __name__ == '__main__':
    app.run(debug=True)
