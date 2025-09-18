# Model loading and vectorizer file for prediction on news
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model
logreg_model = joblib.load("./model/fake_news_model.pkl")
print("Loaded Logistic Regression model from fake_news_model.pkl")

# Load TF-IDF vectorizer
tfidf = joblib.load("./model/tfidf_vectorizer.pkl")
print("Loaded TF-IDF vectorizer from tfidf_vectorizer.pkl")

# Predict examples
new_texts = [
    "Breaking: The president signs a new bill to boost renewable energy across the nation.",
    "You won't believe what this celebrity did last night! Shocking photos inside!",
    "Scientists discover a new species of frog in the Amazon rainforest.",
    "Click here to win $10,000 instantly! Limited time offer, hurry!",
    "Local government announces new measures to improve public transport safety."
]


# Text Cleaning
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

processed_texts = [preprocess(t) for t in new_texts]

# Transform with loaded TF-IDF
X_new = tfidf.transform(processed_texts)

# Predict
predictions = logreg_model.predict(X_new)
prediction_labels = ["Fake" if p == 0 else "True" for p in predictions]

# Print results
for text, label in zip(new_texts, prediction_labels):
    print(f"\nText: {text}\nPredicted Label: {label}")
