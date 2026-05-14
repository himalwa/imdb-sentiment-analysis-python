import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords

# 2. Load Dataset
uploaded_file = 'IMDB Dataset.csv'
df = pd.read_csv(uploaded_file)
print("Dataset:\n", df.head())

# 3. Text Preprocessing
custom_stopwords = set(stopwords.words('english')) - {'not'}
lemmatizer = nltk.WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    words = nltk.word_tokenize(text)  # Tokenize
    words = [lemmatizer.lemmatize(word) for word in words if word not in custom_stopwords]  # Lemmatize and remove stopwords
    return ' '.join(words)

def handle_negation(text):
    words = text.split()
    new_words = []
    skip_next = False
    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        if word == 'not' and i + 1 < len(words):
            new_words.append(f"{word}_{words[i+1]}")  # Combine 'not' with the next word
            skip_next = True
        else:
            new_words.append(word)
    return ' '.join(new_words)

# Handle missing values
df = df.dropna(subset=['review'])

# Apply preprocessing and negation handling
df['cleaned_review'] = df['review'].apply(preprocess_text).apply(handle_negation)
print("\nData after preprocessing:\n", df[['review', 'cleaned_review']].head())

# 4. Split the Data into Training and Testing
X = df['cleaned_review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Text Representation with TfidfVectorizer
vectorizer = TfidfVectorizer(
    max_features=10000,  # Limit features
    ngram_range=(1, 2),  # Use unigrams and bigrams
    stop_words='english',  # Remove stopwords
    min_df=5,  # Ignore terms that appear in less than 5 documents
    max_df=0.7  # Ignore terms that appear in more than 70% of documents
)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 6. Train SVM Model with Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'loss': ['hinge', 'squared_hinge']
}
grid_search = GridSearchCV(LinearSVC(max_iter=1000), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_vectorized, y_train)

print("\nBest Parameters:", grid_search.best_params_)
model = grid_search.best_estimator_

# 7. Model Evaluation
y_pred = model.predict(X_test_vectorized)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Predicting Sentiment of New Review
def predict_sentiment(text):
    text_cleaned = preprocess_text(text)
    text_cleaned = handle_negation(text_cleaned)
    text_vectorized = vectorizer.transform([text_cleaned])
    prediction = model.predict(text_vectorized)
    return 'Positive' if prediction[0] == 'positive' else 'Negative'

# Test prediction
sample_text = "The movie was not bad, actually it was quite good!"
print("\nPrediction Example:")
print("Review:", sample_text)
print("Predicted Sentiment:", predict_sentiment(sample_text))