# 🎬 IMDB Sentiment Analysis using SVM and TF-IDF

## 📌 Project Overview
This project focuses on sentiment analysis of IMDB movie reviews using Natural Language Processing (NLP) and Machine Learning techniques in Python.

The model classifies movie reviews into positive and negative sentiment categories using TF-IDF vectorization and Support Vector Machine (SVM) classification.

## 🎯 Objectives
- Analyze sentiment from IMDB movie reviews
- Build a text classification model using machine learning
- Compare sentiment patterns from textual data
- Improve understanding of NLP preprocessing workflows

## 🛠️ Tools & Technologies

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/TF--IDF-Vectorization-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SVM-Classification-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NLP-Sentiment_Analysis-purple?style=for-the-badge"/>
</p>

## 📊 Dataset Information

- Dataset: IMDB Movie Reviews
- Data type: Text reviews
- Target label:
  - Positive
  - Negative

The dataset contains movie review text and sentiment labels used for supervised machine learning classification.

## 🔄 Machine Learning Workflow

1. Load Dataset
2. Text Preprocessing
3. Stopword Removal
4. Lemmatization
5. Negation Handling
6. TF-IDF Vectorization
7. Train-Test Split
8. SVM Model Training
9. Hyperparameter Tuning
10. Model Evaluation
11. Sentiment Prediction

## 🧹 Text Preprocessing

The preprocessing stage includes:
- lowercase conversion
- removing special characters
- tokenization
- stopword removal
- lemmatization
- handling missing values
- negation handling

Example:

```text
not good → not_good
```

This approach helps preserve sentiment meaning during preprocessing.

## 🤖 Model Development

### TF-IDF Vectorization
TF-IDF is used to convert text into numerical feature vectors for machine learning processing.

### Support Vector Machine (SVM)
The classification model uses `LinearSVC` from scikit-learn for sentiment prediction.

### Hyperparameter Tuning
GridSearchCV was implemented to optimize:
- regularization parameter (`C`)
- loss function configuration

## 📈 Model Evaluation

The model performance was evaluated using:
- Accuracy Score
- Classification Report
- Precision
- Recall
- F1-Score

The evaluation process helps measure the effectiveness of sentiment classification on unseen review data.

## 💡 Key Insights

- NLP preprocessing significantly improves text quality before modeling.
- Negation handling helps preserve sentiment context in movie reviews.
- TF-IDF effectively transforms text into machine learning features.
- SVM performs well for binary text classification problems.

## 🚀 Future Improvements

Potential future enhancements:
- sentiment visualization
- confusion matrix visualization
- word cloud analysis
- deep learning implementation
- deployment as web application
