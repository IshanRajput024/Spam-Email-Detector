# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# Load your dataset. Replace 'spam.csv' with your dataset file.
# 'v1' should be the column containing the email content, and 'v2' should be the column containing labels ('spam' or 'ham').
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data['v1'], data['v2'], test_size=0.2, random_state=42)

# TF-IDF Vectorization: Convert text data into numerical features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train a Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_tfidf)

# Suppress UndefinedMetricWarning to avoid warning messages
warnings.filterwarnings("ignore", category=UserWarning)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2%}')  # Print accuracy as a percentage

confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion)

# Generate a classification report with precision, recall, and F1-score
# Set zero_division to 1 to avoid warnings when there are no predicted samples for a class
report = classification_report(y_test, y_pred, zero_division=1)
print('Classification Report:')
print(report)

# Add a prompt to keep the console open after execution
input("Press Enter to exit...")
