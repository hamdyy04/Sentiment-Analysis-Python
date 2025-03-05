import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('sentiment_analysis.csv')

# Preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['text'] = df['text'].apply(preprocess_text)

# Split the data
X = df['text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(solver='liblinear')
model.fit(X_train_vec, y_train)

# Predict sentiment for user-entered text
new_text = input("Enter a sentence to predict its sentiment: ")
new_text_processed = preprocess_text(new_text)
new_text_vec = vectorizer.transform([new_text_processed])
prediction = model.predict(new_text_vec)
print("Predicted Sentiment:", prediction[0])

# Evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
