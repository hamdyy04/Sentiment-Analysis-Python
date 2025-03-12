import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset
dataset = pd.read_csv('sentiment_analysis.csv')

# Ensure the dataset has 'text' and 'sentiment' columns
if 'text' not in dataset.columns or 'sentiment' not in dataset.columns:
    raise ValueError("Dataset must contain 'text' and 'sentiment' columns.")

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    dataset['text'], dataset['sentiment'], test_size=0.2, random_state=42
)

# Function to classify sentiment using TextBlob
def classify_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Predict sentiment for the test data
predicted_labels = [classify_sentiment(text) for text in test_data]

# Calculate accuracy
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Function to classify user input
def analyze_user_input():
    user_input = input("Enter a sentence to analyze its sentiment: ")
    sentiment = classify_sentiment(user_input)
    print(f"The sentiment of the sentence is: {sentiment}")

# Run the user input analysis
analyze_user_input()
