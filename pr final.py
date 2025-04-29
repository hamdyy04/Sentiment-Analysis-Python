import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
from pathlib import Path
import string

# Model's Configuration
MODEL_PATH = "sentiment_model.pkl"
REPEAT_CORRECTIONS = 10 # (controls how many times a user corrected sentiment is duplicated and added to the training data).

# Text Preprocessing function (for the dataset and the user's input) - Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join(char for char in text if not char.isdigit())
    text = ' '.join(text.split())
    return text

# Loading and preprocessing the dataset's data - Data Collection
dataset = pd.read_csv('sentiment_analysis.csv')
dataset['text'] = dataset['text'].apply(preprocess_text)  # Preprocess entire dataset
assert {'text', 'sentiment'}.issubset(dataset.columns), "Missing required columns"

# Split preprocessed data
train_data, test_data, train_labels, test_labels = train_test_split(
    dataset['text'], dataset['sentiment'], test_size=0.3, random_state=42)

# Model Training
def train_model(X_train, y_train):
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))), # Feature Extraction
        ('classifier', MultinomialNB(alpha=0.1))
    ])
    model.fit(X_train, y_train)
    return model

# Initializing Model - Deployment
def initialize_model():
    if Path(MODEL_PATH).exists():
        model = joblib.load(MODEL_PATH)
        print("Loaded trained model from disk")
    else:
        print(f"\n[{datetime.now()}] Training model")
        model = train_model(train_data, train_labels)
        joblib.dump(model, MODEL_PATH)
    return model

# Model's Verification Function
def verify_correction(model, text, expected_label):
    prediction = model.predict([text])[0]
    if prediction != expected_label:
        print(f"Model still predicts '{prediction}' instead of '{expected_label}'")
        return False
    return True

# Main Method
def main():
    global train_data, train_labels
    model = initialize_model()

# Model's Accuracy Evaluation - Evaluation
    ml_acc = accuracy_score(test_labels, model.predict(test_data))
    print(f"Model Accuracy: {ml_acc * 100:.2f}%")
    while True:
        user_input = input("\nEnter A Sentence To Analyze Its Sentiment/Tone (or 'quit'):\n> ").strip()
        if len(user_input) > 10 and ' ' not in user_input:
            print("Please Enter A valid Sentence to predict its sentiment")
            break
        elif user_input.lower() == 'quit':
            joblib.dump(model, MODEL_PATH)
            break
            
# Fetching the sentiment's prediction    
        processed_input = preprocess_text(user_input)  # Preprocess user input
        prediction = model.predict([processed_input])[0]
        print(f"\nCurrent Prediction: {prediction}")
        
# Handling the sentiment's correction
        if input("Is this sentiment correct? (y/n): ").lower() == 'n':
            correct_label = input("What is the correct sentiment (positive/negative/neutral): ").strip().lower()
            
# Store preprocessed corrections
            new_data = pd.Series([processed_input] * REPEAT_CORRECTIONS)
            new_labels = pd.Series([correct_label] * REPEAT_CORRECTIONS)
            train_data = pd.concat([train_data, new_data])
            train_labels = pd.concat([train_labels, new_labels])
            
# Model Retraining and verification
            print(f"\n[{datetime.now()}] Retraining with correction")
            model = train_model(train_data, train_labels)
            
# Immediate verification (New Result appears without starting a new instance)    
            if not verify_correction(model, processed_input, correct_label):
                extra_data = pd.Series([processed_input] * 5)
                extra_labels = pd.Series([correct_label] * 5)
                train_data = pd.concat([train_data, extra_data])
                train_labels = pd.concat([train_labels, extra_labels])
                model = train_model(train_data, train_labels)
            
# Evaluating and saving the model 
            new_acc = accuracy_score(test_labels, model.predict(test_data))
            print(f"New Accuracy: {new_acc * 100:.2f}%")
            joblib.dump(model, MODEL_PATH)

if __name__ == "__main__": main()
