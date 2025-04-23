import pandas as pd  # Data manipulation (DataFrames)
from sklearn.model_selection import train_test_split  # Splits data into training/testing sets
from sklearn.metrics import accuracy_score  # Measures model accuracy
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text to numerical features
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier for text
from sklearn.pipeline import Pipeline  # Combines preprocessing + model into a single workflow
import joblib  # Saves/loads trained models
from datetime import datetime  # Timestamps for logging
from pathlib import Path  # Handles file paths
import string  # For text preprocessing (punctuation removal)

# Model's Configuration
MODEL_PATH = "sentiment_model.pkl" # File to save/load the trained model
REPEAT_CORRECTIONS = 10 #(controls how many times a user corrected sentiment is duplicated and added to the training data.)

# Text Preprocessing function (for the dataset and the user's input) - Preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase (e.g., "Hello" → "hello")
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation (e.g., "Hi!" → "Hi")
    text = ''.join(char for char in text if not char.isdigit())  # Remove numbers (e.g., "123abc" → "abc")
    text = ' '.join(text.split())  # Remove extra whitespace (e.g., "  hello   world  " → "hello world")
    return text

# Loading the dataset's data - Data Collection
dataset = pd.read_csv('sentiment_analysis.csv')  # Load dataset
dataset['text'] = dataset['text'].apply(preprocess_text)  # Clean all text
assert {'text', 'sentiment'}.issubset(dataset.columns), "Missing required columns"  # Ensure dataset has text and sentiment columns

# Split preprocessed data 
# The code splits the dataset after preprocessing (preprocess_text()) but before training the model.
train_data, test_data, train_labels, test_labels = train_test_split(
    dataset['text'], dataset['sentiment'], test_size=0.2, random_state=42)
# test_size=0.2: 20% of data is reserved for testing.
# random_state=42: Ensures reproducibility (same split every time).

# Model Training
def train_model(X_train, y_train):
    # (X_train) : features - user input
    # (y_train) : Labels - Sentiment labels
    model = Pipeline([ 
        # A Pipeline is a way to chain multiple data processing steps and a final model into a single object, 
        # it ensures that all steps are executed in sequence.
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),  # Converts text to TF-IDF features (unigrams + bigrams)
        # TfidfVectorizer: Transforms text into numerical features (weights words by importance).
        ('classifier', MultinomialNB(alpha=0.1))  # Naive Bayes classifier with smoothing
        # MultinomialNB: A probabilistic classifier suitable for text data.
    ])
    model.fit(X_train, y_train)  # Train the model
    return model

# Initializing Model - Deployment
def initialize_model():
    if Path(MODEL_PATH).exists():  # Check if a trained model exists
        model = joblib.load(MODEL_PATH)  # Load it
        print("Loaded trained model from disk")
        # Avoids retraining if a saved model exists (sentiment_model.pkl).
    else:
        print(f"\n[{datetime.now()}] Training model")  # Log training start time
        model = train_model(train_data, train_labels)  # Train new model
        joblib.dump(model, MODEL_PATH)  # Save it
    return model

# Model's Verification Function (Checks if the model learned from user corrections.)
def verify_correction(model, text, expected_label):
    prediction = model.predict([text])[0]  # Get model's prediction
    if prediction != expected_label:  # If prediction is wrong
        print(f"Model still predicts '{prediction}' instead of '{expected_label}'")
        return False
    return True

# Main Method
def main():
    global train_data, train_labels
    model = initialize_model()

# Model's Accuracy Evaluation - Evaluation
    ml_acc = accuracy_score(test_labels, model.predict(test_data)) # Evaluate accuracy
    print(f"Model Accuracy: {ml_acc * 100:.2f}%")
    while True:
        user_input = input("\nEnter A Sentence To Analyze Its Sentiment/Tone (or 'quit'):\n> ").strip()
        if len(user_input) > 10 and ' ' not in user_input: # Basic input validation
            print("Please Enter A valid Sentence to predict its sentiment")
            break
        elif user_input.lower() == 'quit': # Exit condition
            joblib.dump(model, MODEL_PATH) # Save model before quitting
            break
            
# Fetching the sentiment's prediction    
        processed_input = preprocess_text(user_input)  # Preprocess (clean) user input
        prediction = model.predict([processed_input])[0] # Predict sentiment
        print(f"\nCurrent Prediction: {prediction}")
        
# Handling the sentiment's correction
        if input("Is this sentiment correct? (y/n): ").lower() == 'n':
            correct_label = input("What is the correct sentiment (positive/negative/neutral): ").strip().lower()
            
# Duplicate correction to emphasize it (REPEAT_CORRECTIONS times)
            new_data = pd.Series([processed_input] * REPEAT_CORRECTIONS) # Repeats the corrected text 10 times
            new_labels = pd.Series([correct_label] * REPEAT_CORRECTIONS) # Repeats the correct label 10 times
            # Add to Training Data
            train_data = pd.concat([train_data, new_data]) # Adds new_data to existing train_data
            train_labels = pd.concat([train_labels, new_labels]) # Append new_labels to train_labels
            
# Model Retraining and verification
            print(f"\n[{datetime.now()}] Retraining with correction")
            model = train_model(train_data, train_labels) # Retrain to adapt to corrections.
            
# Immediate verification (New Result appears without starting a new instance)    
            if not verify_correction(model, processed_input, correct_label):
# If model still predicts wrong sentiments, 5 more duplicates are added and the model is retrained to ensure that it adapted to the correction.
                extra_data = pd.Series([processed_input] * 5)
                extra_labels = pd.Series([correct_label] * 5)
                train_data = pd.concat([train_data, extra_data])
                train_labels = pd.concat([train_labels, extra_labels])
                model = train_model(train_data, train_labels)
            
# Updating accuracy and saving the model 
            new_acc = accuracy_score(test_labels, model.predict(test_data))
            print(f"New Accuracy: {new_acc * 100:.2f}%")
            joblib.dump(model, MODEL_PATH)

if __name__ == "__main__": main()