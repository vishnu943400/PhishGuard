"""
predict_cli.py

This is a small standalone script that proves the saved model and
vectorizer work correctly WITHOUT needing Flask or a web browser at
all. It's the same loading + prediction logic that app.py uses, just
run directly in the terminal - useful for demonstrating that the
"backend logic" works on its own, separate from the website.

Usage:
    python3 predict_cli.py

Then just type or paste an email when prompted, and press Enter.
Type 'quit' to exit.
"""

import joblib
from preprocess import clean_text

print("Loading saved model and vectorizer...")
vectorizer = joblib.load("models/vectorizer.joblib")
best_model = joblib.load("models/best_model.joblib")
best_model_name = joblib.load("models/best_model_name.joblib")

print(f"Loaded successfully. Using model: {best_model_name}\n")

label_names = {0: "Ham (Legitimate)", 1: "Spam / Phishing"}

while True:
    email_text = input("Paste an email (or type 'quit' to exit):\n> ")

    if email_text.strip().lower() == "quit":
        break

    if email_text.strip() == "":
        print("(empty input, try again)\n")
        continue

    # exact same steps app.py performs behind the scenes
    cleaned = clean_text(email_text)
    email_vector = vectorizer.transform([cleaned])

    probabilities = best_model.predict_proba(email_vector)[0]
    prediction = int(probabilities.argmax())
    confidence = round(probabilities[prediction] * 100, 2)

    print(f"\nCleaned text seen by the model: {cleaned[:150]}")
    print(f"Prediction: {label_names[prediction]}")
    print(f"Confidence: {confidence}%\n")
    print("-" * 50)
