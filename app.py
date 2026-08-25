"""
app.py

This is the Flask web server. It loads the saved model + vectorizer,
and gives the webpage two things it can ask for:

  /classify         -> just the best model's answer
  /classify_compare -> every model's answer, for the "compare models" view
"""

from flask import Flask, render_template, request, jsonify
import joblib

from preprocess import clean_text

app = Flask(__name__)

# load everything we saved in train_model.py
vectorizer = joblib.load("models/vectorizer.joblib")
best_model = joblib.load("models/best_model.joblib")
best_model_name = joblib.load("models/best_model_name.joblib")
all_models = joblib.load("models/all_models.joblib")

label_names = {0: "Ham (Legitimate)", 1: "Spam / Phishing"}


@app.route("/")
def home():
    return render_template("index.html", best_model_name=best_model_name)


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    email_text = data.get("text", "")

    if email_text.strip() == "":
        return jsonify({"error": "Please paste some email text first."}), 400

    cleaned = clean_text(email_text)
    email_vector = vectorizer.transform([cleaned])

    if hasattr(best_model, "predict_proba"):
        # predict_proba gives us [probability of ham, probability of spam]
        probabilities = best_model.predict_proba(email_vector)[0]
        prediction = int(probabilities.argmax())
        confidence = round(probabilities[prediction] * 100, 2)
    else:
        prediction = int(best_model.predict(email_vector)[0])
        sub_predictions = [est.predict(email_vector)[0] for est in best_model.estimators_]
        votes_for_prediction = sum(1 for p in sub_predictions if p == prediction)
        confidence = round((votes_for_prediction / len(sub_predictions)) * 100, 2)

    return jsonify({
        "model_used": best_model_name,
        "prediction": label_names[prediction],
        "prediction_code": prediction,
        "confidence": confidence,
    })


@app.route("/classify_compare", methods=["POST"])
def classify_compare():
    data = request.get_json()
    email_text = data.get("text", "")

    if email_text.strip() == "":
        return jsonify({"error": "Please paste some email text first."}), 400

    cleaned = clean_text(email_text)
    email_vector = vectorizer.transform([cleaned])

    results = {}
    for name, model in all_models.items():
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(email_vector)[0]
            prediction = int(probabilities.argmax())
            confidence = round(probabilities[prediction] * 100, 2)
        else:
            prediction = int(model.predict(email_vector)[0])
            sub_predictions = [est.predict(email_vector)[0] for est in model.estimators_]
            votes_for_prediction = sum(1 for p in sub_predictions if p == prediction)
            confidence = round((votes_for_prediction / len(sub_predictions)) * 100, 2)

        results[name] = {
            "prediction": label_names[prediction],
            "prediction_code": prediction,
            "confidence": confidence,
        }

    return jsonify({"best_model": best_model_name, "results": results})


if __name__ == "__main__":
    app.run(debug=True)
