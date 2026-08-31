# Phishing Email Classifier

A machine learning project that classifies emails as Ham (legitimate) or
Spam/Phishing, with a Flask web app to try it out.

## Project structure

```
phishing_app/
├── data/
│   ├── CEAS.csv
│   └── Nazario.csv
├── models/                   
│   ├── vectorizer.joblib
│   ├── best_model.joblib
│   ├── best_model_name.joblib
│   └── all_models.joblib
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   ├── script.js
│   └── vendor/                
├── preprocess.py               
├── train_model.py             
├── app.py                     
