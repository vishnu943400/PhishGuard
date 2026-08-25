# Phishing Email Classifier

A machine learning project that classifies emails as Ham (legitimate) or
Spam/Phishing, with a Flask web app to try it out.

## Project structure

```
phishing_app/
├── data/
│   ├── CEAS_08.csv
│   └── Nazario_5.csv
├── models/                    <- created by train_model.py
│   ├── vectorizer.joblib
│   ├── best_model.joblib
│   ├── best_model_name.joblib
│   └── all_models.joblib
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   ├── script.js
│   └── vendor/                <- local copy of Bootstrap (no internet needed)
├── preprocess.py               <- cleans the email text
├── train_model.py              <- trains and evaluates everything
├── app.py                      <- Flask web app
└── requirements.txt
```

## Setup

```bash
cd phishing_app
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the models

```bash
python3 train_model.py
```

This script does everything in one place, in order:

1. Loads and cleans both datasets (`preprocess.py`)
2. Compares `CountVectorizer` vs `TfidfVectorizer`, and unigrams vs
   bigrams vs trigrams, using Logistic Regression as a quick test
3. Trains 3 models on the combined dataset: Naive Bayes, Logistic
   Regression, and SVM (using TF-IDF with unigrams + bigrams)
4. Combines the 3 models with `sklearn`'s `VotingClassifier`, once with
   `voting="hard"` (majority vote) and once with `voting="soft"`
   (averaged probabilities)
5. Evaluates all 5 models: accuracy, precision, recall, F1-score, and
   confusion matrix
6. Runs a cross-dataset test: trains on CEAS_08 only and tests on
   Nazario_5, then the other way around, to see how well the model
   generalizes to a different style of phishing email
7. Picks the best model by F1-score and saves it, along with the
   vectorizer, to the `models/` folder

### What the results showed

On the combined dataset, all 5 models scored well (F1 > 0.96), with
**SVM** coming out on top (F1 ≈ 0.995), just ahead of the soft-voting
ensemble and the majority-voting ensemble. Naive Bayes was noticeably
behind the others (F1 ≈ 0.96) — it's fast but its independence
assumption hurts its recall on this kind of text.

The cross-dataset test told a different story: a model trained only on
CEAS_08 and tested on Nazario_5 (or the other way round) dropped to
around 50-55% accuracy. This makes sense — CEAS_08 (general spam) and
Nazario_5 (targeted phishing) use pretty different language, so a model
tuned to one doesn't transfer well to the other. This is exactly why
the final model is trained on the **combined** dataset instead of
either one alone.

## 2. (Optional) Run the extended, more rigorous evaluation

```bash
python3 evaluate_advanced.py
```

This is a separate script (doesn't touch the web app or the saved
models) that adds the extra analysis expected at MSc/Level 7 standard.
It takes several minutes to run because of the cross-validation and
grid search. It produces:

- **Class balance check** - confirms the dataset is roughly 50/50
  ham/spam, so accuracy is a fair metric here
- **5-fold cross-validation** - instead of trusting a single train/test
  split, each model is trained and tested 5 times on different slices
  of the data, and we report the average score +/- standard deviation.
  This shows how *stable* each model's performance is.
- **Hyperparameter tuning (GridSearchCV)** - instead of using
  scikit-learn's default settings, this searches over a small grid of
  values (e.g. different `C` values for Logistic Regression and SVM,
  different `alpha` for Naive Bayes) and picks whichever performs best,
  rather than just accepting the defaults.
- **ROC curves and Precision-Recall curves** (`results/roc_curves.png`,
  `results/pr_curves.png`) - show how each model trades off false
  positives vs. true positives across every possible decision
  threshold, not just the default 50% cutoff. AUC (Area Under Curve)
  close to 1.0 means near-perfect separation between classes.
- **McNemar's test** - a statistical test comparing the two best
  models' predictions on the test set, to check whether one is
  *significantly* better than the other, or whether the difference
  could just be random noise (this uses a binomial test on the cases
  where the two models disagreed).

All of the numbers are also saved to `results/*.csv` and
`results/advanced_summary.txt` so they're easy to pull into a
dissertation write-up.

### What this extra analysis showed

Cross-validation confirmed the main results were stable, not a fluke
of one split - SVM was the strongest and most consistent model
(F1 = 0.993 ± 0.001 across folds), Logistic Regression close behind,
and Naive Bayes clearly weaker on recall.

Tuning `C` and `alpha` gave a small but real improvement over the
defaults for all three models. ROC AUC was above 0.998 for every
model, meaning all three separate spam from ham almost perfectly at
*some* threshold - the differences between them are more about
precision/recall balance than raw discriminative power.

McNemar's test comparing the tuned SVM and Logistic Regression came
back with p ≈ 0.75 - well above the usual 0.05 cutoff. In plain terms:
even though SVM scored slightly higher, that gap is small enough that
we can't statistically rule out it being down to chance on this test
set. That's a genuinely useful, honest finding for a dissertation -
it's more defensible to say "SVM and Logistic Regression perform
comparably, with SVM having a marginal edge" than to overclaim SVM as
definitively the best.

## 3. Run the web app

```bash
python3 app.py
```

Open **http://127.0.0.1:5000**, paste an email, and click **Classify**.
Tick **Compare models** to see how every model scored the same email.

## Notes

- `all_models.joblib` is just a plain Python dictionary of `{name: model}`
  saved with `joblib` — nothing fancy, so it's easy to load back in `app.py`.
- Bootstrap is saved locally in `static/vendor/` instead of loaded from a
  CDN, so the page still works even without an internet connection during
  a demo.
- For production use you'd want a proper WSGI server instead of Flask's
  built-in dev server, but for a project/demo this is fine.
