"""
train_model.py

This is the main script for the project. It does everything in order:

1. Load and clean the two datasets, and chart their class balance
2. Try CountVectorizer vs TfidfVectorizer, and unigrams/bigrams/trigrams
3. Train 3 models: Naive Bayes, Logistic Regression, SVM
4. Combine them into an ensemble (majority voting + soft voting)
5. Evaluate every model (accuracy, precision, recall, F1, confusion matrix)
   and save a confusion matrix image + a bar chart comparing all models
6. Run 5-fold cross-validation on the 3 base models, to check results
   are stable and not just a fluke of one train/test split, and chart
   the results with error bars
7. Test cross-dataset performance (train on one dataset, test on the
   other), and chart it against within-dataset performance
8. Pick the best model and save it + the vectorizer for the web app

Just run:  python train_model.py
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

# this is where the confusion matrix images and comparison chart get saved
os.makedirs("plots", exist_ok=True)

from preprocess import load_dataset


# ---------------------------------------------------------------------
# STEP 1: Load and clean both datasets
# ---------------------------------------------------------------------

print("Loading datasets...")
ceas_df = load_dataset("data/CEAS.csv")
nazario_df = load_dataset("data/Nazario.csv")

# combine both datasets into one big dataset to train the final model on
all_data = pd.concat([ceas_df, nazario_df])
all_data = all_data.drop_duplicates(subset="text")

# chart showing how many spam vs ham emails are in each dataset
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, df, title in [
    (axes[0], ceas_df, "CEAS"),
    (axes[1], nazario_df, "Nazario"),
]:
    counts = df["label"].value_counts().sort_index()
    ax.bar(["Ham", "Spam"], [counts.get(0, 0), counts.get(1, 0)], color=["#4C72B0", "#DD8452"])
    ax.set_title(title)
    ax.set_ylabel("Number of emails")
plt.suptitle("Class Distribution: Ham vs Spam")
plt.tight_layout()
plt.savefig("plots/class_distribution.png", dpi=150)
plt.close()

print("CEAS emails:", len(ceas_df))
print("Nazario emails:", len(nazario_df))
print("Combined emails:", len(all_data))


# ---------------------------------------------------------------------
# STEP 2: Compare CountVectorizer vs TfidfVectorizer, and different
#          n-gram ranges (unigrams, bigrams, trigrams)
# ---------------------------------------------------------------------

print("\nComparing vectorizers...")

train_text, test_text, train_y, test_y = train_test_split(
    all_data["text"], all_data["label"], test_size=0.2, random_state=1
)

ngram_choices = [(1, 1), (1, 2), (1, 3), (2,2), (2,3), (3,3)]  # unigrams, bigrams, trigrams

for ngram in ngram_choices:
    # try CountVectorizer
    cv = CountVectorizer(ngram_range=ngram, max_features=20000, stop_words="english")
    cv_train = cv.fit_transform(train_text)
    cv_test = cv.transform(test_text)
    model = LogisticRegression(max_iter=1000)
    model.fit(cv_train, train_y)
    acc = accuracy_score(test_y, model.predict(cv_test))
    print("CountVectorizer, ngram_range =", ngram, "-> accuracy =", round(acc, 4))

    # try TfidfVectorizer
    tf = TfidfVectorizer(ngram_range=ngram, max_features=20000, stop_words="english")
    tf_train = tf.fit_transform(train_text)
    tf_test = tf.transform(test_text)
    model = LogisticRegression(max_iter=1000)
    model.fit(tf_train, train_y)
    acc = accuracy_score(test_y, model.predict(tf_test))
    print("TfidfVectorizer,  ngram_range =", ngram, "-> accuracy =", round(acc, 4))

# ---------------------------------------------------------------------
# STEP 3: Train the 3 models using TF-IDF with unigrams + bigrams
# ---------------------------------------------------------------------

print("\nTraining final models with TF-IDF (unigrams + bigrams)...")

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=20000, stop_words="english")
X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, train_y)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, train_y)

# SVM - LinearSVC doesn't give probabilities by default, and we need
# probabilities for soft voting later, so we wrap it in
# CalibratedClassifierCV, which adds probability estimates.
svm_model = CalibratedClassifierCV(LinearSVC(), cv=3)
svm_model.fit(X_train, train_y)


# ---------------------------------------------------------------------
# STEP 4: Combine the 3 models into an ensemble
#          (majority/hard voting AND soft voting)
# ---------------------------------------------------------------------

print("Building ensembles...")

estimators = [
    ("nb", nb_model),
    ("lr", lr_model),
    ("svm", svm_model),
]

# Majority voting: each model gets one "vote" for its predicted class
hard_voting_model = VotingClassifier(estimators=estimators, voting="hard")
hard_voting_model.fit(X_train, train_y)

# Soft voting: average the predicted probabilities from each model
soft_voting_model = VotingClassifier(estimators=estimators, voting="soft")
soft_voting_model.fit(X_train, train_y)


# ---------------------------------------------------------------------
# STEP 5: Evaluate every model
# ---------------------------------------------------------------------

def evaluate(name, model):
    predictions = model.predict(X_test)
    acc = accuracy_score(test_y, predictions)
    prec = precision_score(test_y, predictions)
    rec = recall_score(test_y, predictions)
    f1 = f1_score(test_y, predictions)
    cm = confusion_matrix(test_y, predictions)

    print("\n---", name, "---")
    print("Accuracy: ", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:   ", round(rec, 4))
    print("F1 score: ", round(f1, 4))
    print("Confusion matrix:")
    print(cm)

    # save a picture of the confusion matrix, easier to read than the
    # plain numbers above when this goes into the dissertation
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"plots/confusion_matrix_{safe_name}.png", dpi=150)
    plt.close()

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


print("\nEvaluating all models on the test set...")

results = []
results.append(evaluate("Naive Bayes", nb_model))
results.append(evaluate("Logistic Regression", lr_model))
results.append(evaluate("SVM", svm_model))
results.append(evaluate("Majority Voting", hard_voting_model))
results.append(evaluate("Soft Voting", soft_voting_model))

results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
print("\nAll models, best first:")
print(results_df)

# bar chart comparing every model's F1 score, for the dissertation
plt.figure(figsize=(8, 5))
plt.bar(results_df["model"], results_df["f1"], color="#4C72B0")
plt.ylabel("F1 Score")
plt.title("F1 Score Comparison Across Models")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("plots/model_comparison_f1.png", dpi=150)
plt.close()
print("\nSaved confusion matrix images and comparison chart to the plots/ folder.")


# ---------------------------------------------------------------------
# STEP 6: 5-fold cross-validation
# ---------------------------------------------------------------------

print("\n\nRunning 5-fold cross-validation on the 3 base models...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


def run_cross_validation(name, classifier):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=20000, stop_words="english")),
        ("clf", classifier),
    ])

    scores = cross_validate(
        pipeline, train_text, train_y, cv=skf,
        scoring=["accuracy", "f1", "precision", "recall"],
    )

    print("\n---", name, "(5-fold cross-validation) ---")
    print("Accuracy: ", round(scores["test_accuracy"].mean(), 4), "+/-", round(scores["test_accuracy"].std(), 4))
    print("F1 score: ", round(scores["test_f1"].mean(), 4), "+/-", round(scores["test_f1"].std(), 4))
    print("Precision:", round(scores["test_precision"].mean(), 4), "+/-", round(scores["test_precision"].std(), 4))
    print("Recall:   ", round(scores["test_recall"].mean(), 4), "+/-", round(scores["test_recall"].std(), 4))

    return {
        "model": name,
        "f1_mean": scores["test_f1"].mean(),
        "f1_std": scores["test_f1"].std(),
    }


cv_results = []
cv_results.append(run_cross_validation("Naive Bayes", MultinomialNB()))
cv_results.append(run_cross_validation("Logistic Regression", LogisticRegression(max_iter=1000)))
cv_results.append(run_cross_validation("SVM", LinearSVC()))

# bar chart with error bars, showing average F1 score +/- how much it
# varied across the 5 folds - a small error bar means stable performance
cv_results_df = pd.DataFrame(cv_results)
plt.figure(figsize=(8, 5))
plt.bar(
    cv_results_df["model"], cv_results_df["f1_mean"],
    yerr=cv_results_df["f1_std"], capsize=8, color="#55A868",
)
plt.ylabel("F1 Score (mean of 5 folds)")
plt.title("5-Fold Cross-Validation Results (error bars = standard deviation)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("plots/cross_validation_f1.png", dpi=150)
plt.close()
print("\nSaved cross-validation chart to the plots/ folder.")


# ---------------------------------------------------------------------
# STEP 7: Cross-dataset test
#          Train on CEAS only, test on Nazario and the other way around
# ---------------------------------------------------------------------

print("\n\nCross-dataset test...")


def train_and_test_across(train_df, test_df, train_name, test_name):
    tf = TfidfVectorizer(ngram_range=(1, 3), max_features=20000, stop_words="english")
    x_train = tf.fit_transform(train_df["text"])
    x_test = tf.transform(test_df["text"])

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, train_df["label"])
    predictions = model.predict(x_test)

    acc = accuracy_score(test_df["label"], predictions)
    f1 = f1_score(test_df["label"], predictions)
    print(f"Trained on {train_name}, tested on {test_name}: accuracy = {round(acc, 4)}, f1 = {round(f1, 4)}")

    return {"label": f"{train_name} -> {test_name}", "accuracy": acc, "f1": f1}


cross_results = []
cross_results.append(train_and_test_across(ceas_df, nazario_df, "CEAS", "Nazario"))
cross_results.append(train_and_test_across(nazario_df, ceas_df, "Nazario", "CEAS"))


lr_within_accuracy = results_df.loc[results_df["model"] == "Logistic Regression", "accuracy"].values[0]

chart_labels = ["Combined dataset\n(within-dataset)"] + [r["label"] for r in cross_results]
chart_values = [lr_within_accuracy] + [r["accuracy"] for r in cross_results]

plt.figure(figsize=(8, 5))
plt.bar(chart_labels, chart_values, color=["#55A868", "#C44E52", "#C44E52"])
plt.ylabel("Accuracy")
plt.title("Within-Dataset vs Cross-Dataset Accuracy (Logistic Regression)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("plots/cross_dataset_comparison.png", dpi=150)
plt.close()
print("Saved cross-dataset comparison chart to the plots/ folder.")


# ---------------------------------------------------------------------
# STEP 8: Pick the best model and save it
# ---------------------------------------------------------------------

best_result = results_df.iloc[0]
best_name = best_result["model"]
print("\nBest model:", best_name, "with F1 score:", round(best_result["f1"], 4))

# match the name back to the actual model object
models_by_name = {
    "Naive Bayes": nb_model,
    "Logistic Regression": lr_model,
    "SVM": svm_model,
    "Majority Voting": hard_voting_model,
    "Soft Voting": soft_voting_model,
}
best_model = models_by_name[best_name]

# save the vectorizer, the best model, and all models (for the
# "compare models" feature in the web app)
joblib.dump(vectorizer, "models/vectorizer.joblib")
joblib.dump(best_model, "models/best_model.joblib")
joblib.dump(best_name, "models/best_model_name.joblib")
joblib.dump(models_by_name, "models/all_models.joblib")

print("\nSaved vectorizer and models to the models/ folder.")
