from datasets import load_dataset
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Load dataset
dataset = load_dataset(
    "csv",
    data_files={
        "train": "train_multi.csv",
        "validation": "dev.csv",
        "test": "test.csv"
    }
)

print("Train labels:", Counter(dataset["train"]["label"]))
print("Validation labels:", Counter(dataset["validation"]["label"]))
print("Test labels:", Counter(dataset["test"]["label"]))

# Data
X_train = [str(x) for x in dataset["train"]["Reasoning"]]
y_train = dataset["train"]["label"]

X_val = [str(x) for x in dataset["validation"]["Reasoning"]]
y_val = dataset["validation"]["label"]

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# Evaluation
preds = model.predict(X_val_vec)

print("Accuracy:", accuracy_score(y_val, preds))
print("\nClassification Report:\n", classification_report(y_val, preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, preds))

# Save model
joblib.dump(model, "legal_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nSaved here:", os.getcwd())
print("Files created: legal_model.pkl and tfidf_vectorizer.pkl")

# Explainability
feature_names = vectorizer.get_feature_names_out()

top_positive = model.coef_[0].argsort()[-20:]
top_negative = model.coef_[0].argsort()[:20]

print("\nTop Class 1 Words:")
for i in reversed(top_positive):
    print(feature_names[i])

print("\nTop Class 0 Words:")
for i in top_negative:
    print(feature_names[i])