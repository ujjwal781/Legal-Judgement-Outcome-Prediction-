from datasets import load_dataset
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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

# Use Reasoning column
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

# Linear SVM
base_svm = LinearSVC(class_weight="balanced", max_iter=5000)

# Calibration gives probability-like scores
model = CalibratedClassifierCV(base_svm)

print("\nTraining Linear SVM...")
model.fit(X_train_vec, y_train)

# Evaluation
preds = model.predict(X_val_vec)

print("\nAccuracy:", accuracy_score(y_val, preds))
print("\nClassification Report:\n", classification_report(y_val, preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, preds))

# Save model
joblib.dump(model, "svm_legal_model.pkl")
joblib.dump(vectorizer, "svm_tfidf_vectorizer.pkl")

print("\nSaved here:", os.getcwd())
print("Files created: svm_legal_model.pkl and svm_tfidf_vectorizer.pkl")

X_test = [str(x) for x in dataset["test"]["Reasoning"]]
y_test = dataset["test"]["label"]

X_test_vec = vectorizer.transform(X_test)
test_preds = model.predict(X_test_vec)

print("\nTest Accuracy:", accuracy_score(y_test, test_preds))
print("\nTest Classification Report:\n", classification_report(y_test, test_preds))
print("\nTest Confusion Matrix:\n", confusion_matrix(y_test, test_preds))