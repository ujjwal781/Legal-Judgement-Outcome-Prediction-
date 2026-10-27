import joblib

model = joblib.load("legal_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

while True:
    text = input("\nEnter legal reasoning: ")

    vec = vectorizer.transform([text])

    pred = model.predict(vec)[0]

    if pred == 0:
        print("Predicted Outcome: Class 0")
    else:
        print("Predicted Outcome: Class 1")