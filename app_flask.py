from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load final SVM model and vectorizer
model = joblib.load("svm_legal_model.pkl")
vectorizer = joblib.load("svm_tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    top_words = []

    if request.method == "POST":

        input_text = request.form["legal_text"]

        # Convert input text into TF-IDF vector
        vec = vectorizer.transform([input_text])

        # Get prediction probabilities
        proba = model.predict_proba(vec)[0]

        class0_prob = proba[0] * 100
        class1_prob = proba[1] * 100

        # ----------------------------
        # FINAL PREDICTION LOGIC
        # ----------------------------

        if class1_prob >= 65:

            prediction = (
                f"Petitioner Likely to Succeed "
                f"(Class 1: {class1_prob:.2f}% | "
                f"Class 0: {class0_prob:.2f}%)"
            )

        elif class0_prob >= 55:

            prediction = (
                f"Petitioner Likely to Fail "
                f"(Class 0: {class0_prob:.2f}% | "
                f"Class 1: {class1_prob:.2f}%)"
            )

        else:

            prediction = (
                f"Outcome Uncertain "
                f"(Class 0: {class0_prob:.2f}% | "
                f"Class 1: {class1_prob:.2f}%)"
            )

        # ----------------------------
        # EXPLAINABILITY SECTION
        # ----------------------------

        feature_names = vectorizer.get_feature_names_out()

        # Get SVM weights
        svm_weights = model.calibrated_classifiers_[0].estimator.coef_[0]

        input_indices = vec.nonzero()[1]

        word_scores = []

        for idx in input_indices:

            word = feature_names[idx]
            score = svm_weights[idx]

            word_scores.append(
                (word, round(float(score), 4))
            )

        # Sort by influence
        word_scores = sorted(
            word_scores,
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Top 10 influential words
        top_words = word_scores[:10]

    return render_template(
        "index.html",
        prediction=prediction,
        top_words=top_words
    )


if __name__ == "__main__":
    app.run(debug=True)
