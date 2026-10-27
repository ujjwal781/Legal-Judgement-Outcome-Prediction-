import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# CONFUSION MATRIX
# -----------------------------

cm = np.array([
    [599, 30],
    [10, 558]
])

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix - Linear SVM")

plt.colorbar()

labels = ["Class 0", "Class 1"]

plt.xticks([0, 1], labels)
plt.yticks([0, 1], labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Add numbers inside boxes
for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14
        )

plt.tight_layout()
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.savefig("results/confusion_matrix.png", bbox_inches="tight")

plt.close()


# -----------------------------
# MODEL COMPARISON CHART
# -----------------------------

models = [
    "BERT",
    "TF-IDF + LR",
    "TF-IDF + SVM"
]

accuracy = [
    47.45,
    83.88,
    96.66
]

plt.figure(figsize=(8, 5))

bars = plt.bar(models, accuracy)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

# Add values above bars
for bar, acc in zip(bars, accuracy):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        acc + 1,
        f"{acc}%",
        ha="center",
        fontsize=11
    )

plt.tight_layout()
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.savefig("results/model_comparison.png", bbox_inches="tight")

plt.close()

print("Graphs generated successfully!")