import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import joblib

# Load dataset
data = pd.read_csv("asl_data.csv", header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

labels = ["A", "B", "C", "D", "E", "F", "G"]

print("Dataset size:", len(data))
print("Class counts:")
print(y.value_counts().sort_index())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

print("\nCross-validation scores:", cv_scores)
print("Average cross-validation accuracy:", cv_scores.mean())

# Test accuracy
y_pred = model.predict(X_test)

test_accuracy = model.score(X_test, y_test)
print("Test accuracy:", test_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=labels))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot()
plt.title("Confusion Matrix for ASL Gesture Classification")
plt.show()

# Save model
joblib.dump(model, "asl_model.pkl")

print("Model saved as asl_model.pkl")