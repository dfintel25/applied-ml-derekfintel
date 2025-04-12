import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    VotingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Load the dataset (download from UCI and save in the same folder)
df = pd.read_csv("winequality-red.csv", sep=";")

# Display structure and first few rows
df.info()
df.head()

# Helper Function:
# Takes one input, the quality (which we will temporarily name "q" while in the function)
# And returns a string of the quality label (low, medium, high)
# This function will be used to create the "quality_label" column
def quality_to_label(q):
    if q <= 4:
        return "low"
    elif q <= 6:
        return "medium"
    else:
        return "high"


# Call the apply() method on the quality column to create the new quality_label column
df["quality_label"] = df["quality"].apply(quality_to_label)


# Then, create a numeric column for modeling: 0 = low, 1 = medium, 2 = high
def quality_to_number(q):
    if q <= 4:
        return 0
    elif q <= 6:
        return 1
    else:
        return 2


df["quality_numeric"] = df["quality"].apply(quality_to_number)

# Define input features (X) and target (y)
# Features: all columns except 'quality' and 'quality_label' and 'quality_numberic' - drop these from the input array
# Target: quality_label (the new column we just created)
X = df.drop(columns=["quality", "quality_label", "quality_numeric"])  # Features
y = df["quality_numeric"]  # Target

# Train/test split (stratify to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Helper function to train and evaluate models
def evaluate_model(name, model, X_train, y_train, X_test, y_test, results):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")

    print(f"\n{name} Results")
    print("Confusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))
    print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Train F1 Score: {train_f1:.4f}, Test F1 Score: {test_f1:.4f}")

    results.append(
        {
            "Model": name,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Train F1": train_f1,
            "Test F1": test_f1,
        }
    )

    results = []

# 1. Random Forest
evaluate_model(
    "Random Forest (100)",
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train,
    y_train,
    X_test,
    y_test,
    results,
)

# 2. Random Forest (200, max depth=10) 
#evaluate_model(
#    "Random Forest (200, max_depth=10)",
#    RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
#    X_train,
#    y_train,
#    X_test,
#    y_test,
#    results,
#)

# 3. AdaBoost 
#evaluate_model(
#    "AdaBoost (100)",
#    AdaBoostClassifier(n_estimators=100, random_state=42),
#    X_train,
#    y_train,
#    X_test,
#    y_test,
#    results,
#)

# 4. AdaBoost (200, lr=0.5) 
#evaluate_model(
#    "AdaBoost (200, lr=0.5)",
#    AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42),
#    X_train,
#    y_train,
#    X_test,
#    y_test,
#    results,
#)

# 5. Gradient Boosting
evaluate_model(
    "Gradient Boosting (100)",
    GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    ),
    X_train,
    y_train,
    X_test,
    y_test,
    results,
)

# 6. Voting Classifier (DT, SVM, NN) 
#voting1 = VotingClassifier(
#    estimators=[
#        ("DT", DecisionTreeClassifier()),
#        ("SVM", SVC(probability=True)),
#        ("NN", MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)),
#    ],
#    voting="soft",
#)
#evaluate_model(
#    "Voting (DT + SVM + NN)", voting1, X_train, y_train, X_test, y_test, results
#)

# 7. Voting Classifier (RF, LR, KNN) 
#voting2 = VotingClassifier(
#    estimators=[
#        ("RF", RandomForestClassifier(n_estimators=100)),
#        ("LR", LogisticRegression(max_iter=1000)),
#        ("KNN", KNeighborsClassifier()),
#    ],
#    voting="soft",
#)
#evaluate_model(
#    "Voting (RF + LR + KNN)", voting2, X_train, y_train, X_test, y_test, results
#)

# 8. Bagging 
#evaluate_model(
#    "Bagging (DT, 100)",
#    BaggingClassifier(
#        estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42
#    ),
#    X_train,
#    y_train,
#    X_test,
#    y_test,
#    results,
#)

# 9. MLP Classifier 
#evaluate_model(
#    "MLP Classifier",
#    MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
#    X_train,
#    y_train,
#    X_test,
#    y_test,
#    results,
#)

# Create a table of results 
results_df = pd.DataFrame(results)

# Sort by 'Test Accuracy' in descending order
df_sorted = results_df.sort_values(by="Test Accuracy", ascending=False)

print("\nSummary of All Models:")
display(results_df)