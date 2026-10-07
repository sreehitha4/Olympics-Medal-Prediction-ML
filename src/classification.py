from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def train_classifiers(
    X_train_scaled,
    X_train,
    y_train
):
    """Train the three classification models."""

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    # probability=False avoids the deprecated probability option
    svc_model = SVC(
        kernel="rbf",
        random_state=42
    )

    logistic_model.fit(
        X_train_scaled,
        y_train
    )

    random_forest_model.fit(
        X_train,
        y_train
    )

    svc_model.fit(
        X_train_scaled,
        y_train
    )

    return {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
        "SVC": svc_model
    }


def evaluate_classifier(
    model,
    X,
    y,
    scaled=False
):
    """Evaluate a classification model."""

    predictions = model.predict(X)

    return {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),
        "F1_Score": f1_score(
            y,
            predictions,
            zero_division=0
        ),
        "Confusion_Matrix": confusion_matrix(
            y,
            predictions
        ),
        "Predictions": predictions
    }


def evaluate_all_classifiers(
    models,
    X_val_scaled,
    X_val,
    y_val
):
    """Evaluate all classifiers on validation data."""

    results = {}

    for name, model in models.items():

        if name == "Random Forest":
            result = evaluate_classifier(
                model,
                X_val,
                y_val
            )
        else:
            result = evaluate_classifier(
                model,
                X_val_scaled,
                y_val
            )

        results[name] = result

    return results


def select_best_classifier(results):
    """Select the classifier with the highest validation F1-score."""

    best_name = max(
        results,
        key=lambda name: results[name]["F1_Score"]
    )

    return best_name


if __name__ == "__main__":
    print("Classification module loaded successfully.")