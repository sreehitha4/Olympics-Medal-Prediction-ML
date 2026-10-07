import numpy as np


def generate_two_stage_predictions(
    classification_model,
    regression_model,
    X_test_scaled,
    X_test,
    test_data
):
    # Stage 1: predict whether a country will win any medal
    classification_predictions = classification_model.predict(X_test_scaled)

    # Stage 2: predict medal count
    regression_predictions = regression_model.predict(X_test_scaled)

    # Final prediction starts at zero
    final_predictions = np.zeros(len(test_data))

    # Apply regression prediction only to countries
    # classified as medal winners
    winner_mask = classification_predictions == 1

    final_predictions[winner_mask] = regression_predictions[winner_mask]

    # Medal count cannot be negative
    final_predictions = np.maximum(final_predictions, 0)

    results = test_data[
        ["Year", "NOC", "Total_Medals", "Medal_Winner"]
    ].copy()

    results["Classification_Prediction"] = classification_predictions
    results["Regression_Prediction"] = regression_predictions
    results["Final_Medal_Prediction"] = final_predictions

    return results


if __name__ == "__main__":
    print("Prediction module loaded successfully.")