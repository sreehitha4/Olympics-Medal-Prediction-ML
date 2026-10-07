from src.data_preprocessing import (
    load_data,
    create_temporal_split,
    prepare_features,
    scale_features
)

from src.classification import (
    train_classifiers,
    evaluate_all_classifiers,
    select_best_classifier,
    evaluate_classifier
)

from src.regression import (
    train_regressors,
    evaluate_all_regressors,
    select_best_regressor,
    evaluate_regressor
)

from src.prediction import generate_two_stage_predictions

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def main():
    # Load processed dataset
    data = load_data()

    # Create chronological train-validation-test split
    train_data, validation_data, test_data = create_temporal_split(data)

    # Prepare features and targets
    (
        X_train,
        X_val,
        X_test,
        y_train_cls,
        y_val_cls,
        y_test_cls,
        y_train_reg,
        y_val_reg,
        y_test_reg
    ) = prepare_features(
        train_data,
        validation_data,
        test_data
    )

    # Scale features using training data only
    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(
        X_train,
        X_val,
        X_test
    )

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------

    classifiers = train_classifiers(
        X_train_scaled,
        X_train,
        y_train_cls
    )

    classification_results = evaluate_all_classifiers(
        classifiers,
        X_val_scaled,
        X_val,
        y_val_cls
    )

    best_classifier_name = select_best_classifier(
        classification_results
    )

    best_classifier = classifiers[best_classifier_name]

    # ---------------------------------------------------------
    # Regression
    # ---------------------------------------------------------

    regressors = train_regressors(
        X_train_scaled,
        X_train,
        y_train_reg
    )

    regression_results = evaluate_all_regressors(
        regressors,
        X_val_scaled,
        X_val,
        y_val_reg
    )

    best_regressor_name = select_best_regressor(
        regression_results
    )

    best_regressor = regressors[best_regressor_name]

    # ---------------------------------------------------------
    # Final 2016 test evaluation
    # ---------------------------------------------------------

    classification_test_results = evaluate_classifier(
        best_classifier,
        X_test_scaled,
        y_test_cls
    )

    regression_test_results = evaluate_regressor(
        best_regressor,
        X_test_scaled,
        y_test_reg
    )

    # ---------------------------------------------------------
    # Two-stage prediction
    # ---------------------------------------------------------

    two_stage_results = generate_two_stage_predictions(
        best_classifier,
        best_regressor,
        X_test_scaled,
        X_test,
        test_data
    )

    final_predictions = (
        two_stage_results["Final_Medal_Prediction"]
    )

    two_stage_mae = mean_absolute_error(
        y_test_reg,
        final_predictions
    )

    two_stage_rmse = (
        mean_squared_error(
            y_test_reg,
            final_predictions
        ) ** 0.5
    )

    two_stage_r2 = r2_score(
        y_test_reg,
        final_predictions
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\n===== Olympics Medal Prediction =====")

    print("\nBest Classification Model:")
    print(best_classifier_name)

    print("\nBest Regression Model:")
    print(best_regressor_name)

    print("\n===== 2016 Classification =====")
    print(
        "Accuracy:",
        round(classification_test_results["Accuracy"], 4)
    )
    print(
        "Precision:",
        round(classification_test_results["Precision"], 4)
    )
    print(
        "Recall:",
        round(classification_test_results["Recall"], 4)
    )
    print(
        "F1 Score:",
        round(classification_test_results["F1_Score"], 4)
    )

    print("\n===== 2016 Regression =====")
    print(
        "MAE:",
        round(regression_test_results["MAE"], 4)
    )
    print(
        "RMSE:",
        round(regression_test_results["RMSE"], 4)
    )
    print(
        "R2:",
        round(regression_test_results["R2"], 4)
    )

    print("\n===== 2016 Two-Stage System =====")
    print("MAE:", round(two_stage_mae, 4))
    print("RMSE:", round(two_stage_rmse, 4))
    print("R2:", round(two_stage_r2, 4))

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    import os
    import pandas as pd

    os.makedirs(
        "results/metrics",
        exist_ok=True
    )

    os.makedirs(
        "results/predictions",
        exist_ok=True
    )

    # Final metrics table
    final_metrics = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Task": "Classification",
            "Accuracy": classification_test_results["Accuracy"],
            "Precision": classification_test_results["Precision"],
            "Recall": classification_test_results["Recall"],
            "F1": classification_test_results["F1_Score"],
            "MAE": None,
            "RMSE": None,
            "R2": None
        },
        {
            "Model": "Linear Regression",
            "Task": "Regression",
            "Accuracy": None,
            "Precision": None,
            "Recall": None,
            "F1": None,
            "MAE": regression_test_results["MAE"],
            "RMSE": regression_test_results["RMSE"],
            "R2": regression_test_results["R2"]
        },
        {
            "Model": "Two-Stage System",
            "Task": "Classification + Regression",
            "Accuracy": None,
            "Precision": None,
            "Recall": None,
            "F1": None,
            "MAE": two_stage_mae,
            "RMSE": two_stage_rmse,
            "R2": two_stage_r2
        }
    ])

    final_metrics.to_csv(
        "results/metrics/final_results_2016.csv",
        index=False
    )

    # Save country-level 2016 predictions
    two_stage_results.to_csv(
        "results/predictions/2016_predictions.csv",
        index=False
    )

    print("\nFinal metrics saved to:")
    print(
        "results/metrics/final_results_2016.csv"
    )

    print("\n2016 predictions saved to:")
    print(
        "results/predictions/2016_predictions.csv"
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()