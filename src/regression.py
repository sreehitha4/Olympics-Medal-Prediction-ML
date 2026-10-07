from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def train_regressors(X_train_scaled, X_train, y_train):
    linear_model = LinearRegression()
    ridge_model = Ridge(alpha=1.0)
    random_forest_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    linear_model.fit(X_train_scaled, y_train)
    ridge_model.fit(X_train_scaled, y_train)
    random_forest_model.fit(X_train, y_train)

    return {
        "Linear Regression": linear_model,
        "Ridge Regression": ridge_model,
        "Random Forest": random_forest_model
    }


def evaluate_regressor(model, X, y):
    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Predictions": predictions
    }


def evaluate_all_regressors(models, X_val_scaled, X_val, y_val):
    results = {}

    for name, model in models.items():
        if name == "Random Forest":
            result = evaluate_regressor(model, X_val, y_val)
        else:
            result = evaluate_regressor(model, X_val_scaled, y_val)

        results[name] = result

    return results


def select_best_regressor(results):
    best_name = min(
        results,
        key=lambda name: results[name]["RMSE"]
    )

    return best_name


if __name__ == "__main__":
    print("Regression module loaded successfully.")