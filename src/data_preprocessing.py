import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "Previous_Medals",
    "Previous_2_Medals_Avg",
    "GDP_per_capita",
    "Population",
    "Life_Expectancy"
]

CLASSIFICATION_TARGET = "Medal_Winner"
REGRESSION_TARGET = "Total_Medals"


def load_data(path="data/processed/ml_data.csv"):
    return pd.read_csv(path)


def create_temporal_split(data):
    train_data = data[data["Year"].between(1988, 2008)].copy()
    validation_data = data[data["Year"] == 2012].copy()
    test_data = data[data["Year"] == 2016].copy()

    return train_data, validation_data, test_data


def prepare_features(train_data, validation_data, test_data):

    X_train = train_data[FEATURE_COLUMNS]
    X_val = validation_data[FEATURE_COLUMNS]
    X_test = test_data[FEATURE_COLUMNS]

    y_train_cls = train_data[CLASSIFICATION_TARGET]
    y_val_cls = validation_data[CLASSIFICATION_TARGET]
    y_test_cls = test_data[CLASSIFICATION_TARGET]

    y_train_reg = train_data[REGRESSION_TARGET]
    y_val_reg = validation_data[REGRESSION_TARGET]
    y_test_reg = test_data[REGRESSION_TARGET]

    return (
        X_train,
        X_val,
        X_test,
        y_train_cls,
        y_val_cls,
        y_test_cls,
        y_train_reg,
        y_val_reg,
        y_test_reg
    )


def scale_features(X_train, X_val, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )


if __name__ == "__main__":

    data = load_data()

    train_data, validation_data, test_data = create_temporal_split(data)

    print("Dataset shape:", data.shape)
    print("Training set:", train_data.shape)
    print("Validation set:", validation_data.shape)
    print("Test set:", test_data.shape)

    print("\nTraining years:", sorted(train_data["Year"].unique()))
    print("Validation year:", sorted(validation_data["Year"].unique()))
    print("Test year:", sorted(test_data["Year"].unique()))