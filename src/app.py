from pathlib import Path
from pickle import dump

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_URL = (
    "https://raw.githubusercontent.com/4GeeksAcademy/"
    "regularized-linear-regression-project-tutorial/main/"
    "demographic_health_data.csv"
)
TARGET_COLUMN = "Heart disease_number"
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURE_FRACTION = 0.3
LASSO_ALPHA = 1.0


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL).drop_duplicates().reset_index(drop=True)


def scale_features(data: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    numeric_columns.remove(TARGET_COLUMN)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(data[numeric_columns])

    scaled_data = pd.DataFrame(
        scaled_values,
        index=data.index,
        columns=numeric_columns,
    )
    scaled_data[TARGET_COLUMN] = data[TARGET_COLUMN]
    return scaled_data


def select_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    k = max(1, int(len(X_train.columns) * FEATURE_FRACTION))
    selection_model = SelectKBest(score_func=f_regression, k=k)
    selection_model.fit(X_train, y_train)
    selected_columns = X_train.columns[selection_model.get_support()]

    X_train_sel = pd.DataFrame(
        selection_model.transform(X_train),
        index=X_train.index,
        columns=selected_columns,
    )
    X_test_sel = pd.DataFrame(
        selection_model.transform(X_test),
        index=X_test.index,
        columns=selected_columns,
    )

    return X_train_sel, X_test_sel, y_train, y_test


def save_processed_data(
    X_train_sel: pd.DataFrame,
    X_test_sel: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_dir: Path,
) -> tuple[Path, Path]:
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_output = X_train_sel.copy()
    test_output = X_test_sel.copy()
    train_output[TARGET_COLUMN] = y_train
    test_output[TARGET_COLUMN] = y_test

    train_path = processed_dir / "clean_train.csv"
    test_path = processed_dir / "clean_test.csv"
    train_output.to_csv(train_path, index=False)
    test_output.to_csv(test_path, index=False)
    return train_path, test_path


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Lasso:
    model = Lasso(alpha=LASSO_ALPHA, max_iter=10000)
    model.fit(X_train, y_train)
    return model


def save_model(model: Lasso, base_dir: Path) -> Path:
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"lasso_alpha-{LASSO_ALPHA}.sav"
    with model_path.open("wb") as model_file:
        dump(model, model_file)
    return model_path


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent

    total_data = load_data()
    total_data_scaled = scale_features(total_data)
    X_train_sel, X_test_sel, y_train, y_test = select_features(total_data_scaled)

    train_path, test_path = save_processed_data(
        X_train_sel,
        X_test_sel,
        y_train,
        y_test,
        base_dir,
    )

    lasso_model = train_model(X_train_sel, y_train)
    predictions = lasso_model.predict(X_test_sel)
    score = r2_score(y_test, predictions)
    model_path = save_model(lasso_model, base_dir)

    print(f"Processed train data saved to: {train_path}")
    print(f"Processed test data saved to: {test_path}")
    print(f"Selected features: {list(X_train_sel.columns)}")
    print(f"Coefficients: {lasso_model.coef_}")
    print(f"R2 score: {score}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
