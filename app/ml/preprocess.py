import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_prepare_data(
    train_path: str, test_path: str | None = None
) -> tuple[pd.DataFrame, pd.Series, LabelEncoder, pd.DataFrame | None, pd.Series | None]:
    """Load CSVs and return features + target (all diseases)."""
    df_train = pd.read_csv(train_path)

    if "Unnamed: 133" in df_train.columns:
        df_train = df_train.drop(columns=["Unnamed: 133"])

    df_train["prognosis"] = df_train["prognosis"].str.strip()

    symptom_cols = [c for c in df_train.columns if c != "prognosis"]

    le = LabelEncoder()
    le.fit(sorted(df_train["prognosis"].unique()))

    X_train = df_train[symptom_cols].astype(float)
    y_train = pd.Series(le.transform(df_train["prognosis"]))

    X_test, y_test = None, None
    if test_path:
        df_test = pd.read_csv(test_path)
        if "Unnamed: 133" in df_test.columns:
            df_test = df_test.drop(columns=["Unnamed: 133"])
        df_test["prognosis"] = df_test["prognosis"].str.strip()

        X_test = df_test[symptom_cols].astype(float)
        y_test = pd.Series(le.transform(df_test["prognosis"]))

    return X_train, y_train, le, X_test, y_test
