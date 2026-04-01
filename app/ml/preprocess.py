import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "age",
    "sex",
    "high_chol",
    "chol_check",
    "bmi",
    "smoker",
    "phys_activity",
    "fruits",
    "veggies",
    "hvy_alcohol_consump",
    "gen_hlth",
    "ment_hlth",
    "phys_hlth",
    "diff_walk",
]

COLUMN_MAPPING = {
    "Age": "age",
    "Sex": "sex",
    "HighChol": "high_chol",
    "CholCheck": "chol_check",
    "BMI": "bmi",
    "Smoker": "smoker",
    "PhysActivity": "phys_activity",
    "Fruits": "fruits",
    "Veggies": "veggies",
    "HvyAlcoholConsump": "hvy_alcohol_consump",
    "GenHlth": "gen_hlth",
    "MentHlth": "ment_hlth",
    "PhysHlth": "phys_hlth",
    "DiffWalk": "diff_walk",
}

DISEASE_CLASSES = {
    0: "Healthy",
    1: "Single Condition",
    2: "Multiple Conditions",
}


def create_preprocessor() -> ColumnTransformer:
    """Create a preprocessing pipeline with scaling."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_COLUMNS),
        ]
    )
    return preprocessor


def _assign_class(row) -> int:
    """Assign a multi-class target from binary disease columns."""
    conditions = sum([
        row["Diabetes"] == 1,
        row["HighBP"] == 1,
        row["Stroke"] == 1,
        row["HeartDiseaseorAttack"] == 1,
    ])
    if conditions == 0:
        return 0  # Healthy
    if conditions == 1:
        return 1  # Single Condition
    return 2  # Multiple Conditions


def load_and_prepare_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the diabetes CSV and return features + multi-class target."""
    df = pd.read_csv(csv_path)

    # Create multi-class target
    df["target"] = df.apply(_assign_class, axis=1)

    # Balance classes by undersampling to the smallest class size
    min_count = df["target"].value_counts().min()
    balanced_dfs = []
    for cls in df["target"].unique():
        cls_df = df[df["target"] == cls].sample(n=min_count, random_state=42)
        balanced_dfs.append(cls_df)
    df = pd.concat(balanced_dfs, ignore_index=True)

    # Rename columns
    df = df.rename(columns=COLUMN_MAPPING)

    X = df[FEATURE_COLUMNS]
    y = df["target"]

    return X, y
