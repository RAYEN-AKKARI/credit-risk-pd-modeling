import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """
    Load the credit risk dataset.

    Parameters
    ----------
    path : str
        Path to the dataset file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the credit risk dataset.

    This function removes the ID column, handles missing values,
    and drops low-information variables when necessary.
    """
    df = df.copy()

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    if "tx_imp" in df.columns:
        df = df.drop(columns=["tx_imp"])

    if "ancienneté1" in df.columns:
        df["ancienneté1"] = df["ancienneté1"].fillna(df["ancienneté1"].mean())

    if "engagement2" in df.columns:
        df["engagement2"] = df["engagement2"].fillna(df["engagement2"].mean())

    if "type_imp" in df.columns:
        df["type_imp"] = df["type_imp"].fillna("No incident")

    return df


def prepare_features(df: pd.DataFrame, target_column: str = "défaut"):
    """
    Split the dataset into features and target, then encode categorical variables.
    """
    df = df.copy()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_encoded = pd.get_dummies(X, drop_first=True)

    return X_encoded, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split features and target into training and test sets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
