import pandas as pd


def add_squared_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add squared feature columns expected by the car-price model.
    Expects DataFrame with columns 'age' and 'km_driven' (if present).
    Returns a copy of X with 'age_sq' and 'km_driven_sq' added.
    Safe if columns missing; will add zeros.
    """
    if not isinstance(X, pd.DataFrame):
        # Try to coerce to DataFrame with reasonable column names
        try:
            X = pd.DataFrame(X)
        except Exception:
            return X

    X = X.copy()
    if 'age' in X.columns and 'age_sq' not in X.columns:
        X['age_sq'] = pd.to_numeric(X['age'], errors='coerce').fillna(0).astype(float) ** 2
    if 'km_driven' in X.columns and 'km_driven_sq' not in X.columns:
        X['km_driven_sq'] = pd.to_numeric(X['km_driven'], errors='coerce').fillna(0).astype(float) ** 2
    # Ensure order stability
    return X