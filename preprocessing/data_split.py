import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target_column='yield_ton_per_hectare', test_size=0.2, random_state=42):
    """
    Splits input DataFrame into train and test sets.

    Parameters:
    - df: pandas DataFrame containing all features and target
    - target_column: name of the target/output column to predict
    - test_size: fraction (0-1) of data to reserve for testing
    - random_state: seed for reproducibility

    Returns:
    - X_train, X_test, y_train, y_test: separated feature sets and labels
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    return X_train, X_test, y_train, y_test

# Example usage:
# data_df = pd.read_csv('engineered_features.csv')
# X_train, X_test, y_train, y_test = split_data(data_df)
