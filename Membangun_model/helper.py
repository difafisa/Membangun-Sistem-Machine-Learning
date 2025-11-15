import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split(data_path, target_column="anomaly", test_size=0.2, random_state=42):

    # 1. Load data
    data = pd.read_csv(data_path)

    # 2. Pisahkan fitur & target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
