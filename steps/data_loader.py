import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from zenml import step


@step
def load_data() -> dict:
    """Load and preprocess the data."""
    # Paths to data files
    mainPath = "/content/drive/MyDrive/REU 2023 Team 1: Ice Bed Topography Prediction/Research/Yi_Work/"
    data_full_ = mainPath + "/Data/data_full.csv"
    data_1201_ = mainPath + "/Data/df_1201_validation_data.csv"

    # Load data
    df_all = pd.read_csv(data_full_)
    df1201 = pd.read_csv(data_1201_)

    # Drop unnecessary columns
    df1201 = df1201.drop(columns=["Unnamed: 0"])
    df_all = df_all.drop(columns=["Unnamed: 0"])

    # Compute v_mag
    df_all["v_mag"] = np.sqrt(df_all["surf_vx"] ** 2 + df_all["surf_vy"] ** 2)
    df1201["v_mag"] = np.sqrt(df1201["surf_vx"] ** 2 + df1201["surf_vy"] ** 2)

    # Prepare features and target
    feature_cols = ["surf_vx", "surf_vy", "surf_elv", "surf_dhdt", "surf_SMB", "v_mag"]
    X_given = df_all[feature_cols]
    Y_given = df_all["track_bed_target"]

    # Combine all data for standardization
    X_all = np.concatenate((X_given, df1201[feature_cols]))
    Y_all = pd.DataFrame(Y_given)

    # Standardize
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_all_std = scaler_X.fit_transform(X_all)
    Y_all_std = scaler_Y.fit_transform(Y_all)

    # Split data
    X_non1201 = X_all_std[0:632706, :]
    X_1201_data = X_all_std[632706:, :]

    # Train-test split
    generated = 168
    train_size_ = 0.6
    x_train, x_test, y_train, y_test = train_test_split(
        X_non1201, Y_all_std, train_size=train_size_, random_state=generated
    )

    # Validation split
    val_split = 0.2
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size=1 - val_split, random_state=generated
    )

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "x_1201": X_1201_data,
        "scaler_Y": scaler_Y,
    }
