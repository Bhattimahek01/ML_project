import numpy as np
import pandas as pd
import pickle
from model import LogisticRegressionScratch

def train():
    # Load data
    print("Loading data...")
    # The csv uses ';' as separator based on previous `Get-Content`
    df = pd.read_csv("cardio_train.csv", sep=";")

    # Drop ID if present
    if "id" in df.columns:
        print("Dropping 'id' column...")
        df = df.drop("id", axis=1)
    
    # We expect 11 features + target 'cardio'
    expected_cols = {'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio'}
    if not set(df.columns) == expected_cols:
        print(f"Warning: Columns do not match expected set. Found: {df.columns}")

    # Separate features and target
    X = df.drop("cardio", axis=1).values
    y = df["cardio"].values

    print(f"Feature shape: {X.shape}")

    # Normalization
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    # Avoid division by zero if max == min
    denom = X_max - X_min
    denom[denom == 0] = 1
    X_norm = (X - X_min) / denom

    # Train model
    # Train model
    print(f"Training model on {X.shape[1]} features for 20000 epochs with lr=1.5...")
    model = LogisticRegressionScratch(lr=1.5, epochs=20000) 
    model.fit(X_norm, y)

    # Calculate Accuracy
    preds = model.predict(X_norm)
    accuracy = np.mean(preds == y)
    print(f"New Training Accuracy (Epochs=20000, lr=1.5): {accuracy:.4f}")


    # Save
    print("Saving artifacts...")
    pickle.dump(model, open("cardio_model.pkl", "wb"))
    pickle.dump((X_min, X_max), open("scaler.pkl", "wb"))
    print("Done! Model and scaler updated.")

if __name__ == "__main__":
    train()
