import argparse
import pandas as pd
import joblib
import os
import numpy as np

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def load_features(features_path):
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    return pd.read_csv(features_path).values

def main():
    parser = argparse.ArgumentParser(description="Generate predictions from a trained model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (e.g., .pkl)')
    parser.add_argument('--features_path', type=str, required=True, help='Path to the new features CSV file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save predictions CSV')
    args = parser.parse_args()

    # Load model and features
    model = load_model(args.model_path)
    X_new = load_features(args.features_path)

    # Predict
    if hasattr(model, 'predict'):
        preds = model.predict(X_new)
    else:
        raise AttributeError("Loaded model does not have a predict method.")

    # Save predictions
    pd.DataFrame({'prediction': np.ravel(preds)}).to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")

if __name__ == "__main__":
    main()