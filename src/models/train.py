import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib
import os

# Example imports for models (adjust as needed)
from models.traditionals_ml_models import LGBMAlphaModel
from models.traditionals_ml_models import RandomForestAlphaModel, GradientBoostingAlphaModel

def load_config(config_path='ml_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(features_path, labels_path):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)
    return X.values, y.values.ravel()

def get_model(model_name, params):
    if model_name == 'lgbm':
        return LGBMAlphaModel(params)
    elif model_name == 'random_forest':
        return RandomForestAlphaModel(params)
    elif model_name == 'gbr':
        return GradientBoostingAlphaModel(params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    config = load_config()
    X, y = load_data(config['features_path'], config['labels_path'])

    kf = KFold(n_splits=config.get('n_folds', 5), shuffle=True, random_state=42)
    best_score = float('inf')
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold+1}/{kf.n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = get_model(config['model'], config['model_params'])
        if hasattr(model, 'train'):
            if config['model'] == 'lgbm':
                model.train(X_train, y_train, val_data=(X_val, y_val))
            else:
                model.train(X_train, y_train)
        else:
            raise AttributeError("Model does not have a train method.")

        preds = model.predict(X_val)
        score = mean_squared_error(y_val, preds, squared=False)
        print(f"Validation RMSE: {score:.4f}")

        if score < best_score:
            best_score = score
            best_model = model

    # Save the best model
    os.makedirs(config.get('model_save_dir', 'models'), exist_ok=True)
    model_path = os.path.join(config.get('model_save_dir', 'models'), f"{config['model']}_best.pkl")
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path} with RMSE: {best_score:.4f}")

if __name__ == "__main__":
    main()