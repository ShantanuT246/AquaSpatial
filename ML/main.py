# main.py
"""
Hybrid training & inference script.
Trains an XGBoost residual model to correct deterministic rooftop harvest estimates.

Usage:
  1) generate dataset: python dataset.py
  2) train model: python main.py --train
  3) after training, use main.predict_sample(sample_input) or import Predictor class
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import xgboost as xgb

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

class Predictor:
    def __init__(self, model_path=None, scaler_path=None, encoder_path=None):
        self.model = None
        self.scaler = None
        self.encoder = None
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        if encoder_path and os.path.exists(encoder_path):
            self.encoder = joblib.load(encoder_path)
        # Feature schema used for ML model
        self.numerical_cols = ['rooftop_area', 'annual_rainfall', 'runoff_coefficient', 'slope', 'drainage_density', 'aquifer_depth', 'dwellers']
        self.categorical_cols = ['roof_material', 'soil_type']

    def _prepare_input(self, df):
        # df is a pandas DataFrame (or single-row DataFrame)
        # encode categorical using saved encoder, scale numerical using saved scaler
        # Return numpy array X ready for model.predict
        cat = df[self.categorical_cols].astype(str)
        num = df[self.numerical_cols].astype(float)

        if self.encoder is None:
            raise ValueError("Encoder not loaded")
        if self.scaler is None:
            raise ValueError("Scaler not loaded")

        cat_enc_sparse = self.encoder.transform(cat)
        if hasattr(cat_enc_sparse, "toarray"):
            cat_enc = cat_enc_sparse.toarray()
        else:
            cat_enc = cat_enc_sparse
        num_scaled = self.scaler.transform(num)
        X = np.concatenate([num_scaled, cat_enc], axis=1)
        return X

    def predict(self, input_dict):
        # Prepare a single-row DataFrame
        df = pd.DataFrame([input_dict])
        # ensure deterministic liters present or compute it
        if 'deterministic_liters' not in df.columns:
            df['runoff_coefficient'] = df.get('runoff_coefficient', None)
            # if no runoff specified, set a default based on roof_material
            def default_runoff(material):
                mapping = {'metal':0.95,'concrete':0.9,'tiles':0.8,'asphalt':0.88,'thatch':0.6}
                return mapping.get(material, 0.8)
            df['runoff_coefficient'] = df['runoff_coefficient'].fillna(df['roof_material'].apply(default_runoff))
            df['deterministic_liters'] = df['rooftop_area'] * df['annual_rainfall'] * df['runoff_coefficient']

        deterministic = float(df['deterministic_liters'].iloc[0])

        # ML prediction of residual
        if self.model is None:
            # no ML model available: return deterministic only
            return {
                'deterministic_liters': deterministic,
                'ml_adjusted_liters': deterministic,
                'residual_pred': 0.0,
                'used_ml': False
            }

        X = self._prepare_input(df)
        residual_pred = float(self.model.predict(xgb.DMatrix(X))[0])
        adjusted = deterministic + residual_pred

        return {
            'deterministic_liters': deterministic,
            'residual_pred': residual_pred,
            'ml_adjusted_liters': adjusted,
            'used_ml': True
        }

def train_and_save_model(csv_path="synthetic_dataset.csv", n_estimators=500):
    print("Loading dataset:", csv_path)
    df = pd.read_csv(csv_path)

    # Ensure deterministic_liters exists or compute
    if 'deterministic_liters' not in df.columns:
        df['deterministic_liters'] = df['rooftop_area'] * df['annual_rainfall'] * df['runoff_coefficient']

    # target: residual = observed_liters - deterministic_liters
    df['residual'] = df['observed_liters'] - df['deterministic_liters']

    # Select features
    numerical_cols = ['rooftop_area', 'annual_rainfall', 'runoff_coefficient', 'slope', 'drainage_density', 'aquifer_depth', 'dwellers']
    categorical_cols = ['roof_material', 'soil_type']

    # Fill missing numeric with median (just in case)
    for c in numerical_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(df[c].median())

    # Prepare categorical
    df[categorical_cols] = df[categorical_cols].fillna('unknown').astype(str)

    # Split
    X_num = df[numerical_cols].values
    X_cat = df[categorical_cols].values
    y = df['residual'].values

    # Fit scaler and encoder
    scaler = StandardScaler().fit(X_num)
    X_num_scaled = scaler.transform(X_num)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X_cat)
    X_cat_enc = encoder.transform(X_cat)

    X = np.hstack([X_num_scaled, X_cat_enc])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Train XGBoost (using DMatrix for speed)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    print("Training XGBoost...")
    bst = xgb.train(params, dtrain, num_boost_round=n_estimators, evals=watchlist, early_stopping_rounds=30, verbose_eval=50)

    # Evaluate
    preds = bst.predict(dtest)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"Evaluation on test set -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

    # Save model, scaler, encoder
    model_path = os.path.join(MODEL_DIR, "residual_xgb.joblib")
    # xgboost object saved via joblib
    joblib.dump(bst, model_path)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))
    print("Saved model and preprocessors to", MODEL_DIR)

    # Print feature importance
    try:
        fmap = bst.get_score(importance_type='gain')
        # correlate feature names
        cat_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
        feature_names = numerical_cols + cat_feature_names
        # map fmap keys (f0,f1...) - fallback to feature names if available
        print("Top feature importances (gain):")
        imp = bst.get_score(importance_type='gain')
        # convert to sorted list of (feature, gain)
        sorted_imp = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
        for k, v in sorted_imp[:10]:
            print(k, v)
    except Exception as e:
        print("Could not print feature importance:", e)

    return bst, scaler, encoder

def load_predictor():
    model_path = os.path.join(MODEL_DIR, "residual_xgb.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    encoder_path = os.path.join(MODEL_DIR, "encoder.pkl")
    predictor = Predictor(model_path=model_path, scaler_path=scaler_path, encoder_path=encoder_path)
    # joblib load of xgboost.Booster returns Booster object; Predictor expects joblib-loaded object
    # if joblib.load produced Booster directly that's fine; else we may need to handle conversion.
    predictor.model = joblib.load(model_path) if os.path.exists(model_path) else None
    return predictor

def main(args):
    if args.train:
        bst, scaler, encoder = train_and_save_model(csv_path=args.csv, n_estimators=args.n_estimators)
    else:
        # quick prediction demo: load predictor and run on a sample
        predictor = load_predictor()
        sample_input = {
            'rooftop_area': 90.0,
            'dwellers': 2,
            'roof_material': 'metal',
            'annual_rainfall': 1200.0,
            'aquifer_depth': 20.0,
            'soil_type': 'loamy',
            'slope': 3.0,
            'drainage_density': 0.9,
            'runoff_coefficient': 0.72
        }
        out = predictor.predict(sample_input)
        print("Sample input prediction:")
        print(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help="Train residual model")
    parser.add_argument('--csv', type=str, default="synthetic_dataset.csv", help="Path to synthetic CSV")
    parser.add_argument('--n_estimators', type=int, default=500, help="XGBoost n_estimators")
    args = parser.parse_args()
    main(args)
