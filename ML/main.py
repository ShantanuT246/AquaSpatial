"""
Hybrid training & inference script.
Trains an XGBoost residual model to correct deterministic rooftop harvest estimates.

Optimized for memory usage with Render.com 512MB limit.

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
import xgboost as xbg

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
        
        # Precompute mapping for efficiency :cite[3]
        self._runoff_mapping = {'metal': 0.95, 'concrete': 0.9, 'tiles': 0.8, 'asphalt': 0.88, 'thatch': 0.6}

    def _prepare_input(self, df):
        # Optimized: use local variables and avoid multiple transformations :cite[3]
        cat = df[self.categorical_cols].astype(str)
        num = df[self.numerical_cols].astype(np.float32)  # Use float32 to save memory :cite[5]

        if self.encoder is None:
            raise ValueError("Encoder not loaded")
        if self.scaler is None:
            raise ValueError("Scaler not loaded")

        # Single transformation call
        cat_enc_sparse = self.encoder.transform(cat)
        if hasattr(cat_enc_sparse, "toarray"):
            cat_enc = cat_enc_sparse.toarray()
        else:
            cat_enc = cat_enc_sparse
        num_scaled = self.scaler.transform(num)
        
        # Use float32 for final array
        X = np.concatenate([num_scaled, cat_enc], axis=1).astype(np.float32)
        return X

    def predict(self, input_dict):
        # Prepare a single-row DataFrame with optimized data types
        df = pd.DataFrame([input_dict])
        
        # Use precomputed mapping for efficiency :cite[3]
        if 'deterministic_liters' not in df.columns:
            runoff_coeff = df.get('runoff_coefficient', None)
            if runoff_coeff.isnull().any():
                material = df['roof_material'].iloc[0]
                df['runoff_coefficient'] = self._runoff_mapping.get(material, 0.8)
            df['deterministic_liters'] = df['rooftop_area'] * df['annual_rainfall'] * df['runoff_coefficient']

        deterministic = float(df['deterministic_liters'].iloc[0])

        # ML prediction of residual
        if self.model is None:
            return {
                'deterministic_liters': deterministic,
                'ml_adjusted_liters': deterministic,
                'residual_pred': 0.0,
                'used_ml': False
            }

        X = self._prepare_input(df)
        
        # Use float32 DMatrix for prediction :cite[5]
        dmatrix = xgb.DMatrix(X.astype(np.float32))
        residual_pred = float(self.model.predict(dmatrix)[0])
        adjusted = deterministic + residual_pred

        return {
            'deterministic_liters': deterministic,
            'residual_pred': residual_pred,
            'ml_adjusted_liters': adjusted,
            'used_ml': True
        }

def train_and_save_model(csv_path="synthetic_dataset.csv", n_estimators=300):  # Reduced from 500
    print("Loading dataset:", csv_path)
    
    # Load data with optimized data types :cite[5]
    dtype_spec = {
        'rooftop_area': np.float32,
        'annual_rainfall': np.float32,
        'runoff_coefficient': np.float32,
        'slope': np.float32,
        'drainage_density': np.float32,
        'aquifer_depth': np.float32,
        'dwellers': np.int32,
        'roof_material': 'category',
        'soil_type': 'category',
        'observed_liters': np.float32
    }
    
    df = pd.read_csv(csv_path, dtype=dtype_spec)

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

    # Prepare categorical - use existing optimized categories
    df[categorical_cols] = df[categorical_cols].fillna('unknown').astype(str)

    # Split data first to minimize memory during transformation
    X_num = df[numerical_cols].values.astype(np.float32)
    X_cat = df[categorical_cols].values
    y = df['residual'].values.astype(np.float32)

    # Free memory by deleting original dataframe early
    del df

    # Fit scaler and encoder on training data only
    X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test = train_test_split(
        X_num, X_cat, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Transform data in chunks to reduce memory peak
    scaler = StandardScaler().fit(X_train_num)
    X_train_num_scaled = scaler.transform(X_train_num).astype(np.float32)
    X_test_num_scaled = scaler.transform(X_test_num).astype(np.float32)

    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore').fit(X_train_cat)  # Use sparse to save memory :cite[5]
    X_train_cat_enc = encoder.transform(X_train_cat)
    X_test_cat_enc = encoder.transform(X_test_cat)

    # Use hstack for sparse matrices
    from scipy import sparse
    X_train = sparse.hstack([X_train_num_scaled, X_train_cat_enc]).tocsr()
    X_test = sparse.hstack([X_test_num_scaled, X_test_cat_enc]).tocsr()

    # Train XGBoost with memory-optimized parameters :cite[5]:cite[8]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Optimized parameters for memory usage :cite[5]:cite[9]
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',  # hist is more memory efficient :cite[5]
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,  # Use subset of data per tree :cite[8]
        'colsample_bytree': 0.8,  # Use subset of features per tree :cite[8]
        'random_state': 42,
        'max_bin': 128,  # Reduced from default 256 to save memory :cite[5]
        'min_child_weight': 3,  # Increased to create simpler trees :cite[5]
    }
    
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    print("Training XGBoost with memory optimization...")
    
    # Use early stopping to avoid unnecessary trees :cite[2]
    bst = xgb.train(params, dtrain, num_boost_round=n_estimators, 
                   evals=watchlist, early_stopping_rounds=20, verbose_eval=50)  # Reduced early stopping rounds

    # Evaluate
    preds = bst.predict(dtest)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"Evaluation on test set -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

    # Save model, scaler, encoder
    model_path = os.path.join(MODEL_DIR, "residual_xgb.joblib")
    joblib.dump(bst, model_path)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))
    print("Saved model and preprocessors to", MODEL_DIR)

    # Clean up large variables
    del X_train, X_test, dtrain, dtest
    
    # Print feature importance
    try:
        imp = bst.get_score(importance_type='gain')
        sorted_imp = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
        print("Top feature importances (gain):")
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
    
    # Load model only if paths exist
    if os.path.exists(model_path):
        predictor.model = joblib.load(model_path)
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
    parser.add_argument('--n_estimators', type=int, default=300, help="XGBoost n_estimators (reduced for memory)")
    args = parser.parse_args()
    main(args)