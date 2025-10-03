import sys
import json
import traceback
from pathlib import Path
import concurrent.futures
from typing import Dict, Any, Tuple, Optional
from functools import lru_cache

# --- Flask and CORS Setup ---
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---- Constants and Configuration ----
# Adjust these paths if your directory structure is different
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BACKEND_DIR = ROOT / "backend"
ML_DIR = ROOT / "ML"
DATASETS_DIR = ROOT / "datasets"

# Add backend and ML directories to the path to ensure imports work
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ML_DIR))

# ---- Module Imports ----
try:
    from backend.drainage_density import compute_drainage_density as original_compute_drainage_density
    from backend.get_rainfall import get_rainfall_data as original_get_rainfall_data
    from backend.get_soil import get_soil_type as original_get_soil_type
    from backend.runoff_coeff import get_runoff_coefficient_strict as original_get_runoff_coefficient_strict
    from ML.main import RTRWHPredictor
    import joblib
except ImportError as e:
    print(f"FATAL: A required module could not be imported. Please check your installation and paths. Details: {e}", file=sys.stderr)
    sys.exit(1)

@lru_cache(maxsize=1000)
def compute_drainage_density(lat: float, lon: float, verbose: bool = False) -> Any:
    return original_compute_drainage_density(lat=lat, lon=lon, verbose=verbose)

@lru_cache(maxsize=1000)
def get_rainfall_data(lat: float, lon: float) -> Any:
    return original_get_rainfall_data(lat=lat, lon=lon)

@lru_cache(maxsize=1000)
def get_soil_type(lat: float, lon: float) -> Any:
    return original_get_soil_type(lat=lat, lon=lon)

@lru_cache(maxsize=1000)
def get_runoff_coefficient_strict(roof_type: str) -> Any:
    return original_get_runoff_coefficient_strict(roof_type=roof_type)

# ---- Default Paths & Global Predictor ----
SOIL_TIF_PATH = str(DATASETS_DIR / "SOILTEXTURE.tif")
MODEL_DIR = Path(ML_DIR)
ML_PREDICTOR: Optional[RTRWHPredictor] = None
ML_LOAD_ERROR: Optional[str] = None

def load_ml_predictor_on_startup():
    """Loads the ML model once when the server starts."""
    global ML_PREDICTOR, ML_LOAD_ERROR
    print("--- Loading ML Model Artifacts on Startup ---")
    
    try:
        if not MODEL_DIR.is_dir():
            raise FileNotFoundError(f"ML model directory not found: {MODEL_DIR}")

        predictor = RTRWHPredictor()
        expected_artifacts = {"encoder": "encoder.pkl", "scaler": "scaler.pkl", "target_scaler": "target_scaler.pkl", "model": "model.pth"}
        
        missing_files = [name for name, filename in expected_artifacts.items() if not (MODEL_DIR / filename).exists()]
        if missing_files:
            raise FileNotFoundError(f"ML artifacts missing: {', '.join(missing_files)}.")

        predictor.encoder = joblib.load(MODEL_DIR / expected_artifacts["encoder"])
        predictor.scaler = joblib.load(MODEL_DIR / expected_artifacts["scaler"])
        predictor.target_scaler = joblib.load(MODEL_DIR / expected_artifacts["target_scaler"])
        predictor.load_model(str(MODEL_DIR))
        ML_PREDICTOR = predictor
        print("--- ML Predictor Loaded Successfully ---")
    except Exception as e:
        ML_LOAD_ERROR = f"Failed to load ML artifacts on startup: {traceback.format_exc()}"
        print(f"FATAL: {ML_LOAD_ERROR}", file=sys.stderr)

def run_task(func, key, **kwargs):
    """Helper to run a function and capture its result or error."""
    try:
        return {key: func(**kwargs)}
    except Exception as e:
        return {f"{key}_error": str(e)}

def run_aqua_spatial_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates the analysis by running tasks sequentially and then the ML prediction.
    This function is adapted from your original `main` function.
    """
    results: Dict[str, Any] = {}

    # 1. Run I/O-Bound Tasks Sequentially
    results.update(run_task(compute_drainage_density, "drainage_density_km_per_km2", lat=data["latitude"], lon=data["longitude"], verbose=False))
    results.update(run_task(get_rainfall_data, "annual_rainfall_mm_total", lat=data["latitude"], lon=data["longitude"]))
    results.update(run_task(get_soil_type, "soil_description", lat=data["latitude"], lon=data["longitude"]))
    results.update(run_task(get_runoff_coefficient_strict, "runoff_coefficient", roof_type=data["roof_material"]))

    # 2. Run CPU-Bound ML Prediction
    if ML_PREDICTOR:
        try:
            sample_input = {
                'rooftop_area': data["rooftop_area"], 'dwellers': data["dwellers"],
                'roof_material': data["roof_material"],
                'annual_rainfall': results.get("annual_rainfall_mm_total", 0.0),
                'aquifer_depth': 5.0,  # Using a default value as it's not in the form
                'soil_type': results.get("soil_description", "Unknown"),
                'slope': 1.0,  # Using a default value
                'drainage_density': results.get("drainage_density_km_per_km2", 0.0),
                'runoff_coefficient': results.get("runoff_coefficient", 0.0)
            }
            prediction_output = ML_PREDICTOR.predict(sample_input)
            results["ml_prediction"] = prediction_output
        except Exception as e:
            results["ml_prediction_error"] = str(e)
    else:
        results["ml_prediction_error"] = f"ML model could not be used. Load error: {ML_LOAD_ERROR}"

    return results

# ---- Flask App ----
app = Flask(__name__)
CORS(app)  # This enables Cross-Origin Resource Sharing for your frontend

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to handle prediction requests."""
    print("Received a request on /predict")
    
    # 1. Get and validate data from the frontend
    try:
        data = request.get_json()
        print(f"Request JSON data: {data}")
        
        # Basic validation and type casting
        inputs = {
            "rooftop_area": float(data['rooftop_area']),
            "dwellers": int(data['dwellers']),
            "roof_material": str(data['roof_material']).strip().lower(),
            "latitude": float(data['latitude']),
            "longitude": float(data['longitude'])
        }
    except (TypeError, KeyError, ValueError) as e:
        print(f"Error processing input data: {e}")
        return jsonify({"error": f"Invalid or missing input data: {e}"}), 400

    # 2. Run the analysis
    analysis_results = run_aqua_spatial_analysis(inputs)
    print(f"Analysis complete. Raw results: {analysis_results}")

    # 3. Format the response to match frontend expectations
    if "ml_prediction_error" in analysis_results or "ml_prediction" not in analysis_results:
        error_message = analysis_results.get("ml_prediction_error", "An unknown error occurred during ML prediction.")
        return jsonify({"error": error_message}), 500

    ml_results = analysis_results["ml_prediction"]
    
    # Assuming a static conversion rate for demonstration
    usd_to_inr_rate = 83.5 

    formatted_response = {
        "rechargePotential": (ml_results.get('recharge_potential_m3_per_year', 0) * 1000),
        "suitabilityScore": ml_results.get('suitability_score', 0),
        "harvestDemandRatio": ml_results.get('harvest_demand_ratio', 0),
        "costEstimation": (ml_results.get('cost_estimation_usd', 0) * usd_to_inr_rate),
        "latitude": inputs["latitude"],
        "longitude": inputs["longitude"]
    }
    
    print(f"Sending formatted response to frontend: {formatted_response}")
    return jsonify(formatted_response)

if __name__ == '__main__':
    load_ml_predictor_on_startup()  # Load the model before starting the server
    # Use host='0.0.0.0' to make it accessible on your local network
    app.run(debug=False, host='0.0.0.0')