# import sys
# import json
# import traceback
# from pathlib import Path
# import concurrent.futures
# from typing import Dict, Any, Tuple, Optional, Callable
# import random  # <-- 1. IMPORT ADDED
# import os

# # ---- Flask-related imports ----
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from flask import render_template   


# # ---- Constants and Configuration ----
# ROOT = Path(__file__).resolve().parent.parent
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# BACKEND_DIR = ROOT / "backend"
# ML_DIR = ROOT / "ML"
# DATASETS_DIR = ROOT / "datasets"

# # ---- Module Imports ----
# try:
#     from backend.drainage_density import compute_drainage_density
#     from backend.get_rainfall import get_rainfall_data
#     from backend.get_soil import get_soil_type
#     from backend.runoff_coeff import get_runoff_coefficient_strict
#     from ML.main import RTRWHPredictor
# except ImportError as e:
#     print(f"FATAL: A required module could not be imported. Please check your installation. Details: {e}", file=sys.stderr)
#     sys.exit(1)

# # ---- Default Inputs ----
# DEFAULTS = {
#     "rooftop_area": 90.0, "dwellers": 40, "roof_material": "metal",
#     "longitude": 88.3639, "latitude": 22.5726,
#     "model_dir": str(ML_DIR)
# }

# # --- Helper Functions (Unchanged) ---
# def try_load_ml_predictor(model_dir: Path) -> Tuple[Optional[Any], str]:
#     import joblib
#     if not model_dir.is_dir():
#         return None, f"ML model directory not found: {model_dir}"
#     predictor = RTRWHPredictor()
#     expected_artifacts = {"encoder": "encoder.pkl", "scaler": "scaler.pkl", "target_scaler": "target_scaler.pkl", "model": "model.pth"}
#     missing_files = [name for name, filename in expected_artifacts.items() if not (model_dir / filename).exists()]
#     if missing_files:
#         return None, f"ML artifacts missing: {', '.join(missing_files)}."
#     try:
#         predictor.encoder = joblib.load(model_dir / expected_artifacts["encoder"])
#         predictor.scaler = joblib.load(model_dir / expected_artifacts["scaler"])
#         predictor.target_scaler = joblib.load(model_dir / expected_artifacts["target_scaler"])
#         predictor.load_model(str(model_dir))
#         return predictor, "ML predictor loaded successfully."
#     except Exception:
#         return None, f"Failed to load ML artifacts: {traceback.format_exc()}"

# def run_task(func: Callable, key: str, **kwargs) -> Dict[str, Any]:
#     try:
#         return {key: func(**kwargs)}
#     except Exception as e:
#         return {f"{key}_error": str(e), f"{key}_error_trace": traceback.format_exc()}

# # --- Core Logic Refactored for Web Server ---
# def run_analysis(request_data: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Main orchestrator for the AquaSpatial analysis.
#     """
#     inputs = {
#         "rooftop_area": request_data.get("rooftop_area", DEFAULTS["rooftop_area"]),
#         "dwellers": request_data.get("dwellers", DEFAULTS["dwellers"]),
#         "roof_material": request_data.get("roof_material", DEFAULTS["roof_material"]).strip().lower(),
#         "longitude": request_data.get("longitude", DEFAULTS["longitude"]),
#         "latitude": request_data.get("latitude", DEFAULTS["latitude"]),
#         "model_dir": Path(DEFAULTS["model_dir"])
#     }

#     results: Dict[str, Any] = {"inputs": inputs.copy(), "outputs": {}}

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [
#             executor.submit(run_task, compute_drainage_density, "drainage_density_km_per_km2", lat=inputs["latitude"], lon=inputs["longitude"], verbose=False),
#             executor.submit(run_task, get_rainfall_data, "annual_rainfall_mm_total", lat=inputs["latitude"], lon=inputs["longitude"]),
#             executor.submit(run_task, get_soil_type, "soil_description", lat=inputs["latitude"], lon=inputs["longitude"]),
#             executor.submit(run_task, get_runoff_coefficient_strict, "runoff_coefficient", roof_type=inputs["roof_material"]),
#             executor.submit(run_task, try_load_ml_predictor, "ml_predictor_status", model_dir=inputs["model_dir"])
#         ]
#         for future in concurrent.futures.as_completed(futures):
#             results["outputs"].update(future.result())

#     predictor, ml_msg = results["outputs"].pop("ml_predictor_status", (None, "Status unknown."))
#     results["outputs"]["ml_load_status"] = ml_msg

#     # ---- START OF THE FIX ----
#     # Check if rainfall data is null and assign a random fallback value if it is.
#     rainfall = results["outputs"].get("annual_rainfall_mm_total")
#     if rainfall is None:
#         print("WARNING: Rainfall data is null. Assigning a random fallback value between 650 and 900.")
#         fallback_rainfall = random.uniform(650, 900)
#         rainfall = fallback_rainfall
        
#         # Update the results dictionary so the frontend receives the random value (rounded for clarity)
#         results["outputs"]["annual_rainfall_mm_total"] = round(rainfall, 2)
#         # Add a status field to inform the frontend that the data is an estimate
#         results["outputs"]["rainfall_data_status"] = "Estimated value (actual data not available)"
#     else:
#         # Add the status field even on success for consistency
#         results["outputs"]["rainfall_data_status"] = "Actual data received"
#     # ---- END OF THE FIX ----

#     if predictor:
#         try:
#             # Robustly get other values, providing a default if the value is None or missing.
#             soil = results["outputs"].get("soil_description")
#             drainage = results["outputs"].get("drainage_density_km_per_km2")
#             runoff = results["outputs"].get("runoff_coefficient")

#             sample_input = {
#                 'rooftop_area': inputs["rooftop_area"],
#                 'dwellers': inputs["dwellers"],
#                 'roof_material': inputs["roof_material"],
#                 'annual_rainfall': rainfall,  # This will now be either the real value or the random one
#                 'aquifer_depth': 5.0,
#                 'soil_type': soil if soil is not None else "Unknown",
#                 'slope': 1.0,
#                 'drainage_density': drainage if drainage is not None else 0.0,
#                 'runoff_coefficient': runoff if runoff is not None else 0.0
#             }
            
#             results["outputs"]["ml_prediction"] = predictor.predict(sample_input)
#         except Exception as e:
#             results["outputs"]["ml_prediction_error"] = str(e)
#     else:
#         results["outputs"]["ml_prediction"] = "ML model not loaded or failed to load."

#     serializable_results = json.loads(json.dumps(results, default=str))
#     out_file = ROOT / "last_run_result.json"
#     try:
#         with out_file.open("w", encoding="utf-8") as f:
#             json.dump(serializable_results, f, indent=4)
#         print(f"Saved run summary to: {out_file}")
#     except Exception as e:
#         print(f"Warning: Failed to save run summary. Reason: {e}", file=sys.stderr)
        
#     return serializable_results

# # --- Flask App Definition ---
# from flask import send_from_directory
# frontend_dir = (ROOT / "frontend").resolve()
# static_dir = (frontend_dir / "static").resolve()
# app = Flask(
#     __name__,
#     template_folder=str(frontend_dir),
#     static_folder=str(static_dir)
# )
# CORS(app)

# @app.route("/")
# def home():
#     # Serve index.html from the absolute frontend directory
#     return render_template("index.html")

# @app.route('/analyze', methods=['POST'])
# def analyze_endpoint():
#     print("Received a request on /analyze")
#     if not request.is_json:
#         return jsonify({"error": "Request must be JSON"}), 400
#     request_data = request.get_json()
#     try:
#         results = run_analysis(request_data)
#         return jsonify(results)
#     except Exception as e:
#         print(f"An error occurred during analysis: {e}\n{traceback.format_exc()}", file=sys.stderr)
#         return jsonify({"error": "An internal server error occurred."}), 500

# # # --- Main Execution Block ---
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)

# backend/app.py
import sys
import json
import traceback
from pathlib import Path
import concurrent.futures
from typing import Dict, Any, Tuple, Optional, Callable
import random
import os

# ---- Flask-related imports ----
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ---- Constants and Configuration ----
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BACKEND_DIR = ROOT / "backend"
ML_DIR = ROOT / "ML"
MODEL_ARTIFACT_DIR = ML_DIR / "models"
DATASETS_DIR = ROOT / "datasets"

# ---- Module Imports ----
try:
    from backend.drainage_density import compute_drainage_density
    from backend.get_rainfall import get_rainfall_data
    from backend.get_soil import get_soil_type
    from backend.runoff_coeff import get_runoff_coefficient_strict
    # Import Predictor (the class in ML/main.py). We'll alias it as RTRWHPredictor for backwards compatibility.
    from ML.main import Predictor as RTRWHPredictor
except ImportError as e:
    print(f"FATAL: A required module could not be imported. Please check your installation. Details: {e}", file=sys.stderr)
    sys.exit(1)

# ---- Default Inputs ----
DEFAULTS = {
    "rooftop_area": 90.0,
    "dwellers": 40,
    "roof_material": "metal",
    "longitude": 88.3639,
    "latitude": 22.5726,
    "model_dir": str(MODEL_ARTIFACT_DIR)
}

# --- Helper Functions ---
def try_load_ml_predictor(model_dir: Path) -> Tuple[Optional[Any], str]:
    """
    Load artifacts produced by ML/main.py training:
      - residual_xgb.joblib  (XGBoost booster saved via joblib)
      - scaler.pkl
      - encoder.pkl
    Returns (predictor_instance, message)
    """
    import joblib

    if not model_dir.exists():
        return None, f"ML model directory not found: {model_dir}"

    predictor = RTRWHPredictor()

    expected_files = {
        "model": "residual_xgb.joblib",
        "scaler": "scaler.pkl",
        "encoder": "encoder.pkl"
    }

    missing = [v for v in expected_files.values() if not (model_dir / v).exists()]
    if missing:
        return None, f"ML artifacts missing in {model_dir}: {', '.join(missing)}"

    try:
        predictor.model = joblib.load(str(model_dir / expected_files["model"]))
        predictor.scaler = joblib.load(str(model_dir / expected_files["scaler"]))
        predictor.encoder = joblib.load(str(model_dir / expected_files["encoder"]))
        return predictor, "ML predictor loaded successfully."
    except Exception:
        return None, f"Failed to load ML artifacts: {traceback.format_exc()}"

def run_task(func: Callable, key: str, **kwargs) -> Dict[str, Any]:
    try:
        return {key: func(**kwargs)}
    except Exception as e:
        return {f"{key}_error": str(e), f"{key}_error_trace": traceback.format_exc()}

# --- Core Logic Refactored for Web Server ---
def run_analysis(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main orchestrator for the AquaSpatial analysis.
    """
    inputs = {
        "rooftop_area": request_data.get("rooftop_area", DEFAULTS["rooftop_area"]),
        "dwellers": request_data.get("dwellers", DEFAULTS["dwellers"]),
        "roof_material": request_data.get("roof_material", DEFAULTS["roof_material"]).strip().lower(),
        "longitude": request_data.get("longitude", DEFAULTS["longitude"]),
        "latitude": request_data.get("latitude", DEFAULTS["latitude"]),
        "model_dir": Path(request_data.get("model_dir", DEFAULTS["model_dir"]))
    }

    results: Dict[str, Any] = {"inputs": inputs.copy(), "outputs": {}}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_task, compute_drainage_density, "drainage_density_km_per_km2", lat=inputs["latitude"], lon=inputs["longitude"], verbose=False),
            executor.submit(run_task, get_rainfall_data, "annual_rainfall_mm_total", lat=inputs["latitude"], lon=inputs["longitude"]),
            executor.submit(run_task, get_soil_type, "soil_description", lat=inputs["latitude"], lon=inputs["longitude"]),
            executor.submit(run_task, get_runoff_coefficient_strict, "runoff_coefficient", roof_type=inputs["roof_material"]),
            executor.submit(run_task, try_load_ml_predictor, "ml_predictor_status", model_dir=inputs["model_dir"])
        ]
        for future in concurrent.futures.as_completed(futures):
            results["outputs"].update(future.result())

    predictor, ml_msg = results["outputs"].pop("ml_predictor_status", (None, "Status unknown."))
    results["outputs"]["ml_load_status"] = ml_msg

    # Rainfall fallback
    rainfall = results["outputs"].get("annual_rainfall_mm_total")
    if rainfall is None:
        print("WARNING: Rainfall data is null. Assigning a fallback value between 650 and 900.")
        fallback_rainfall = random.uniform(650, 900)
        rainfall = fallback_rainfall
        results["outputs"]["annual_rainfall_mm_total"] = round(rainfall, 2)
        results["outputs"]["rainfall_data_status"] = "Estimated value (actual data not available)"
    else:
        results["outputs"]["rainfall_data_status"] = "Actual data received"

    # Build sample input for predictor & deterministic calculations
    # Use defaults if any value missing
    rooftop_area = float(inputs.get("rooftop_area") or DEFAULTS["rooftop_area"])
    dwellers = int(inputs.get("dwellers") or DEFAULTS["dwellers"])
    roof_material = inputs.get("roof_material") or DEFAULTS["roof_material"]
    annual_rainfall = float(rainfall)
    aquifer_depth = float(request_data.get("aquifer_depth", 5.0))
    soil_type = results["outputs"].get("soil_description") or request_data.get("soil_type", "unknown")
    slope = float(request_data.get("slope", 1.0))
    drainage_density = float(results["outputs"].get("drainage_density_km_per_km2") or request_data.get("drainage_density", 0.0))
    runoff_coefficient = float(results["outputs"].get("runoff_coefficient") or request_data.get("runoff_coefficient", 0.0))

    # If runoff coefficient still zero / missing, pick a default by roof material
    if runoff_coefficient == 0.0:
        mapping = {'metal':0.95,'concrete':0.9,'tiles':0.8,'asphalt':0.88,'thatch':0.6}
        runoff_coefficient = mapping.get(roof_material, 0.8)

    sample_input = {
        'rooftop_area': rooftop_area,
        'dwellers': dwellers,
        'roof_material': roof_material,
        'annual_rainfall': annual_rainfall,
        'aquifer_depth': aquifer_depth,
        'soil_type': soil_type,
        'slope': slope,
        'drainage_density': drainage_density,
        'runoff_coefficient': runoff_coefficient
    }

    # Call predictor if available
    if predictor:
        try:
            pred_raw = predictor.predict(sample_input)
        except Exception as e:
            results["outputs"]["ml_prediction_error"] = f"Predictor failure: {e}"
            pred_raw = None
    else:
        pred_raw = None

    # Determine liters/year to use (prefer ML-adjusted if available)
    if pred_raw and isinstance(pred_raw, dict):
        deterministic_liters = float(pred_raw.get('deterministic_liters', rooftop_area * annual_rainfall * runoff_coefficient))
        ml_adjusted_liters = float(pred_raw.get('ml_adjusted_liters', deterministic_liters))
        residual_pred = float(pred_raw.get('residual_pred', ml_adjusted_liters - deterministic_liters))
        used_ml = bool(pred_raw.get('used_ml', False))
    else:
        deterministic_liters = rooftop_area * annual_rainfall * runoff_coefficient
        ml_adjusted_liters = deterministic_liters
        residual_pred = 0.0
        used_ml = False

    # Compute outputs expected by frontend
    # recharge_potential in liters/year (use ML-adjusted liters)
    recharge_potential = float(ml_adjusted_liters)

    # harvested_m3 (cubic meters/year)
    harvested_m3 = recharge_potential / 1000.0

    # demand (m3/year) use 135 L/person/day as in dataset.py (tunable)
    liters_per_person_per_day = 135.0
    annual_demand_liters = dwellers * liters_per_person_per_day * 365.0
    annual_demand_m3 = annual_demand_liters / 1000.0

    # harvest_demand_ratio
    harvest_demand_ratio = float(harvested_m3 / annual_demand_m3) if annual_demand_m3 > 0 else 0.0

    # Suitability score heuristic (0..1) — similar to dataset.py but using local normalization
    # Use a simple normalization cap for harvest_demand_ratio (assume >=2 is saturating)
    harvest_norm = min(harvest_demand_ratio / 2.0, 1.0)  # scale so 2.0 maps to 1.0
    slope_norm = min(max(slope / 45.0, 0.0), 1.0)  # 0..1
    soil_score_map = {'sandy':0.9, 'loamy':0.9, 'clay':0.7, 'rocky':0.4, 'unknown':0.7}
    soil_score = float(soil_score_map.get(str(soil_type).lower(), 0.7))
    suitability_score = 0.6 * harvest_norm + 0.25 * (1.0 - slope_norm) + 0.15 * soil_score
    suitability_score = float(max(0.0, min(1.0, suitability_score)))

    # Cost estimation (same formula as dataset.py)
    base_cost = 5000.0
    storage_cost_per_m3 = 1200.0
    installation_rate_per_m2 = 100.0
    soil_multiplier_map = {'clay': 1.15, 'sandy': 1.0, 'loamy': 1.08, 'rocky': 1.25, 'unknown': 1.0}
    soil_multiplier = float(soil_multiplier_map.get(str(soil_type).lower(), 1.0))
    desired_storage_m3 = harvested_m3 * 0.5
    cost_estimation = base_cost + (rooftop_area * installation_rate_per_m2 * soil_multiplier) + (desired_storage_m3 * storage_cost_per_m3)
    cost_estimation = float(round(cost_estimation, 2))

    # Compose a clean, frontend-friendly ml_prediction object
    ml_prediction = {
        # Primary values frontend expects
        "recharge_potential": round(recharge_potential, 2),  # liters/year
        "suitability_score": round(suitability_score, 4),    # 0..1
        "harvest_demand_ratio": round(harvest_demand_ratio, 3),
        "cost_estimation": round(cost_estimation, 2),

        # also keep raw/residual fields for debugging
        "deterministic_liters": round(deterministic_liters, 2),
        "ml_adjusted_liters": round(ml_adjusted_liters, 2),
        "residual_pred": round(residual_pred, 2),
        "used_ml": used_ml
    }

    results["outputs"]["ml_prediction"] = ml_prediction

    serializable_results = json.loads(json.dumps(results, default=str))
    out_file = ROOT / "last_run_result.json"
    try:
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=4)
        print(f"Saved run summary to: {out_file}")
    except Exception as e:
        print(f"Warning: Failed to save run summary. Reason: {e}", file=sys.stderr)

    return serializable_results

# --- Flask App Definition ---
from flask import send_from_directory
frontend_dir = (ROOT / "frontend").resolve()
static_dir = (frontend_dir / "static").resolve()
app = Flask(
    __name__,
    template_folder=str(frontend_dir),
    static_folder=str(static_dir)
)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    print("Received a request on /analyze")
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    request_data = request.get_json()
    try:
        results = run_analysis(request_data)
        return jsonify(results)
    except Exception as e:
        print(f"An error occurred during analysis: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
