# # app.py - Corrected and Robust Flask Backend
# import sys
# import json
# import traceback
# from pathlib import Path
# import concurrent.futures
# from typing import Dict, Any, Tuple, Optional, Callable

# # ---- Flask-related imports ----
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# # ---- Constants and Configuration ----
# # Corrected typo from _file_ to __file__
# ROOT = Path(__file__).resolve().parent.parent
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# BACKEND_DIR = ROOT / "backend"
# ML_DIR = ROOT / "ML"
# DATASETS_DIR = ROOT / "datasets"

# # ---- Module Imports ----
# try:
#     from drainage_density import compute_drainage_density
#     from get_rainfall import get_rainfall_data
#     from get_soil import get_soil_type
#     from runoff_coeff import get_runoff_coefficient_strict
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

# # --- Helper Functions (Unchanged from your version) ---
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

#     if predictor:
#         try:
#             # ---- START OF THE FIX ----
#             # Robustly get values from the results, providing a default if the value is None or missing.
            
#             rainfall = results["outputs"].get("annual_rainfall_mm_total")
#             soil = results["outputs"].get("soil_description")
#             drainage = results["outputs"].get("drainage_density_km_per_km2")
#             runoff = results["outputs"].get("runoff_coefficient")

#             sample_input = {
#                 'rooftop_area': inputs["rooftop_area"],
#                 'dwellers': inputs["dwellers"],
#                 'roof_material': inputs["roof_material"],
#                 'annual_rainfall': rainfall if rainfall is not None else 0.0,
#                 'aquifer_depth': 5.0,  # Default value
#                 'soil_type': soil if soil is not None else "Unknown",
#                 'slope': 1.0,  # Default value
#                 'drainage_density': drainage if drainage is not None else 0.0,
#                 'runoff_coefficient': runoff if runoff is not None else 0.0
#             }
#             # ---- END OF THE FIX ----
            
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
# app = Flask(__name__)
# CORS(app)

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

# # --- Main Execution Block ---
# # Corrected typo from _name_ == "_main_"
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000, debug=True)

# app.py - Corrected and Robust Flask Backend with Rainfall Fallback
import sys
import json
import traceback
from pathlib import Path
import concurrent.futures
from typing import Dict, Any, Tuple, Optional, Callable
import random  # <-- 1. IMPORT ADDED

# ---- Flask-related imports ----
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---- Constants and Configuration ----
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BACKEND_DIR = ROOT / "backend"
ML_DIR = ROOT / "ML"
DATASETS_DIR = ROOT / "datasets"

# ---- Module Imports ----
try:
    from drainage_density import compute_drainage_density
    from get_rainfall import get_rainfall_data
    from get_soil import get_soil_type
    from runoff_coeff import get_runoff_coefficient_strict
    from ML.main import RTRWHPredictor
except ImportError as e:
    print(f"FATAL: A required module could not be imported. Please check your installation. Details: {e}", file=sys.stderr)
    sys.exit(1)

# ---- Default Inputs ----
DEFAULTS = {
    "rooftop_area": 90.0, "dwellers": 40, "roof_material": "metal",
    "longitude": 88.3639, "latitude": 22.5726,
    "model_dir": str(ML_DIR)
}

# --- Helper Functions (Unchanged) ---
def try_load_ml_predictor(model_dir: Path) -> Tuple[Optional[Any], str]:
    import joblib
    if not model_dir.is_dir():
        return None, f"ML model directory not found: {model_dir}"
    predictor = RTRWHPredictor()
    expected_artifacts = {"encoder": "encoder.pkl", "scaler": "scaler.pkl", "target_scaler": "target_scaler.pkl", "model": "model.pth"}
    missing_files = [name for name, filename in expected_artifacts.items() if not (model_dir / filename).exists()]
    if missing_files:
        return None, f"ML artifacts missing: {', '.join(missing_files)}."
    try:
        predictor.encoder = joblib.load(model_dir / expected_artifacts["encoder"])
        predictor.scaler = joblib.load(model_dir / expected_artifacts["scaler"])
        predictor.target_scaler = joblib.load(model_dir / expected_artifacts["target_scaler"])
        predictor.load_model(str(model_dir))
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
        "model_dir": Path(DEFAULTS["model_dir"])
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

    # ---- START OF THE FIX ----
    # Check if rainfall data is null and assign a random fallback value if it is.
    rainfall = results["outputs"].get("annual_rainfall_mm_total")
    if rainfall is None:
        print("WARNING: Rainfall data is null. Assigning a random fallback value between 650 and 900.")
        fallback_rainfall = random.uniform(650, 900)
        rainfall = fallback_rainfall
        
        # Update the results dictionary so the frontend receives the random value (rounded for clarity)
        results["outputs"]["annual_rainfall_mm_total"] = round(rainfall, 2)
        # Add a status field to inform the frontend that the data is an estimate
        results["outputs"]["rainfall_data_status"] = "Estimated value (actual data not available)"
    else:
        # Add the status field even on success for consistency
        results["outputs"]["rainfall_data_status"] = "Actual data received"
    # ---- END OF THE FIX ----

    if predictor:
        try:
            # Robustly get other values, providing a default if the value is None or missing.
            soil = results["outputs"].get("soil_description")
            drainage = results["outputs"].get("drainage_density_km_per_km2")
            runoff = results["outputs"].get("runoff_coefficient")

            sample_input = {
                'rooftop_area': inputs["rooftop_area"],
                'dwellers': inputs["dwellers"],
                'roof_material': inputs["roof_material"],
                'annual_rainfall': rainfall,  # This will now be either the real value or the random one
                'aquifer_depth': 5.0,
                'soil_type': soil if soil is not None else "Unknown",
                'slope': 1.0,
                'drainage_density': drainage if drainage is not None else 0.0,
                'runoff_coefficient': runoff if runoff is not None else 0.0
            }
            
            results["outputs"]["ml_prediction"] = predictor.predict(sample_input)
        except Exception as e:
            results["outputs"]["ml_prediction_error"] = str(e)
    else:
        results["outputs"]["ml_prediction"] = "ML model not loaded or failed to load."

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
app = Flask(__name__)
CORS(app)

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

# --- Main Execution Block ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)