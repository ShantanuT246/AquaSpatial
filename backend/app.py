# app.py - modular orchestrator for AquaSpatial (uses your backend/ and ML/ modules)
import sys
import json
import traceback
from pathlib import Path
import argparse
import concurrent.futures
from typing import Dict, Any, Tuple, Optional, Callable

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
    "soil_tif_path": str(DATASETS_DIR / "SOILTEXTURE.tif"),
    "model_dir": str(ML_DIR), "rain_year": 2023
}

def try_load_ml_predictor(model_dir: Path) -> Tuple[Optional[Any], str]:
    """Loads the ML model artifacts from a directory."""
    import joblib  # Import only when needed
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
    """A helper to run a function, capture its result or error."""
    try:
        return {key: func(**kwargs)}
    except Exception as e:
        return {f"{key}_error": str(e), f"{key}_error_trace": traceback.format_exc()}

def main(argv: Optional[list] = None) -> None:
    """Main orchestrator for the AquaSpatial analysis."""
    parser = argparse.ArgumentParser(description="AquaSpatial modular orchestrator.")
    parser.add_argument("--area", type=float, default=DEFAULTS["rooftop_area"])
    parser.add_argument("--dwellers", type=int, default=DEFAULTS["dwellers"])
    parser.add_argument("--roof", type=str, default=DEFAULTS["roof_material"])
    parser.add_argument("--lon", type=float, default=DEFAULTS["longitude"])
    parser.add_argument("--lat", type=float, default=DEFAULTS["latitude"])
    parser.add_argument("--model-dir", type=str, default=DEFAULTS["model_dir"])
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress informational output and print only the final JSON result.")
    args = parser.parse_args(argv)

    inputs = {
        "rooftop_area": args.area, "dwellers": args.dwellers,
        "roof_material": args.roof.strip().lower(), "longitude": args.lon, "latitude": args.lat,
        "model_dir": Path(args.model_dir)
    }

    results: Dict[str, Any] = {"inputs": inputs.copy(), "outputs": {}}
    
    # ---- 1. Run All Independent I/O-Bound Tasks in Parallel (including ML model loading) ----
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

    # ---- 2. Run CPU-Bound ML Prediction (dependent on previous results) ----
    predictor, ml_msg = results["outputs"].pop("ml_predictor_status", (None, "Status unknown."))
    results["outputs"]["ml_load_status"] = ml_msg

    if predictor:
        try:
            sample_input = {
                'rooftop_area': inputs["rooftop_area"], 'dwellers': inputs["dwellers"],
                'roof_material': inputs["roof_material"],
                'annual_rainfall': results["outputs"].get("annual_rainfall_mm_total", 0.0),
                'aquifer_depth': 5.0, 'soil_type': results["outputs"].get("soil_description", "Unknown"),
                'slope': 1.0, 'drainage_density': results["outputs"].get("drainage_density_km_per_km2", 0.0),
                'runoff_coefficient': results["outputs"].get("runoff_coefficient", 0.0)
            }
            results["outputs"]["ml_prediction"] = predictor.predict(sample_input)
        except Exception as e:
            results["outputs"]["ml_prediction_error"] = str(e)
    else:
        results["outputs"]["ml_prediction"] = "ML model not loaded or failed to load."

    # ---- 3. Report and Save Results ----
    # Convert Path objects to strings for clean serialization
    serializable_results = json.loads(json.dumps(results, default=str))
    
    if args.quiet:
        # In quiet mode, print only the final, clean JSON results.
        print(json.dumps(serializable_results, indent=2))
    else:
        # In normal mode, print the user-friendly summary.
        print("\n=== AquaSpatial Run Initiated ===")
        print(f"Working directory: {ROOT}\n")
        print("--- Run Summary ---")
        print("Inputs:")
        for key, value in serializable_results["inputs"].items():
            print(f"  {key:20s}: {value}")
        print("\nOutputs:")
        for key, value in serializable_results["outputs"].items():
            if "trace" not in key:
                print(f"  {key:20s}: {value}")
        print("-------------------\n")

    out_file = ROOT / "last_run_result.json"
    try:
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=4)
        if not args.quiet:
            print(f"Saved detailed run summary to: {out_file}")
    except Exception as e:
        print(f"Warning: Failed to save run summary. Reason: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

    # prakul update