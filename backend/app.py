# app.py - modular orchestrator for AquaSpatial (uses your backend/ and ML/ modules)
import sys
import os
import json
import traceback
from pathlib import Path
import argparse

# Ensure backend/ and ML/ are importable (adjust if your layout is different)
ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
ML_DIR = ROOT / "ML"

# Add to sys.path (if not already)
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

# ---- Import your modules (they exist in the repo as you provided) ----
# Use try/except to provide clear error messages if something is missing
missing = []

try:
    from drainage_density import compute_drainage_density
except Exception as e:
    compute_drainage_density = None
    missing.append(("drainage_density", str(e)))

try:
    from get_rainfall import get_rainfall_data
except Exception as e:
    get_rainfall_data = None
    missing.append(("get_rainfall", str(e)))

try:
    from get_soil import get_point_description, tif_file as SOIL_TIF
except Exception as e:
    get_point_description = None
    SOIL_TIF = None
    missing.append(("get_soil", str(e)))

try:
    from runoff_coeff import get_runoff_coefficient_strict
except Exception as e:
    get_runoff_coefficient_strict = None
    missing.append(("runoff_coeff", str(e)))

try:
    from main import RTRWHPredictor
except Exception as e:
    # try alternative import path
    try:
        from ML.main import RTRWHPredictor
    except Exception as e2:
        RTRWHPredictor = None
        missing.append(("ML.main / RTRWHPredictor", f"{e} / {e2}"))

# ---- Default inputs (change these inline as desired) ----
DEFAULTS = {
    "rooftop_area": 90.0,
    "dwellers": 2,
    "roof_material": "metal",   # must be one of: tile, metal, asphalt, concrete
    "longitude": 88.3639,
    "latitude": 22.5726,
    # path to your soil TIF (from get_soil.py you used datasets/SOILTEXTURE.tif)
    "soil_tif_path": str(ROOT / "datasets" / "SOILTEXTURE.tif"),
    # model dir for ML artifacts
    "model_dir": str(ROOT),
    # IMD rainfall year (your get_rainfall defaults to 2023)
    "rain_year": 2023
}

def try_load_ml_predictor(model_dir: str):
    """
    If RTRWHPredictor class is available, try to load saved model artifacts (scalers/encoder/model).
    Returns (predictor_instance_or_None, message)
    """
    if RTRWHPredictor is None:
        return None, "RTRWHPredictor class not importable (missing ML dependencies or wrong import)."

    predictor = RTRWHPredictor()
    # expected files per your ML/main.py save/load
    model_pth = os.path.join(model_dir, "model.pth")
    scaler_pkl = os.path.join(model_dir, "scaler.pkl")
    encoder_pkl = os.path.join(model_dir, "encoder.pkl")
    target_pkl = os.path.join(model_dir, "target_scaler.pkl")

    missing_files = [p for p in [model_pth, scaler_pkl, encoder_pkl, target_pkl] if not os.path.exists(p)]
    if missing_files:
        return predictor, f"ML artifacts missing: {', '.join([os.path.basename(p) for p in missing_files])}. Predictor created but not loaded."

    # attempt load
    try:
        # load encoder & scalers first (your load_model expects encoder exists)
        import joblib
        predictor.encoder = joblib.load(encoder_pkl)
        predictor.scaler = joblib.load(scaler_pkl)
        predictor.target_scaler = joblib.load(target_pkl)
        predictor.load_model(model_dir)  # this should load model.pth into predictor.model
        return predictor, "ML predictor loaded successfully."
    except Exception as e:
        return predictor, f"Failed to load ML artifacts: {e}"

def main(argv=None):
    parser = argparse.ArgumentParser(description="AquaSpatial modular orchestrator (uses backend/ and ML/).")
    parser.add_argument("--area", type=float, default=DEFAULTS["rooftop_area"])
    parser.add_argument("--dwellers", type=int, default=DEFAULTS["dwellers"])
    parser.add_argument("--roof", type=str, default=DEFAULTS["roof_material"])
    parser.add_argument("--lon", type=float, default=DEFAULTS["longitude"])
    parser.add_argument("--lat", type=float, default=DEFAULTS["latitude"])
    parser.add_argument("--soil-tif", type=str, default=DEFAULTS["soil_tif_path"])
    parser.add_argument("--model-dir", type=str, default=DEFAULTS["model_dir"])
    parser.add_argument("--rain-year", type=int, default=DEFAULTS["rain_year"])
    args = parser.parse_args(argv)

    # Print initial status
    print("\nAquaSpatial modular runner\nWorking directory:", ROOT)
    if missing:
        print("NOTE: the following modules failed to import (functionality may be limited):")
        for name, error in missing:
            print("  -", name, "->", error)
    print()

    # Build inputs dict
    inputs = {
        "rooftop_area": args.area,
        "dwellers": args.dwellers,
        "roof_material": args.roof.strip().lower(),
        "longitude": args.lon,
        "latitude": args.lat,
        "soil_tif": args.soil_tif,
        "model_dir": args.model_dir,
        "rain_year": args.rain_year
    }

    results = {"inputs": inputs.copy(), "outputs": {}}

    # 1) Drainage density
    try:
        if compute_drainage_density is None:
            raise RuntimeError("compute_drainage_density not importable")
        dd_out = compute_drainage_density(inputs["latitude"], inputs["longitude"], verbose=False)
        # some earlier versions returned tuple so handle both
        if isinstance(dd_out, (list, tuple)):
            dd_val = float(dd_out[0])
        else:
            dd_val = float(dd_out)
        results["outputs"]["drainage_density_km_per_km2"] = dd_val
    except Exception as e:
        results["outputs"]["drainage_density_error"] = str(e)
        results["outputs"]["drainage_density_error_trace"] = traceback.format_exc()

    # 2) Annual rainfall (IMD)
    try:
        if get_rainfall_data is None:
            raise RuntimeError("get_rainfall_data not importable")
        rain = get_rainfall_data(inputs["latitude"], inputs["longitude"], start_year=inputs["rain_year"], data_dir=str(BACKEND_DIR))
        results["outputs"]["annual_rainfall_mm_total"] = rain
    except Exception as e:
        results["outputs"]["annual_rainfall_error"] = str(e)
        results["outputs"]["annual_rainfall_error_trace"] = traceback.format_exc()

    # 3) Soil lookup
    try:
        if get_point_description is None:
            raise RuntimeError("get_point_description not importable")
        # Use provided TIFF path argument; fallback to SOIL_TIF from module if that exists
        tif_to_use = inputs["soil_tif"] or SOIL_TIF
        pv, desc = get_point_description(tif_to_use, inputs["latitude"], inputs["longitude"])
        results["outputs"]["soil_pixel_value"] = int(pv)
        results["outputs"]["soil_description"] = desc
    except Exception as e:
        results["outputs"]["soil_error"] = str(e)
        results["outputs"]["soil_error_trace"] = traceback.format_exc()

    # 4) Runoff coefficient
    try:
        if get_runoff_coefficient_strict is None:
            raise RuntimeError("get_runoff_coefficient_strict not importable")
        rc = get_runoff_coefficient_strict(inputs["roof_material"])
        results["outputs"]["runoff_coefficient"] = float(rc)
    except Exception as e:
        results["outputs"]["runoff_coefficient_error"] = str(e)
        results["outputs"]["runoff_coefficient_error_trace"] = traceback.format_exc()

    # 5) ML prediction (optional — only if model artifacts exist and class imported)
    try:
        predictor, ml_msg = try_load_ml_predictor(inputs["model_dir"])
        results["outputs"]["ml_load_status"] = ml_msg
        if predictor is not None and getattr(predictor, "model", None) is not None:
            # prepare sample_input in the shape your predictor expects
            sample_input = {
                'rooftop_area': inputs["rooftop_area"],
                'dwellers': inputs["dwellers"],
                'roof_material': inputs["roof_material"],
                'annual_rainfall': results["outputs"].get("annual_rainfall_mm_total", 0.0) or 0.0,
                'aquifer_depth': 5.0,
                'soil_type': results["outputs"].get("soil_description", "Unknown"),
                'slope': 1.0,
                'drainage_density': results["outputs"].get("drainage_density_km_per_km2", 0.0),
                'runoff_coefficient': results["outputs"].get("runoff_coefficient", 0.0)
            }
            try:
                ml_pred = predictor.predict(sample_input)
                results["outputs"]["ml_prediction"] = ml_pred
            except Exception as e:
                results["outputs"]["ml_prediction_error"] = str(e)
                results["outputs"]["ml_prediction_error_trace"] = traceback.format_exc()
        else:
            results["outputs"]["ml_prediction"] = "ML model not loaded (no artifacts or incomplete loading)."
    except Exception as e:
        results["outputs"]["ml_error"] = str(e)
        results["outputs"]["ml_error_trace"] = traceback.format_exc()

    # ---- Print results to terminal (clean summary) ----
    print("\n=== AquaSpatial Run Summary ===")
    print("Inputs:")
    for k, v in results["inputs"].items():
        print(f"  {k:20s}: {v}")
    print("\nOutputs:")
    for k, v in results["outputs"].items():
        # keep printed output short for big objects like ML preds
        if isinstance(v, dict):
            print(f"  {k:20s}:")
            for kk, vv in v.items():
                print(f"    - {kk:18s}: {vv}")
        else:
            print(f"  {k:20s}: {v}")
    print("===============================\n")

    # Save JSON summary
    out_file = ROOT / "last_run_result.json"
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("Saved run summary to:", out_file)
    except Exception as e:
        print("Warning: failed to save run summary:", e)

if __name__ == "__main__":
    main()
