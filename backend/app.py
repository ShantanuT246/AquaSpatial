import sys
import json
import traceback
from pathlib import Path
import concurrent.futures
from typing import Dict, Any, Tuple, Optional, Callable
import random
import os
import math

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

# ---- NEW: Geological Data (Mock Functions) ----
def get_principal_aquifer(lat: float, lon: float) -> str:
    """
    Mock function to return principal aquifer type.
    In a real-world scenario, this would query a geological survey API.
    """
    # Simple mock logic based on latitude zones
    if 10 < lat < 25:
        return "Alluvial Aquifer System"
    elif lat >= 25:
        return "Indo-Ganga-Brahmaputra Alluvium"
    else:
        return "Crystalline Aquifers"

def get_groundwater_depth(lat: float, lon: float) -> float:
    """
    Mock function to return depth to groundwater level in meters.
    This would also query a hydrological data service in a real application.
    """
    # Mock data: random value in a realistic range
    return round(random.uniform(5.0, 45.0), 1)

# ---- NEW: Structure Dimension Calculator ----
def calculate_structure_dimensions(structure_type: str, recharge_volume_m3: float) -> Dict[str, Any]:
    """
    Calculates recommended dimensions for recharge structures.
    If recharge_volume_m3 is annual harvested m3, use a conservative design fraction.
    """
    if recharge_volume_m3 <= 0:
        return {"details": "Recharge volume is zero or negative; no dimensions calculated."}

    # Use 25% of annual potential for structure sizing (heuristic)
    design_volume = float(recharge_volume_m3) * 0.25  # m3

    # Provide structured outputs for pit/trench/shaft/tank
    out = {}
    # Recharge pit - assume depth of 2.5 m
    depth_pit = 2.5
    radius = math.sqrt(max(design_volume, 0.0001) / (math.pi * depth_pit))
    out['recharge_pit'] = {
        "type": "Recharge Pit",
        "design_volume_m3": round(design_volume, 3),
        "depth_m": depth_pit,
        "diameter_m": round(radius * 2, 2),
        "notes": "Pit sizing uses V = π r² h, recommended depth ≈ 2.5 m (adjust locally)."
    }

    # Recharge trench - assume depth 0.6 m and width 0.6 m
    width_trench = 0.6
    depth_trench = 0.6
    length_trench = design_volume / (width_trench * depth_trench)
    out['recharge_trench'] = {
        "type": "Recharge Trench",
        "design_volume_m3": round(design_volume, 3),
        "width_m": width_trench,
        "depth_m": depth_trench,
        "length_m": round(length_trench, 2),
        "notes": "Trench sizing uses V = L * W * D. Adjust width/depth to site constraints."
    }

    # Recharge shaft - assume depth 3.0 m
    depth_shaft = 3.0
    radius_shaft = math.sqrt(max(design_volume, 0.0001) / (math.pi * depth_shaft))
    out['recharge_shaft'] = {
        "type": "Recharge Shaft",
        "design_volume_m3": round(design_volume, 3),
        "depth_m": depth_shaft,
        "diameter_m": round(radius_shaft * 2, 2),
        "notes": "Shaft sizing approximated; final design must consider borehole yield."
    }

    # Storage tank (if storage preferred)
    out['storage_tank'] = {
        "type": "Storage Tank",
        "volume_m3": round(design_volume, 3),
        "volume_liters": int(round(design_volume * 1000)),
        "notes": "Use when direct recharge not favorable; consider overflow & first-flush."
    }

    return out

# ---- NEW: Cost-Benefit Analysis ----
def perform_cost_benefit_analysis(estimated_cost: float, harvested_m3: float) -> Dict[str, Any]:
    """
    Performs a simple cost-benefit analysis.
    """
    # Assumed cost of municipal water in INR per cubic meter (example; change if needed)
    WATER_COST_PER_M3 = 50.0  # ₹ per m3 (example)

    if harvested_m3 <= 0:
        return {"payback_period_years": "N/A", "annual_savings_inr": 0.0}

    annual_savings = harvested_m3 * WATER_COST_PER_M3
    if annual_savings <= 0:
        return {"payback_period_years": "Infinite", "annual_savings_inr": 0.0}

    payback_period = estimated_cost / annual_savings if annual_savings > 0 else None

    return {
        "payback_period_years": round(payback_period, 1) if payback_period is not None else "N/A",
        "annual_savings_inr": round(annual_savings, 2),
        "assumption_water_cost_per_m3_inr": WATER_COST_PER_M3
    }

# ---- Structure decision helper ----
def decide_structure(harvested_m3: float,
                     annual_demand_m3: float,
                     soil_type: str,
                     aquifer_depth_m: float,
                     slope_deg: float) -> Dict[str, Any]:
    """
    Rule-based decision for recommended RTRWH structure.
    Returns: { type: str, rationale: str }
    """
    soil = str(soil_type or "unknown").lower()
    rationale_parts = []

    good_soil_for_recharge = soil in ("sandy", "loamy")
    shallow_aquifer = (aquifer_depth_m <= 30)

    if good_soil_for_recharge and shallow_aquifer:
        structure = "Recharge Pit / Trench"
        rationale_parts.append("Permeable soil and shallow aquifer make direct recharge effective.")
    elif annual_demand_m3 > harvested_m3 * 0.8:
        structure = "Storage Tank"
        rationale_parts.append("High demand relative to harvested volume — storage recommended.")
    elif slope_deg > 15 or soil == "clay":
        structure = "Storage Tank"
        rationale_parts.append("Steep slope or clay soils reduce infiltration; on-site storage recommended.")
    else:
        structure = "Recharge Well"
        rationale_parts.append("Suitable for deeper aquifer recharge based on site indicators.")

    rationale_parts.append(f"Harvest volume ≈ {harvested_m3:.2f} m³/year; annual demand ≈ {annual_demand_m3:.2f} m³/year.")
    rationale = " ".join(rationale_parts)

    return {"type": structure, "rationale": rationale}

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

    # Run independent lookups in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_task, compute_drainage_density, "drainage_density_km_per_km2", lat=inputs["latitude"], lon=inputs["longitude"], verbose=False),
            executor.submit(run_task, get_rainfall_data, "annual_rainfall_mm_total", lat=inputs["latitude"], lon=inputs["longitude"]),
            executor.submit(run_task, get_soil_type, "soil_description", lat=inputs["latitude"], lon=inputs["longitude"]),
            executor.submit(run_task, get_runoff_coefficient_strict, "runoff_coefficient", roof_type=inputs["roof_material"]),
            # NEW TASKS (geological/hydro)
            executor.submit(run_task, get_principal_aquifer, "principal_aquifer", lat=inputs["latitude"], lon=inputs["longitude"]),
            executor.submit(run_task, get_groundwater_depth, "groundwater_depth_m", lat=inputs["latitude"], lon=inputs["longitude"]),
            executor.submit(run_task, try_load_ml_predictor, "ml_predictor_status", model_dir=inputs["model_dir"])
        ]
        for future in concurrent.futures.as_completed(futures):
            # merge each result dict into outputs
            try:
                results["outputs"].update(future.result())
            except Exception:
                # Ensure we don't break if one future returned an unexpected structure
                continue

    # Load ML predictor if available
    predictor, ml_msg = results["outputs"].pop("ml_predictor_status", (None, "Status unknown."))
    results["outputs"]["ml_load_status"] = ml_msg

    # Rainfall fallback
    rainfall = results["outputs"].get("annual_rainfall_mm_total")
    if rainfall is None:
        fallback_rainfall = random.uniform(650, 900)
        rainfall = fallback_rainfall
        results["outputs"]["annual_rainfall_mm_total"] = round(rainfall, 2)
        results["outputs"]["rainfall_data_status"] = "Estimated value (actual data not available)"
    else:
        results["outputs"]["rainfall_data_status"] = "Actual data received"

    # Basic inputs (with defaults)
    rooftop_area = float(inputs.get("rooftop_area") or DEFAULTS["rooftop_area"])
    dwellers = int(inputs.get("dwellers") or DEFAULTS["dwellers"])
    roof_material = inputs.get("roof_material") or DEFAULTS["roof_material"]
    annual_rainfall = float(rainfall)  # in mm/year
    aquifer_depth_supplied = float(request_data.get("aquifer_depth", results["outputs"].get("groundwater_depth_m", 5.0)))
    soil_type = results["outputs"].get("soil_description") or request_data.get("soil_type", "unknown")
    slope = float(request_data.get("slope", 1.0))
    drainage_density = float(results["outputs"].get("drainage_density_km_per_km2") or request_data.get("drainage_density", 0.0))
    runoff_coefficient = float(results["outputs"].get("runoff_coefficient") or request_data.get("runoff_coefficient", 0.0))

    if runoff_coefficient == 0.0:
        mapping = {'metal':0.95,'concrete':0.9,'tiles':0.8,'asphalt':0.88,'thatch':0.6}
        runoff_coefficient = mapping.get(roof_material, 0.8)

    sample_input = {
        'rooftop_area': rooftop_area, 'dwellers': dwellers, 'roof_material': roof_material,
        'annual_rainfall': annual_rainfall, 'aquifer_depth': aquifer_depth_supplied, 'soil_type': soil_type,
        'slope': slope, 'drainage_density': drainage_density, 'runoff_coefficient': runoff_coefficient
    }

    # ML prediction (if predictor loaded)
    pred_raw = None
    if predictor:
        try:
            pred_raw = predictor.predict(sample_input)
        except Exception as e:
            results["outputs"]["ml_prediction_error"] = f"Predictor failure: {e}"
            pred_raw = None

    # Deterministic runoff calculation (physics-based): 1 mm over 1 m^2 = 1 liter
    deterministic_liters = rooftop_area * annual_rainfall * runoff_coefficient  # liters/year
    # ML-adjustment if predictor provided (keep both)
    if pred_raw and isinstance(pred_raw, dict):
        ml_adjusted_liters = float(pred_raw.get('ml_adjusted_liters', deterministic_liters))
    else:
        ml_adjusted_liters = deterministic_liters

    # Convert to m3
    deterministic_m3 = deterministic_liters / 1000.0
    adjusted_m3 = ml_adjusted_liters / 1000.0

    # Harvested / Demand math
    harvested_m3 = adjusted_m3
    liters_per_person_per_day = 135.0
    annual_demand_liters = dwellers * liters_per_person_per_day * 365.0
    annual_demand_m3 = annual_demand_liters / 1000.0
    harvest_demand_ratio = float(harvested_m3 / annual_demand_m3) if annual_demand_m3 > 0 else 0.0

    # Suitability heuristic
    harvest_norm = min(harvest_demand_ratio / 2.0, 1.0)
    slope_norm = min(max(slope / 45.0, 0.0), 1.0)
    soil_score_map = {'sandy':0.9, 'loamy':0.9, 'clay':0.7, 'rocky':0.4, 'unknown':0.7}
    soil_score = float(soil_score_map.get(str(soil_type).lower(), 0.7))
    suitability_score = 0.6 * harvest_norm + 0.25 * (1.0 - slope_norm) + 0.15 * soil_score
    suitability_score = float(max(0.0, min(1.0, suitability_score)))

    # Cost estimation (existing approach)
    base_cost = 5000.0
    storage_cost_per_m3 = 1200.0
    installation_rate_per_m2 = 100.0
    soil_multiplier_map = {'clay': 1.15, 'sandy': 1.0, 'loamy': 1.08, 'rocky': 1.25, 'unknown': 1.0}
    soil_multiplier = float(soil_multiplier_map.get(str(soil_type).lower(), 1.0))
    desired_storage_m3 = harvested_m3 * 0.5
    installation_cost = rooftop_area * installation_rate_per_m2 * soil_multiplier
    storage_cost = desired_storage_m3 * storage_cost_per_m3
    cost_estimation = base_cost + installation_cost + storage_cost
    cost_estimation = float(round(cost_estimation, 2))

    # ---- NEW: Perform new analyses ----
    feasibility = "Feasible" if suitability_score > 0.45 and harvested_m3 > 0.01 else "Not Recommended"
    structure_info = decide_structure(harvested_m3, annual_demand_m3, soil_type, aquifer_depth_supplied, slope)
    dimensions_info = calculate_structure_dimensions(structure_info.get("type", "Storage Tank"), harvested_m3)
    cost_benefit_info = perform_cost_benefit_analysis(cost_estimation, harvested_m3)

    # ---- NEW: Cost breakdown for frontend
    cost_breakdown = {
        "base_cost_inr": round(base_cost, 2),
        "installation_cost_inr": round(installation_cost, 2),
        "installation_rate_per_m2_inr": round(installation_rate_per_m2, 2),
        "storage_cost_inr": round(storage_cost, 2),
        "storage_cost_per_m3_inr": round(storage_cost_per_m3, 2),
        "soil_multiplier": soil_multiplier,
        "desired_storage_m3": round(desired_storage_m3, 3),
        "note": "Breakdown is heuristic. Use for quick estimates; refine with local vendor quotes."
    }

    # ---- NEW: Populate authoritative fields at top-level outputs so frontend can read them directly ----
    results["outputs"]["deterministic_runoff_liters"] = round(deterministic_liters, 2)
    results["outputs"]["deterministic_runoff_m3"] = round(deterministic_m3, 3)
    results["outputs"]["ml_adjusted_liters"] = round(float(ml_adjusted_liters), 2)
    results["outputs"]["ml_adjusted_m3"] = round(float(adjusted_m3), 3)

    results["outputs"]["runoff_generation_liters"] = round(ml_adjusted_liters, 2)
    results["outputs"]["runoff_generation_m3"] = round(adjusted_m3, 3)

    # principal aquifer and groundwater depth (ensure they exist in outputs)
    if "principal_aquifer" not in results["outputs"] or not results["outputs"].get("principal_aquifer"):
        results["outputs"]["principal_aquifer"] = get_principal_aquifer(inputs["latitude"], inputs["longitude"])
    if "groundwater_depth_m" not in results["outputs"] or results["outputs"].get("groundwater_depth_m") is None:
        results["outputs"]["groundwater_depth_m"] = get_groundwater_depth(inputs["latitude"], inputs["longitude"])

    # recommended dimensions and structure details
    results["outputs"]["suggested_structure"] = structure_info
    results["outputs"]["recommended_dimensions"] = dimensions_info
    results["outputs"]["feasibility_check"] = feasibility
    results["outputs"]["suitability_score"] = round(suitability_score, 4)
    results["outputs"]["harvest_demand_ratio"] = round(harvest_demand_ratio, 3)
    results["outputs"]["cost_estimation_inr"] = round(cost_estimation, 2)
    results["outputs"]["cost_breakdown"] = cost_breakdown
    results["outputs"]["cost_benefit"] = cost_benefit_info
    results["outputs"]["local_rainfall_mm"] = round(annual_rainfall, 1)

    # Also add a summary object (for compatibility)
    final_output = {
        "feasibility_check": feasibility,
        "runoff_generation_liters": round(ml_adjusted_liters, 2),
        "runoff_generation_m3": round(adjusted_m3, 3),
        "suitability_score": round(suitability_score, 4),
        "cost_estimation_inr": round(cost_estimation, 2),
        "cost_breakdown": cost_breakdown,
        "suggested_structure": structure_info,
        "structure_dimensions": dimensions_info,
        "principal_aquifer": results["outputs"].get("principal_aquifer", "Data not available"),
        "groundwater_depth_m": results["outputs"].get("groundwater_depth_m", "Data not available"),
        "local_rainfall_mm": round(annual_rainfall, 1),
        "cost_benefit": cost_benefit_info
    }

    results["outputs"]["analysis_results"] = final_output

    # Keep smaller metadata and remove any large ML internals if present
    results.pop("ml_prediction", None)

    # Serialize & save a copy
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

import time

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    print("Received a request on /analyze")
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    request_data = request.get_json()
    try:
        start_time = time.time()
        results = run_analysis(request_data)
        end_time = time.time()
        elapsed = end_time - start_time
        results['calculation_time_sec'] = round(elapsed, 2)
        return jsonify(results)
    except Exception as e:
        import traceback
        err_trace = traceback.format_exc()
        print(f"ANALYZE ERROR: {e}\n{err_trace}", file=sys.stderr)
        # Return full error temporarily for debugging
        return jsonify({
            "error": str(e),
            "traceback": err_trace,
            "calculation_time_sec": 0
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
