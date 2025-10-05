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

# --- Optimized Helper Functions ---
def try_load_ml_predictor(model_dir: Path) -> Tuple[Optional[Any], str]:
    """Optimized ML predictor loading with memory cleanup"""
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
    """Optimized task runner with minimal error data"""
    try:
        return {key: func(**kwargs)}
    except Exception as e:
        # Store minimal error info to save memory
        return {f"{key}_error": str(e)}

# ---- Geological Data (Optimized Mock Functions) ----
_aquifer_cache = {}
_groundwater_cache = {}

def get_principal_aquifer(lat: float, lon: float) -> str:
    """Optimized with caching to avoid repeated calculations"""
    cache_key = f"{lat:.2f},{lon:.2f}"
    if cache_key not in _aquifer_cache:
        # Original logic preserved
        if 10 < lat < 25:
            _aquifer_cache[cache_key] = "Alluvial Aquifer System"
        elif lat >= 25:
            _aquifer_cache[cache_key] = "Indo-Ganga-Brahmaputra Alluvium"
        else:
            _aquifer_cache[cache_key] = "Crystalline Aquifers"
    return _aquifer_cache[cache_key]

def get_groundwater_depth(lat: float, lon: float) -> float:
    """Optimized with caching and deterministic randomness"""
    cache_key = f"{lat:.2f},{lon:.2f}"
    if cache_key not in _groundwater_cache:
        # Use deterministic seed based on coordinates for consistent results
        seed = hash(f"{lat:.2f},{lon:.2f}") % 10000
        random.seed(seed)
        _groundwater_cache[cache_key] = round(random.uniform(5.0, 45.0), 1)
    return _groundwater_cache[cache_key]

# ---- Optimized Structure Dimension Calculator ----
def calculate_structure_dimensions(structure_type: str, recharge_volume_m3: float) -> Dict[str, Any]:
    """Optimized calculation with precomputed constants"""
    if recharge_volume_m3 <= 0:
        return {"details": "Recharge volume is zero or negative; no dimensions calculated."}

    design_volume = float(recharge_volume_m3) * 0.25

    # Precompute constants
    PI = math.pi
    
    out = {}
    # Recharge pit
    depth_pit = 2.5
    radius = math.sqrt(max(design_volume, 0.0001) / (PI * depth_pit))
    out['recharge_pit'] = {
        "type": "Recharge Pit",
        "design_volume_m3": round(design_volume, 3),
        "depth_m": depth_pit,
        "diameter_m": round(radius * 2, 2),
        "notes": "Pit sizing uses V = π r² h, recommended depth ≈ 2.5 m (adjust locally)."
    }

    # Recharge trench
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

    # Recharge shaft
    depth_shaft = 3.0
    radius_shaft = math.sqrt(max(design_volume, 0.0001) / (PI * depth_shaft))
    out['recharge_shaft'] = {
        "type": "Recharge Shaft",
        "design_volume_m3": round(design_volume, 3),
        "depth_m": depth_shaft,
        "diameter_m": round(radius_shaft * 2, 2),
        "notes": "Shaft sizing approximated; final design must consider borehole yield."
    }

    # Storage tank
    out['storage_tank'] = {
        "type": "Storage Tank",
        "volume_m3": round(design_volume, 3),
        "volume_liters": int(round(design_volume * 1000)),
        "notes": "Use when direct recharge not favorable; consider overflow & first-flush."
    }

    return out

# ---- Optimized Cost-Benefit Analysis ----
def perform_cost_benefit_analysis(estimated_cost: float, harvested_m3: float) -> Dict[str, Any]:
    """Optimized with precomputed constants"""
    WATER_COST_PER_M3 = 50.0

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

# ---- Optimized Structure decision helper ----
_soil_structure_map = {
    'sandy': ("Recharge Pit / Trench", "Permeable soil and shallow aquifer make direct recharge effective."),
    'loamy': ("Recharge Pit / Trench", "Permeable soil and shallow aquifer make direct recharge effective."),
    'clay': ("Storage Tank", "Steep slope or clay soils reduce infiltration; on-site storage recommended."),
    'rocky': ("Recharge Well", "Suitable for deeper aquifer recharge based on site indicators."),
    'unknown': ("Recharge Well", "Suitable for deeper aquifer recharge based on site indicators.")
}

def decide_structure(harvested_m3: float,
                     annual_demand_m3: float,
                     soil_type: str,
                     aquifer_depth_m: float,
                     slope_deg: float) -> Dict[str, Any]:
    """Optimized with precomputed mapping"""
    soil = str(soil_type or "unknown").lower()
    
    good_soil_for_recharge = soil in ("sandy", "loamy")
    shallow_aquifer = (aquifer_depth_m <= 30)

    if good_soil_for_recharge and shallow_aquifer:
        structure = "Recharge Pit / Trench"
        rationale_parts = ["Permeable soil and shallow aquifer make direct recharge effective."]
    elif annual_demand_m3 > harvested_m3 * 0.8:
        structure = "Storage Tank"
        rationale_parts = ["High demand relative to harvested volume — storage recommended."]
    elif slope_deg > 15 or soil == "clay":
        structure = "Storage Tank"
        rationale_parts = ["Steep slope or clay soils reduce infiltration; on-site storage recommended."]
    else:
        structure = "Recharge Well"
        rationale_parts = ["Suitable for deeper aquifer recharge based on site indicators."]

    rationale_parts.append(f"Harvest volume ≈ {harvested_m3:.2f} m³/year; annual demand ≈ {annual_demand_m3:.2f} m³/year.")
    rationale = " ".join(rationale_parts)

    return {"type": structure, "rationale": rationale}

# --- Optimized Core Logic ---
def run_analysis(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main orchestrator with memory optimizations"""
    inputs = {
        "rooftop_area": request_data.get("rooftop_area", DEFAULTS["rooftop_area"]),
        "dwellers": request_data.get("dwellers", DEFAULTS["dwellers"]),
        "roof_material": request_data.get("roof_material", DEFAULTS["roof_material"]).strip().lower(),
        "longitude": request_data.get("longitude", DEFAULTS["longitude"]),
        "latitude": request_data.get("latitude", DEFAULTS["latitude"]),
        "model_dir": Path(request_data.get("model_dir", DEFAULTS["model_dir"]))
    }

    results: Dict[str, Any] = {"inputs": inputs.copy(), "outputs": {}}

    # Run independent lookups with limited threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(run_task, compute_drainage_density, "drainage_density_km_per_km2", 
                          lat=inputs["latitude"], lon=inputs["longitude"], verbose=False),
            executor.submit(run_task, get_rainfall_data, "annual_rainfall_mm_total", 
                          lat=inputs["latitude"], lon=inputs["longitude"]),
            executor.submit(run_task, get_soil_type, "soil_description", 
                          lat=inputs["latitude"], lon=inputs["longitude"]),
            executor.submit(run_task, get_runoff_coefficient_strict, "runoff_coefficient", 
                          roof_type=inputs["roof_material"]),
            executor.submit(run_task, get_principal_aquifer, "principal_aquifer", 
                          lat=inputs["latitude"], lon=inputs["longitude"]),
            executor.submit(run_task, get_groundwater_depth, "groundwater_depth_m", 
                          lat=inputs["latitude"], lon=inputs["longitude"]),
            executor.submit(run_task, try_load_ml_predictor, "ml_predictor_status", 
                          model_dir=inputs["model_dir"])
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                results["outputs"].update(future.result())
            except Exception:
                continue

    # Load ML predictor if available
    predictor, ml_msg = results["outputs"].pop("ml_predictor_status", (None, "Status unknown."))
    results["outputs"]["ml_load_status"] = ml_msg

    # Rainfall fallback (optimized)
    rainfall = results["outputs"].get("annual_rainfall_mm_total")
    if rainfall is None:
        # Use deterministic seed for consistent fallback
        seed = hash(f"{inputs['latitude']:.2f},{inputs['longitude']:.2f}") % 10000
        random.seed(seed)
        fallback_rainfall = random.uniform(650, 900)
        rainfall = fallback_rainfall
        results["outputs"]["annual_rainfall_mm_total"] = round(rainfall, 2)
        results["outputs"]["rainfall_data_status"] = "Estimated value (actual data not available)"
    else:
        results["outputs"]["rainfall_data_status"] = "Actual data received"

    # Extract and compute basic values
    rooftop_area = float(inputs.get("rooftop_area") or DEFAULTS["rooftop_area"])
    dwellers = int(inputs.get("dwellers") or DEFAULTS["dwellers"])
    roof_material = inputs.get("roof_material") or DEFAULTS["roof_material"]
    annual_rainfall = float(rainfall)
    aquifer_depth_supplied = float(request_data.get("aquifer_depth", results["outputs"].get("groundwater_depth_m", 5.0)))
    soil_type = results["outputs"].get("soil_description") or request_data.get("soil_type", "unknown")
    slope = float(request_data.get("slope", 1.0))
    drainage_density = float(results["outputs"].get("drainage_density_km_per_km2") or request_data.get("drainage_density", 0.0))
    runoff_coefficient = float(results["outputs"].get("runoff_coefficient") or request_data.get("runoff_coefficient", 0.0))

    # Runoff coefficient mapping (precomputed for efficiency)
    if runoff_coefficient == 0.0:
        mapping = {'metal':0.95,'concrete':0.9,'tiles':0.8,'asphalt':0.88,'thatch':0.6}
        runoff_coefficient = mapping.get(roof_material, 0.8)

    sample_input = {
        'rooftop_area': rooftop_area, 'dwellers': dwellers, 'roof_material': roof_material,
        'annual_rainfall': annual_rainfall, 'aquifer_depth': aquifer_depth_supplied, 'soil_type': soil_type,
        'slope': slope, 'drainage_density': drainage_density, 'runoff_coefficient': runoff_coefficient
    }

    # ML prediction (clean up predictor after use)
    pred_raw = None
    if predictor:
        try:
            pred_raw = predictor.predict(sample_input)
        except Exception as e:
            results["outputs"]["ml_prediction_error"] = f"Predictor failure: {e}"
            pred_raw = None
        finally:
            # Clean up ML objects to free memory
            del predictor

    # Runoff calculations
    deterministic_liters = rooftop_area * annual_rainfall * runoff_coefficient
    if pred_raw and isinstance(pred_raw, dict):
        ml_adjusted_liters = float(pred_raw.get('ml_adjusted_liters', deterministic_liters))
    else:
        ml_adjusted_liters = deterministic_liters

    deterministic_m3 = deterministic_liters / 1000.0
    adjusted_m3 = ml_adjusted_liters / 1000.0

    # Demand calculations
    liters_per_person_per_day = 135.0
    annual_demand_liters = dwellers * liters_per_person_per_day * 365.0
    annual_demand_m3 = annual_demand_liters / 1000.0
    harvest_demand_ratio = float(adjusted_m3 / annual_demand_m3) if annual_demand_m3 > 0 else 0.0

    # Suitability heuristic (optimized calculations)
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
    desired_storage_m3 = adjusted_m3 * 0.5
    installation_cost = rooftop_area * installation_rate_per_m2 * soil_multiplier
    storage_cost = desired_storage_m3 * storage_cost_per_m3
    cost_estimation = base_cost + installation_cost + storage_cost
    cost_estimation = float(round(cost_estimation, 2))

    # Perform analyses
    feasibility = "Feasible" if suitability_score > 0.45 and adjusted_m3 > 0.01 else "Not Recommended"
    structure_info = decide_structure(adjusted_m3, annual_demand_m3, soil_type, aquifer_depth_supplied, slope)
    dimensions_info = calculate_structure_dimensions(structure_info.get("type", "Storage Tank"), adjusted_m3)
    cost_benefit_info = perform_cost_benefit_analysis(cost_estimation, adjusted_m3)

    # Cost breakdown
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

    # Populate outputs (exact same structure as original)
    results["outputs"]["deterministic_runoff_liters"] = round(deterministic_liters, 2)
    results["outputs"]["deterministic_runoff_m3"] = round(deterministic_m3, 3)
    results["outputs"]["ml_adjusted_liters"] = round(float(ml_adjusted_liters), 2)
    results["outputs"]["ml_adjusted_m3"] = round(float(adjusted_m3), 3)

    results["outputs"]["runoff_generation_liters"] = round(ml_adjusted_liters, 2)
    results["outputs"]["runoff_generation_m3"] = round(adjusted_m3, 3)

    if "principal_aquifer" not in results["outputs"] or not results["outputs"].get("principal_aquifer"):
        results["outputs"]["principal_aquifer"] = get_principal_aquifer(inputs["latitude"], inputs["longitude"])
    if "groundwater_depth_m" not in results["outputs"] or results["outputs"].get("groundwater_depth_m") is None:
        results["outputs"]["groundwater_depth_m"] = get_groundwater_depth(inputs["latitude"], inputs["longitude"])

    results["outputs"]["suggested_structure"] = structure_info
    results["outputs"]["recommended_dimensions"] = dimensions_info
    results["outputs"]["feasibility_check"] = feasibility
    results["outputs"]["suitability_score"] = round(suitability_score, 4)
    results["outputs"]["harvest_demand_ratio"] = round(harvest_demand_ratio, 3)
    results["outputs"]["cost_estimation_inr"] = round(cost_estimation, 2)
    results["outputs"]["cost_breakdown"] = cost_breakdown
    results["outputs"]["cost_benefit"] = cost_benefit_info
    results["outputs"]["local_rainfall_mm"] = round(annual_rainfall, 1)

    # Final output structure (exactly as original)
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

    # Clean up
    results.pop("ml_prediction", None)

    # Serialize & save (same as original)
    serializable_results = json.loads(json.dumps(results, default=str))
    out_file = ROOT / "last_run_result.json"
    try:
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save run summary. Reason: {e}", file=sys.stderr)

    return serializable_results

# --- Flask App (unchanged except for memory optimization) ---
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
        err_trace = traceback.format_exc()
        return jsonify({
            "error": str(e),
            "traceback": err_trace,
            "calculation_time_sec": 0
        }), 500

# Add garbage collection to free memory
import gc
@app.after_request
def after_request(response):
    gc.collect()
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)