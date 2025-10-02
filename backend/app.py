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
    # Import predictor wrapper from ML.main (class Predictor)
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

# -------------------------
# Recommendation Engine
# -------------------------
def recommend_recharge_structure(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple rule-based recommendation engine.
    inputs keys used:
      - infiltration_mm_hr (optional): measured or estimated infiltration in mm/hr
      - groundwater_depth_m
      - soil_type
      - slope_pct
      - available_area_m2
      - runoff_quality: 'good'|'dirty'|'contaminated'
      - urban: bool
    Returns a dict with a ranked list of candidate structures + hints.
    """
    # read inputs with defaults
    inf = float(inputs.get('infiltration_mm_hr', inputs.get('infiltration_estimate_mm_hr', 10.0)))
    gw = float(inputs.get('groundwater_depth_m', 10.0))
    soil = str(inputs.get('soil_type', 'loamy')).lower()
    slope = float(inputs.get('slope_pct', inputs.get('slope', 2.0)))
    area = float(inputs.get('available_area_m2', 100.0))
    quality = str(inputs.get('runoff_quality', 'good')).lower()
    urban = bool(inputs.get('urban', False))

    # base candidate list
    candidates = ['rooftop_recharge', 'soak_pit', 'infiltration_trench',
                  'percolation_pond', 'check_dam', 'recharge_well', 'pervious_pavement']

    scores = {c: 0 for c in candidates}

    # infiltration influence
    if inf >= 30:
        scores['soak_pit'] += 2
        scores['infiltration_trench'] += 2
        scores['percolation_pond'] += 2
        scores['pervious_pavement'] += 1
    elif inf >= 5:
        scores['soak_pit'] += 1
        scores['infiltration_trench'] += 1
        scores['percolation_pond'] += 1
    else:
        # poor infiltration -> favor recharge wells if aquifer is deep/permeable
        scores['recharge_well'] += 2

    # groundwater depth
    if gw < 3:
        scores['recharge_well'] -= 2
        scores['soak_pit'] += 1
    elif gw > 15:
        scores['recharge_well'] += 2

    # slope
    if slope > 10:
        scores['check_dam'] += 2
        scores['infiltration_trench'] += 1

    # available area
    if area < 25:
        scores['rooftop_recharge'] += 2
        scores['pervious_pavement'] += 1
        scores['percolation_pond'] -= 2
    elif area < 100:
        scores['soak_pit'] += 1

    # urban and quality
    if urban:
        scores['pervious_pavement'] += 1
        # in many cities small recharge wells or rooftop to pit with pretreatment are used
        scores['recharge_well'] += 1

    if quality in ['dirty', 'contaminated']:
        # direct injection is discouraged without treatment
        scores['recharge_well'] -= 2
        scores['percolation_pond'] -= 1

    # soil influence
    if soil in ['sandy', 'loamy']:
        scores['soak_pit'] += 1
        scores['infiltration_trench'] += 1
    elif soil in ['clay', 'rocky']:
        scores['recharge_well'] += 1

    # prepare sizing hints (very approximate)
    def sizing_hint(name):
        if name == 'rooftop_recharge':
            return "Use rooftop area A(m²): annual captured (m³) = A*R(mm)*C/1000. Add first-flush & filter."
        if name == 'soak_pit':
            return "Typical pit: 1–3 m diameter, 1–3 m deep. Size by infiltration rate and target volume."
        if name == 'infiltration_trench':
            return "Trench length depends on desired recharge volume and sidewall area; use filter media and velocity checks."
        if name == 'percolation_pond':
            return "Large-area pond; compute area = V/(infiltration_rate*storage_depth)."
        if name == 'check_dam':
            return "Small check-dams across ephemeral channels to slow runoff and allow infiltration."
        if name == 'recharge_well':
            return "Injection/recharge well requires hydrogeological design and strict pretreatment."
        if name == 'pervious_pavement':
            return "Pervious pavement with sub-base storage; good in urban areas with limited space."
        return ""

    ranked = sorted([(k, int(v)) for k, v in scores.items()], key=lambda x: x[1], reverse=True)

    # return top 3 with hints
    out = []
    for name, sc in ranked[:3]:
        out.append({
            "structure": name,
            "score": sc,
            "sizing_hint": sizing_hint(name)
        })

    return {
        "input_summary": {
            "infiltration_mm_hr": inf,
            "groundwater_depth_m": gw,
            "soil_type": soil,
            "slope_pct": slope,
            "available_area_m2": area,
            "runoff_quality": quality,
            "urban": urban
        },
        "recommendations": out
    }

# -------------------------
# Core Logic Refactored for Web Server
# -------------------------
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
    recharge_potential = float(ml_adjusted_liters)
    harvested_m3 = recharge_potential / 1000.0
    liters_per_person_per_day = 135.0
    annual_demand_liters = dwellers * liters_per_person_per_day * 365.0
    annual_demand_m3 = annual_demand_liters / 1000.0
    harvest_demand_ratio = float(harvested_m3 / annual_demand_m3) if annual_demand_m3 > 0 else 0.0
    harvest_norm = min(harvest_demand_ratio / 2.0, 1.0)
    slope_norm = min(max(slope / 45.0, 0.0), 1.0)
    soil_score_map = {'sandy':0.9, 'loamy':0.9, 'clay':0.7, 'rocky':0.4, 'unknown':0.7}
    soil_score = float(soil_score_map.get(str(soil_type).lower(), 0.7))
    suitability_score = 0.6 * harvest_norm + 0.25 * (1.0 - slope_norm) + 0.15 * soil_score
    suitability_score = float(max(0.0, min(1.0, suitability_score)))
    base_cost = 5000.0
    storage_cost_per_m3 = 1200.0
    installation_rate_per_m2 = 100.0
    soil_multiplier_map = {'clay': 1.15, 'sandy': 1.0, 'loamy': 1.08, 'rocky': 1.25, 'unknown': 1.0}
    soil_multiplier = float(soil_multiplier_map.get(str(soil_type).lower(), 1.0))
    desired_storage_m3 = harvested_m3 * 0.5
    cost_estimation = base_cost + (rooftop_area * installation_rate_per_m2 * soil_multiplier) + (desired_storage_m3 * storage_cost_per_m3)
    cost_estimation = float(round(cost_estimation, 2))

    ml_prediction = {
        "recharge_potential": round(recharge_potential, 2),
        "suitability_score": round(suitability_score, 4),
        "harvest_demand_ratio": round(harvest_demand_ratio, 3),
        "cost_estimation": round(cost_estimation, 2),
        "deterministic_liters": round(deterministic_liters, 2),
        "ml_adjusted_liters": round(ml_adjusted_liters, 2),
        "residual_pred": round(residual_pred, 2),
        "used_ml": used_ml
    }

    results["outputs"]["ml_prediction"] = ml_prediction

    # Build recommendation input (estimate infiltration if not provided)
    # If frontend supplied infiltration_mm_hr use that; else estimate from soil_type
    soil_to_inf = {'sandy': 30.0, 'loamy': 10.0, 'clay': 1.0, 'rocky': 0.5, 'unknown': 5.0}
    infiltration_est = float(request_data.get('infiltration_mm_hr', soil_to_inf.get(str(soil_type).lower(), 5.0)))

    rec_inputs = {
        'infiltration_mm_hr': infiltration_est,
        'groundwater_depth_m': float(request_data.get('groundwater_depth_m', aquifer_depth)),
        'soil_type': soil_type,
        'slope_pct': float(request_data.get('slope', slope)),
        'available_area_m2': float(request_data.get('available_area_m2', rooftop_area)),
        'runoff_quality': request_data.get('runoff_quality', 'good'),
        'urban': bool(request_data.get('urban', False))
    }

    # get recommendations
    try:
        rec = recommend_recharge_structure(rec_inputs)
        # add both the input_summary and recommendations into outputs for frontend use
        results["outputs"]["recommendation_input_summary"] = rec.get("input_summary")
        results["outputs"]["recommendations"] = rec.get("recommendations")
    except Exception as e:
        results["outputs"]["recommendations_error"] = str(e)

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
