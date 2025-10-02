# dataset.py
"""
Realistic synthetic dataset generator for AquaSpatial.
Creates synthetic_dataset.csv with simulated observed values and logically
derived fields (deterministic baseline, suitability, demand, cost, etc).
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N = 5000  # number of synthetic examples to generate

# 1) Sample realistic distributions for inputs
# rooftop area in m^2 (typical residential: 20 - 400)
rooftop_area = np.clip(np.random.normal(loc=120, scale=60, size=N), 10, 600)

# dwellers: small households most common
dwellers = np.clip(np.random.poisson(lam=3.0, size=N), 1, 12)

# roof materials and probabilities (more metal/tiles)
roof_materials = ['metal', 'concrete', 'tiles', 'asphalt', 'thatch']
roof_probs = [0.35, 0.25, 0.25, 0.1, 0.05]
roof_material = np.random.choice(roof_materials, size=N, p=roof_probs)

# annual rainfall in mm (wide variability)
annual_rainfall = np.clip(np.random.normal(loc=900, scale=350, size=N), 100, 4000)

# aquifer depth (m)
aquifer_depth = np.round(np.random.uniform(2, 80, size=N), 1)

# soil types (affects infiltration)
soil_types = ['clay', 'sandy', 'loamy', 'rocky']
soil_probs = [0.25, 0.35, 0.30, 0.10]
soil_type = np.random.choice(soil_types, size=N, p=soil_probs)

# slope in degrees (rooftop or small parcel slope): 0..45
slope = np.clip(np.random.beta(a=2, b=5, size=N) * 25, 0.0, 45.0)  # many gentle slopes

# drainage density (km/km2) local geomorph metric 0.2-3
drainage_density = np.clip(np.random.normal(loc=1.0, scale=0.5, size=N), 0.1, 3.0)

# base runoff coefficient by material (prior)
base_runoff = {
    'metal': 0.95,
    'concrete': 0.90,
    'tiles': 0.80,
    'asphalt': 0.88,
    'thatch': 0.6
}
runoff_coefficient = np.array([base_runoff[m] for m in roof_material], dtype=float)

# small modifications by slope: very low slopes lose more (gutter issues),
# very steep roofs may shed quickly (slight increase)
runoff_coefficient = runoff_coefficient + np.clip((slope - 5.0) / 200.0, -0.05, 0.05)

# add small random variation / measurement noise
runoff_coefficient += np.random.normal(0.0, 0.02, size=N)
runoff_coefficient = np.clip(runoff_coefficient, 0.4, 0.99)

# 2) Deterministic baseline (exact physics)
# liters_per_year = area_m2 * annual_rainfall_mm * runoff_coefficient
deterministic_liters = rooftop_area * annual_rainfall * runoff_coefficient

# 3) Simulate "observed" liters that include realistic losses + systematic biases
# We model observed = deterministic * (1 - losses + bias) + random_noise
# Losses depend on:
#  - soil_type: clay -> higher loss to surface runoff? (useful proxy)
#  - drainage_density: high drainage density -> more partitioning to drains (less local recharge)
#  - slope: low slope may cause stagnation and overflow issues -> small extra loss
# We'll design a systematic "residual fraction" (observed - deterministic) / deterministic

# base loss fraction mean (e.g., 10% losses)
base_loss_frac = 0.10

# soil modifiers
soil_modifier = {
    'clay': 0.06,   # slightly more loss or infiltration complexity
    'sandy': -0.02, # sand infiltrates better, so decreased loss for recharge (more effective capture)
    'loamy': 0.0,
    'rocky': 0.08
}
soil_mod = np.array([soil_modifier[s] for s in soil_type], dtype=float)

# drainage density effect: more drainage density -> more quick routing away (higher loss)
drainage_mod = np.clip((drainage_density - 1.0) * 0.05, -0.05, 0.15)

# slope effect: very low slopes (<2) negative, moderate slopes small positive effect
slope_mod = np.where(slope < 2.0, 0.05, np.where(slope > 25.0, -0.02, 0.0))

# roof material systemic bias: e.g., old tile roofs leak a bit
material_bias = {
    'metal': -0.01,
    'concrete': 0.00,
    'tiles': 0.02,
    'asphalt': 0.01,
    'thatch': 0.05
}
material_mod = np.array([material_bias[m] for m in roof_material], dtype=float)

# Combine into a residual fraction (positive fraction means more observed than deterministic,
# negative means actual less than deterministic). Here mostly negative (losses).
residual_frac = - (base_loss_frac + soil_mod + drainage_mod + slope_mod + material_mod)

# Add spatially-varying or random bias: sample small systematic region effects
region_bias = np.random.normal(loc=0.0, scale=0.03, size=N)  # some regions better/worse

# Final residual multiplier on deterministic (observed = deterministic * (1 + residual_frac + region_bias) + noise)
residual_frac += region_bias

# random measurement noise (heteroskedastic: larger roofs -> slightly larger absolute noise)
noise_frac = np.random.normal(0.0, 0.06, size=N)
noise_frac = np.clip(noise_frac, -0.30, 0.30)

observed_liters = deterministic_liters * (1.0 + residual_frac + noise_frac)

# Keep observed positive
observed_liters = np.maximum(observed_liters, 0.0)

# 4) Derived quantities: harvested_rainfall (m3), demand (m3), ratio, suitability, cost
# harvested_rainfall_m3 = observed_liters / 1000 (since observed_liters is already liters/year)
harvested_m3 = observed_liters / 1000.0

# Demand: use 135 liters/person/day as default (tuneable)
liters_per_person_per_day = 135.0
annual_demand_liters = dwellers * liters_per_person_per_day * 365.0
annual_demand_m3 = annual_demand_liters / 1000.0

harvest_demand_ratio = np.divide(harvested_m3, annual_demand_m3, out=np.zeros_like(harvested_m3), where=annual_demand_m3>0)

# suitability_score: combine normalized harvest/demand ratio and inverse slope and soil quality
# create simple heuristic: score = 0.6 * norm(harvest_demand_ratio) + 0.25 * (1 - norm(slope)) + 0.15 * soil_score
# soil_score: sandy and loamy favorable (0.9), clay neutral (0.7), rocky less favorable (0.4)
soil_score_map = {'sandy':0.9, 'loamy':0.9, 'clay':0.7, 'rocky':0.4}
soil_score = np.array([soil_score_map[s] for s in soil_type], dtype=float)

# Normalize harvest_demand_ratio and slope
r_min, r_max = np.nanmin(harvest_demand_ratio), np.nanmax(harvest_demand_ratio)
if r_max - r_min < 1e-8:
    harvest_norm = np.clip(harvest_demand_ratio, 0, 1)
else:
    harvest_norm = (harvest_demand_ratio - r_min) / (r_max - r_min)
# slope normalized 0..1
s_min, s_max = np.nanmin(slope), np.nanmax(slope)
slope_norm = (slope - s_min) / (s_max - s_min + 1e-8)

suitability_score = 0.6 * harvest_norm + 0.25 * (1.0 - slope_norm) + 0.15 * soil_score
suitability_score = np.clip(suitability_score, 0.0, 1.0)

# cost estimation (a simple parametric formula)
base_cost = 5000.0  # fixed baseline (currency)
storage_cost_per_m3 = 1200.0  # cost per cubic meter of storage capacity
installation_rate_per_m2 = 100.0

soil_multiplier_map = {'clay': 1.15, 'sandy': 1.0, 'loamy': 1.08, 'rocky': 1.25}
soil_multiplier = np.array([soil_multiplier_map[s] for s in soil_type], dtype=float)

# desired storage = 50% of annual harvested (a policy decision)
desired_storage_m3 = harvested_m3 * 0.5

cost_estimation = base_cost + (rooftop_area * installation_rate_per_m2 * soil_multiplier) + (desired_storage_m3 * storage_cost_per_m3)
cost_estimation = np.round(cost_estimation, 2)

# 5) Assemble DataFrame and save
df = pd.DataFrame({
    'rooftop_area': rooftop_area,
    'dwellers': dwellers,
    'roof_material': roof_material,
    'annual_rainfall': annual_rainfall,
    'aquifer_depth': aquifer_depth,
    'soil_type': soil_type,
    'slope': slope,
    'drainage_density': drainage_density,
    'runoff_coefficient': np.round(runoff_coefficient, 3),
    'deterministic_liters': np.round(deterministic_liters, 2),
    'observed_liters': np.round(observed_liters, 2),
    'harvested_m3': np.round(harvested_m3, 3),
    'annual_demand_m3': np.round(annual_demand_m3, 3),
    'harvest_demand_ratio': np.round(harvest_demand_ratio, 3),
    'suitability_score': np.round(suitability_score, 4),
    'cost_estimation': cost_estimation
})

OUT = "synthetic_dataset.csv"
df.to_csv(OUT, index=False)
print(f"Synthetic dataset saved to {OUT} with {len(df)} rows")
