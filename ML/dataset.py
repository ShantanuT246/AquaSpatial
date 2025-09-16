import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from gretel_client import configure_session
from gretel_client.projects import create_or_get_unique_project
from gretel_client.helpers import poll
from gretel_client.synthetics import SyntheticModel

# Load environment variables
load_dotenv()

# Connect to Gretel
configure_session(api_key=os.getenv("GRETEL_API_KEY"), cache="yes")

# Create new project
project = create_or_get_unique_project(name="AquaSpatial-Synthetic")

# Create inline seed dataset
seed_df = pd.DataFrame({
    'rooftop_area': [120, 85, 150, 95, 110],
    'dwellers': [4, 3, 5, 2, 4],
    'roof_material': ['tile', 'metal', 'tile', 'asphalt', 'metal'],
    'annual_rainfall': [800, 650, 900, 700, 750],
    'aquifer_depth': [30, 45, 25, 40, 35],
    'soil_type': ['clay', 'sandy', 'loamy', 'clay', 'sandy'],
    'slope': [5, 3, 7, 4, 6],
    'drainage_density': [1.2, 0.8, 1.5, 1.0, 1.1],
    'runoff_coefficient': [0.3, 0.25, 0.35, 0.28, 0.32]
})

# Create model using current Gretel SDK approach (updated)
model = SyntheticModel.create(seed_df, project_name="AquaSpatial-Synthetic")
model.submit_cloud()
poll(model)
record_handler = model.create_record_handler(params={"num_records": 5000})
record_handler.submit()
poll(record_handler)
synthetic_data = record_handler.get_records_as_df()

# Post-processing rules to enforce hydrology domain logic
# Ensure all columns are numeric where needed
for col in ['rooftop_area', 'dwellers', 'annual_rainfall', 'aquifer_depth', 
            'slope', 'drainage_density', 'runoff_coefficient']:
    synthetic_data[col] = pd.to_numeric(synthetic_data[col], errors='coerce')

# Drop any rows with NaN values that might have been created
synthetic_data = synthetic_data.dropna()

# Compute recharge_potential (in cubic meters)
synthetic_data['recharge_potential'] = (
    synthetic_data['annual_rainfall'] / 1000 *  # Convert mm to m
    synthetic_data['rooftop_area'] * 
    synthetic_data['runoff_coefficient']
)

# Normalize recharge_potential and slope for suitability_score calculation
recharge_norm = (
    synthetic_data['recharge_potential'] - synthetic_data['recharge_potential'].min()
) / (synthetic_data['recharge_potential'].max() - synthetic_data['recharge_potential'].min())

slope_norm = (
    synthetic_data['slope'] - synthetic_data['slope'].min()
) / (synthetic_data['slope'].max() - synthetic_data['slope'].min())

# Compute suitability_score between 0 and 1
synthetic_data['suitability_score'] = 0.7 * recharge_norm + 0.3 * (1 - slope_norm)
synthetic_data['suitability_score'] = synthetic_data['suitability_score'].clip(0, 1)

# Compute harvest_demand_ratio: rooftop rainfall vs dwellers' demand
# Assuming 100 liters per dweller per day demand and 365 days
synthetic_data['harvested_rainfall'] = (
    synthetic_data['annual_rainfall'] / 1000 *  # Convert mm to m
    synthetic_data['rooftop_area'] * 
    0.8  # 80% efficiency
)

synthetic_data['demand'] = (
    synthetic_data['dwellers'] * 
    100 *  # liters per person per day
    365 /  # days per year
    1000   # convert liters to cubic meters
)

synthetic_data['harvest_demand_ratio'] = (
    synthetic_data['harvested_rainfall'] / synthetic_data['demand']
)

# Handle division by zero and infinite values
synthetic_data['harvest_demand_ratio'] = synthetic_data['harvest_demand_ratio'].replace(
    [np.inf, -np.inf], 0
).fillna(0)

# Compute cost_estimation with base cost, rooftop area factor, and soil type multiplier
base_cost = 5000  # Base cost in currency units
soil_type_multiplier = synthetic_data['soil_type'].map({
    'clay': 1.2, 
    'sandy': 1.0, 
    'loamy': 1.1,
    'rocky': 1.3  # Added rocky soil type which might be more expensive
}).fillna(1.0)  # Default multiplier for unknown soil types

synthetic_data['cost_estimation'] = (
    base_cost + 
    synthetic_data['rooftop_area'] * 100 * soil_type_multiplier  # Area cost factor
)

# Save the synthetic dataset
synthetic_data.to_csv("synthetic_dataset.csv", index=False)
print(f"Synthetic dataset saved with {len(synthetic_data)} records")

# Visualization: plot histograms of all generated columns
plt.figure(figsize=(20, 15))
columns = [col for col in synthetic_data.columns if col not in ['roof_material', 'soil_type']]
for i, column in enumerate(columns):
    plt.subplot(4, 4, i+1)
    if synthetic_data[column].dtype in [np.int64, np.float64]:
        plt.hist(synthetic_data[column].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title(column)
    plt.tight_layout()

plt.savefig("synthetic_data_distributions.png")
plt.close()
print("Visualization saved as synthetic_data_distributions.png")

# Also create pairplot for numerical columns to show relationships
import seaborn as sns
numerical_cols = synthetic_data.select_dtypes(include=[np.number]).columns.tolist()
plt.figure(figsize=(15, 15))
sns.pairplot(synthetic_data[numerical_cols].sample(min(500, len(synthetic_data))))
plt.savefig("synthetic_data_pairplot.png")
plt.close()
print("Pairplot saved as synthetic_data_pairplot.png")