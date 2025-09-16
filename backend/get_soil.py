import rasterio
from pyproj import CRS, Transformer

# --- Constants defined at the top for clarity and easy modification ---

# Define the path to your GeoTIFF file.
# Ensure this path is correct relative to where you run your script.
TIF_FILE_PATH = "datasets/SOILTEXTURE.tif"

# This new dictionary maps the raw pixel values directly to the simplified,
# single-word output strings as requested.
SIMPLIFIED_SOIL_MAP = {
    4: "other",          # Official description: "Rocky and non soil"
    3: "sandy",          # Official description: "Coarse Texture"
    2: "loamy",          # Official description: "Medium texture"
    1: "clay",           # Official description: "Fine texture"
    0: "other"           # Official description: "DATA NOT AVAILABLE"
}

def get_soil_type(lat, lon):
    """
    Performs a complete analysis for a single point, returning a simplified soil type.

    This all-in-one function takes latitude and longitude, finds the corresponding
    pixel in the GeoTIFF, and returns a simple, human-readable soil category.

    Args:
        lat (float): The latitude of the point to sample (e.g., 15.9165).
        lon (float): The longitude of the point to sample (e.g., 80.1325).

    Returns:
        str: A string representing the simplified soil type: 'clay', 'sandy',
             'loamy', or 'other' if the data is not available, rocky, or an
             error occurs.
    """
    try:
        # Step 1: Open the raster file
        with rasterio.open(TIF_FILE_PATH) as src:
            
            # Step 2: Define coordinate systems
            # The CRS for standard latitude/longitude is WGS84 (EPSG:4326)
            wgs84 = CRS.from_epsg(4326)
            # Get the raster's own coordinate reference system from the file
            raster_crs = src.crs

            # Step 3: Transform the input coordinates to the raster's coordinate system
            # This is a crucial step for accuracy.
            transformer = Transformer.from_crs(wgs84, raster_crs, always_xy=True)
            x, y = transformer.transform(lon, lat)

            # Step 4: Sample the raster at the transformed coordinates
            # The 'sample' method returns a generator; 'next' gets the first value.
            # The result is a numpy array, so we select the first element [0].
            pixel_value = int(next(src.sample([(x, y)], indexes=1))[0])
            
            # Step 5: Look up the simplified description using the map
            # The .get() method safely returns a default value ('other') if the
            # pixel_value is not found in the dictionary keys.
            soil_type = SIMPLIFIED_SOIL_MAP.get(pixel_value, "other")
            
            return soil_type

    except Exception as e:
        # If any error occurs (e.g., file not found, point outside raster bounds),
        # print the error and return the default 'other' category.
        print(f"An error occurred for coordinates ({lat}, {lon}): {e}")
        return "other"

# --- soil slope

SLOPE_LEGEND_DESCRIPTIVE = {
    4: "Steep sloping (15% and above)",
    3: "Moderately sloping (8-15%)",
    2: "Gently sloping (3-8%)",
    1: "Level to very gently sloping (0-3%)",
    0: "DATA NOT AVAILABLE"
}

# Dictionary to map the raster's category value to a single, representative
# numerical slope percentage. This is useful for calculations.
# We take the midpoint of each range. For ">15%", we use 15 as the lower bound.
SLOPE_LEGEND_NUMERICAL = {
    4: 22.5, # Midpoint of 15-30% range
    3: 11.5, # Midpoint of 8-15% range
    2: 5.5,  # Midpoint of 3-8% range
    1: 1.5,  # Midpoint of 0-3% range
    0: -1    # Represents No Data
}


def get_soil_slope(lat, lon, tif_path=TIF_FILE_PATH):
    """
    Analyzes a soil slope GeoTIFF at a specific lat/lon and returns both a
    descriptive category and a representative numerical slope percentage.

    Args:
        lat (float): The latitude of the point to sample.
        lon (float): The longitude of the point to sample.
        tif_path (str): Optional. The full path to the GeoTIFF file.

    Returns:
        tuple: A tuple containing (description_string, numerical_slope_percentage).
               Returns ('error', -1) if processing fails.
    """
    try:
        with rasterio.open(tif_path) as src:
            # 1. TRANSFORM COORDINATES
            wgs84 = CRS.from_epsg(4326)
            raster_crs = src.crs
            if raster_crs != wgs84:
                transformer = Transformer.from_crs(wgs84, raster_crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
            else:
                x, y = lon, lat

            # 2. SAMPLE THE RASTER
            value_array = next(src.sample([(x, y)], indexes=1))
            pixel_value = int(value_array[0])

            # 3. INTERPRET THE VALUE
            description = SLOPE_LEGEND_DESCRIPTIVE.get(pixel_value, 'Unknown Value')
            numerical_slope = SLOPE_LEGEND_NUMERICAL.get(pixel_value, -1)

            return (description, numerical_slope)

    except Exception as e:
        print(f"An error occurred during raster processing: {e}")
        return ('error', -1)

# This block allows you to test the function by running this file directly.
# It will NOT run if you import the function into another script.
if __name__ == '__main__':
    print("--- Testing the get_soil_type function ---")

    # Test Case 1: Andhra Pradesh coast (expected: clay)
    lat1, lon1 = 15.9165, 80.1325
    soil1 = get_soil_type(lat1, lon1)
    print(f"Location: Andhra Pradesh Coast ({lat1}, {lon1})")
    print(f" -> Simplified Soil Type: '{soil1}'\n")

    # Test Case 2: Thar Desert, Rajasthan (expected: sandy)
    lat2, lon2 = 26.9124, 70.9083
    soil2 = get_soil_type(lat2, lon2)
    print(f"Location: Thar Desert ({lat2}, {lon2})")
    print(f" -> Simplified Soil Type: '{soil2}'\n")

    # Test Case 3: Indo-Gangetic Plain, near Delhi (expected: loamy)
    lat3, lon3 = 28.6139, 77.2090
    soil3 = get_soil_type(lat3, lon3)
    print(f"Location: Near Delhi ({lat3}, {lon3})")
    print(f" -> Simplified Soil Type: '{soil3}'\n")
    
    # Test Case 4: Himalayas, rocky area (expected: other)
    lat4, lon4 = 30.3165, 78.0322
    soil4 = get_soil_type(lat4, lon4)
    print(f"Location: Himalayan Region ({lat4}, {lon4})")
    print(f" -> Simplified Soil Type: '{soil4}'\n")

    # Test Case 5: Northeast India, no data (expected: other)
    lat5, lon5 = 27.5141, 96.3653
    soil5 = get_soil_type(lat5, lon5)
    print(f"Location: Northeast India ({lat5}, {lon5})")
    print(f" -> Simplified Soil Type: '{soil5}'\n")

    print("--- Testing the get_soil_slope function ---")

    # A typically flat area like the Indo-Gangetic Plain
    lat1, lon1 = 25.4358, 81.8463 # Prayagraj
    desc1, num1 = get_soil_slope(lat1, lon1)
    print(f"\nLocation: Prayagraj (Lat: {lat1}, Lon: {lon1})")
    print(f" -> Slope Category: '{desc1}'")
    print(f" -> Representative Slope: {num1}%")

    # A moderately sloped area in the Deccan Plateau
    lat2, lon2 = 18.5204, 73.8567 # Pune
    desc2, num2 = get_soil_slope(lat2, lon2)
    print(f"\nLocation: Pune (Lat: {lat2}, Lon: {lon2})")
    print(f" -> Slope Category: '{desc2}'")
    print(f" -> Representative Slope: {num2}%")

    # A steep area in the Himalayas
    lat3, lon3 = 31.1048, 77.1734 # Shimla
    desc3, num3 = get_soil_slope(lat3, lon3)
    print(f"\nLocation: Shimla (Lat: {lat3}, Lon: {lon3})")
    print(f" -> Slope Category: '{desc3}'")
    print(f" -> Representative Slope: {num3}%")