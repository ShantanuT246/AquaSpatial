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

# This block allows you to test the function by running this file directly.
# It will NOT run if you import the function into another script.
# if __name__ == '__main__':
#     print("--- Testing the get_simplified_soil_type function ---")

#     # Test Case 1: Andhra Pradesh coast (expected: clay)
#     lat1, lon1 = 15.9165, 80.1325
#     soil1 = get_simplified_soil_type(lat1, lon1)
#     print(f"Location: Andhra Pradesh Coast ({lat1}, {lon1})")
#     print(f" -> Simplified Soil Type: '{soil1}'\n")

#     # Test Case 2: Thar Desert, Rajasthan (expected: sandy)
#     lat2, lon2 = 26.9124, 70.9083
#     soil2 = get_simplified_soil_type(lat2, lon2)
#     print(f"Location: Thar Desert ({lat2}, {lon2})")
#     print(f" -> Simplified Soil Type: '{soil2}'\n")

#     # Test Case 3: Indo-Gangetic Plain, near Delhi (expected: loamy)
#     lat3, lon3 = 28.6139, 77.2090
#     soil3 = get_simplified_soil_type(lat3, lon3)
#     print(f"Location: Near Delhi ({lat3}, {lon3})")
#     print(f" -> Simplified Soil Type: '{soil3}'\n")
    
#     # Test Case 4: Himalayas, rocky area (expected: other)
#     lat4, lon4 = 30.3165, 78.0322
#     soil4 = get_simplified_soil_type(lat4, lon4)
#     print(f"Location: Himalayan Region ({lat4}, {lon4})")
#     print(f" -> Simplified Soil Type: '{soil4}'\n")

#     # Test Case 5: Northeast India, no data (expected: other)
#     lat5, lon5 = 27.5141, 96.3653
#     soil5 = get_simplified_soil_type(lat5, lon5)
#     print(f"Location: Northeast India ({lat5}, {lon5})")
#     print(f" -> Simplified Soil Type: '{soil5}'\n")