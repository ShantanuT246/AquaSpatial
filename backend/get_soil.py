import rasterio
from pyproj import CRS, Transformer
import numpy as np

tif_file = "datasets/SOILTEXTURE.tif"

# This dictionary holds the official legend from your Raster Attribute Table.
# It makes the output human-readable.
SOIL_LEGEND = {
    4: "Rocky and non soil",
    3: "Coarse Texture (Loamy sand, sand)",
    2: "Medium texture (Loam, silt loam, silt, sandy loam)",
    1: "Fine texture (Loamy clay, Clay, sandy clay, etc.)",
    0: "DATA NOT AVAILABLE"
}

def get_point_value(tif_path, lat, lon):
    """
    Samples a GeoTIFF raster at a specific latitude and longitude to get the raw pixel value.

    Args:
        tif_path (str): The full path to the GeoTIFF file.
        lat (float): The latitude of the point to sample (e.g., 15.9165).
        lon (float): The longitude of the point to sample (e.g., 80.1325).

    Returns:
        int: The integer pixel value at the given location. Returns the raster's nodata
             value if the point is outside the raster's extent or an error occurs.
    """
    try:
        with rasterio.open(tif_path) as src:
            # The CRS for standard latitude/longitude is WGS84 (EPSG:4326)
            wgs84 = CRS.from_epsg(4326)
            
            # Get the raster's own coordinate reference system
            raster_crs = src.crs

            # If the raster and the input coordinates are in different systems, transform the point
            if raster_crs != wgs84:
                transformer = Transformer.from_crs(wgs84, raster_crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
            else:
                x, y = lon, lat # The raster is already in WGS84

            # The sample method returns a generator. We get the first (and only) item.
            # It returns a numpy array, so we select the first element [0] to get the scalar value.
            value_array = next(src.sample([(x, y)], indexes=1))
            
            # Ensure the return value is a standard Python integer
            return int(value_array[0])

    except Exception as e:
        print(f"An error occurred: {e}")
        # In case of any error, it's safer to return a value indicating no data.
        # You can get the specific nodata value from the raster if needed.
        return -1 # Or src.nodata

def get_point_description(tif_path, lat, lon):
    """
    Samples a GeoTIFF and returns both the raw value and its human-readable description.

    This is the recommended function to use for getting meaningful output.

    Args:
        tif_path (str): The full path to the GeoTIFF file.
        lat (float): The latitude of the point to sample.
        lon (float): The longitude of the point to sample.

    Returns:
        tuple: A tuple containing (pixel_value, description_string).
               For example: (1, "Fine texture (Loamy clay, Clay, sandy clay, etc.)")
    """
    # First, get the raw numerical value from the raster
    pixel_value = get_point_value(tif_path, lat, lon)

    # Then, look up the description for that value in our legend.
    # The .get() method safely handles cases where the value might not be in the legend.
    description = SOIL_LEGEND.get(pixel_value, "Unknown or No Data")

    return (pixel_value, description)