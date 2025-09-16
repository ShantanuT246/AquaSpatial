import rasterio
from pyproj import CRS, Transformer

# Define the path to your Groundwater Depth GeoTIFF file.
# You will need to replace 'GROUNDWATER_DEPTH.tif' with your actual filename.
TIF_FILE_PATH = "datasets/TN_AQ_THICK.tif"


def get_groundwater_depth(lat, lon, tif_path=TIF_FILE_PATH):
    """
    Analyzes a groundwater depth GeoTIFF at a specific lat/lon and returns
    the numerical depth value directly from the raster.

    The returned value is the raw pixel value, which typically represents the
    depth in meters below the ground level. This should be confirmed by checking
    the raster's metadata.

    Args:
        lat (float): The latitude of the point to sample.
        lon (float): The longitude of the point to sample.
        tif_path (str): Optional. The full path to the GeoTIFF file.

    Returns:
        float: The depth value at the given location. Returns the raster's nodata
               value (or -9999 if not set) if the point is outside the raster's
               extent or if an error occurs.
    """
    try:
        with rasterio.open(tif_path) as src:
            # Check for a defined NoData value, provide a default if none exists
            nodata_val = src.nodata if src.nodata is not None else -9999

            # 1. TRANSFORM COORDINATES
            wgs84 = CRS.from_epsg(4326)
            raster_crs = src.crs
            if raster_crs != wgs84:
                transformer = Transformer.from_crs(wgs84, raster_crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
            else:
                x, y = lon, lat

            # 2. SAMPLE THE RASTER
            # The sample method returns a generator of numpy arrays.
            value_array = next(src.sample([(x, y)], indexes=1))

            # The value is often a float for continuous data
            depth_value = float(value_array[0])

            return depth_value

    except Exception as e:
        print(f"An error occurred during raster processing: {e}")
        return nodata_val


# This block allows you to test the function directly by running this file.
if __name__ == '__main__':
    print("--- Testing the get_groundwater_depth function ---")
    
    # Note: These coordinates are for demonstration. The output will depend
    # entirely on the data within your actual GROUNDWATER_DEPTH.tif file.

    # Example 1: A location in Punjab, known for significant groundwater usage
    lat1, lon1 = 30.9010, 75.8573 # Ludhiana
    depth1 = get_groundwater_depth(lat1, lon1)
    print(f"\nLocation: Ludhiana (Lat: {lat1}, Lon: {lon1})")
    print(f" -> Groundwater Depth: {depth1} (units assumed to be meters)")

    # Example 2: A location in Rajasthan
    lat2, lon2 = 26.9124, 75.7873 # Jaipur
    depth2 = get_groundwater_depth(lat2, lon2)
    print(f"\nLocation: Jaipur (Lat: {lat2}, Lon: {lon2})")
    print(f" -> Groundwater Depth: {depth2} (units assumed to be meters)")

    lat2, lon2 = 12.9333, 79.35 # Vellore
    depth2 = get_groundwater_depth(lat2, lon2)
    print(f"\nLocation: Vellore (Lat: {lat2}, Lon: {lon2})")
    print(f" -> Groundwater Depth: {depth2} (units assumed to be meters)")