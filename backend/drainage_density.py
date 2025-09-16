"""
drainage_density.py  -- compute drainage density from India-WRIS ArcGIS REST layers

Usage:
    python drainage_density.py
"""

import requests
import json
import math
import tempfile
import os
import traceback
from shapely.geometry import shape, Point
import geopandas as gpd
from pyproj import CRS

# Default ArcGIS REST layer endpoints (you can override if needed)
DEFAULT_WATERSHED_LAYER = "https://gis.nwic.in/server/rest/services/NWIC/Jal_Dharohar/MapServer/3"
DEFAULT_RIVER_LAYER     = "https://gis.nwic.in/server/rest/services/SubInfoSysLCC/Basin_NWIC/MapServer/1"

def _arcgis_query_geojson(layer_base_url, params):
    url = layer_base_url.rstrip("/") + "/query"
    params = params.copy()
    params.setdefault("f", "geojson")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()

def _utm_crs_for_lonlat(lon, lat):
    zone = int((lon + 180.0) / 6.0) + 1
    epsg = 32600 + zone
    return CRS.from_epsg(epsg)

def _ensure_gdf_has_crs(gdf):
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)

def _geojson_to_gdf(geojson):
    """
    Convert a Python dict GeoJSON to GeoDataFrame by writing a temporary file
    (avoids 'Invalid IPv6 URL' issues some Fiona builds throw with geojson:// URI).
    """
    text = json.dumps(geojson)
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False, encoding="utf-8")
        tmp.write(text)
        tmp.flush()
        tmp.close()
        gdf = gpd.read_file(tmp.name)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
    return gdf

def compute_drainage_density(lat, lon,
                             watershed_layer_url=None,
                             rivers_layer_url=None,
                             search_buffer_km=2,
                             verbose=False):
    """
    Find watershed polygon at (lat, lon) from India-WRIS ArcGIS REST, gather rivers within it,
    clip rivers to watershed, compute total stream length (km), basin area (km^2), and drainage density.

    Returns:
        (drainage_density_km_per_km2, total_length_km, basin_area_km2)
    """
    watershed_layer_url = watershed_layer_url or DEFAULT_WATERSHED_LAYER
    rivers_layer_url = rivers_layer_url or DEFAULT_RIVER_LAYER

    if verbose:
        print("Querying watershed layer for point containment...")

    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson"
    }

    try:
        gj = _arcgis_query_geojson(watershed_layer_url, params)
    except Exception as e:
        # try a fallback host if available
        try:
            if "gis.nwic.in" in watershed_layer_url:
                alt = watershed_layer_url.replace("gis.nwic.in", "arc.indiawris.gov.in")
                if verbose:
                    print(f"Primary watershed service failed; trying fallback: {alt}")
                gj = _arcgis_query_geojson(alt, params)
                watershed_layer_url = alt
            else:
                raise
        except Exception as e2:
            tb = traceback.format_exc()
            raise RuntimeError(f"Failed to query watershed service. Errors: {e} / {e2}\nTrace:\n{tb}")

    features = gj.get("features", [])
    if len(features) == 0:
        if verbose:
            print("No watershed returned for point. Expanding search using bbox buffer...")
        km = float(search_buffer_km)
        dlat = km / 111.0
        dlon = km / (111.320 * math.cos(math.radians(lat)))
        xmin, ymin = lon - dlon, lat - dlat
        xmax, ymax = lon + dlon, lat + dlat
        params_bbox = {
            "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson"
        }
        gj = _arcgis_query_geojson(watershed_layer_url, params_bbox)
        features = gj.get("features", [])
        if len(features) == 0:
            raise ValueError("No watershed polygon found near the provided coordinates. Try increasing search_buffer_km or use different coordinates.")

    # Convert to GeoDataFrame
    try:
        w_gdf = _geojson_to_gdf(gj)
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(f"Failed to read watershed GeoJSON into GeoDataFrame: {e}\nTrace:\n{tb}")

    _ensure_gdf_has_crs(w_gdf)

    pt = Point(lon, lat)
    w_gdf['contains_pt'] = w_gdf.geometry.apply(lambda g: g.contains(pt))
    containing = w_gdf[w_gdf['contains_pt'] == True]

    if len(containing) == 1:
        watershed = containing.iloc[0].geometry
        if verbose: print("Found watershed polygon containing the point.")
    elif len(containing) > 1:
        watershed = w_gdf.geometry.union_all()
        if verbose: print("Multiple watershed polygons contain the point; unioned them.")
    else:
        watershed = w_gdf.geometry.union_all()
        if verbose: print("No polygon strictly contains point; using union of returned polygons as watershed.")

    if verbose:
        print("Querying river layer within watershed bounding box...")

    minx, miny, maxx, maxy = watershed.bounds
    bbox_params = {
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson"
    }

    try:
        rivers_gj = _arcgis_query_geojson(rivers_layer_url, bbox_params)
    except Exception as e:
        try:
            if "gis.nwic.in" in rivers_layer_url:
                alt = rivers_layer_url.replace("gis.nwic.in", "arc.indiawris.gov.in")
                if verbose:
                    print(f"Primary rivers service failed; trying fallback: {alt}")
                rivers_gj = _arcgis_query_geojson(alt, bbox_params)
            else:
                raise
        except Exception as e2:
            tb = traceback.format_exc()
            raise RuntimeError(f"Failed to query rivers service: {e} / {e2}\nTrace:\n{tb}")

    if not rivers_gj.get("features"):
        if verbose:
            print("No river features found within watershed bbox. Returning zeros.")
        return 0.0, 0.0, 0.0

    try:
        rivers_gdf = _geojson_to_gdf(rivers_gj)
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(f"Failed to read rivers GeoJSON into GeoDataFrame: {e}\nTrace:\n{tb}")

    _ensure_gdf_has_crs(rivers_gdf)
    ws_gdf = gpd.GeoDataFrame({"geometry": [watershed]}, crs="EPSG:4326")

    if verbose:
        print("DEBUG: river features returned by query:", len(rivers_gdf))

    try:
        rivers_clipped = gpd.clip(rivers_gdf, ws_gdf)
    except Exception:
        rivers_clipped = gpd.overlay(rivers_gdf, ws_gdf, how="intersection")

    if verbose:
        print("DEBUG: river features after clipping:", len(rivers_clipped))

    if rivers_clipped.empty:
        if verbose:
            print("No river geometry remained after clipping. Returning zeros.")
        return 0.0, 0.0, 0.0

    centroid = ws_gdf.geometry.union_all().centroid
    utm_crs = _utm_crs_for_lonlat(centroid.x, centroid.y)
    utm_epsg = utm_crs.to_epsg()
    if verbose:
        print(f"Projecting to UTM EPSG:{utm_epsg} for metric calculations.")

    rivers_proj = rivers_clipped.to_crs(epsg=utm_epsg)
    ws_proj = ws_gdf.to_crs(epsg=utm_epsg)

    rivers_proj["length_m"] = rivers_proj.geometry.length
    total_length_m = rivers_proj["length_m"].sum()
    total_length_km = total_length_m / 1000.0

    basin_area_m2 = ws_proj.geometry.area.sum()
    basin_area_km2 = basin_area_m2 / 1e6

    if basin_area_km2 <= 0:
        raise ValueError("Computed basin area is zero or invalid.")

    drainage_density = total_length_km / basin_area_km2

    if verbose:
        print(">> rivers_clipped features:", len(rivers_clipped))
        print(">> total_length_km:", total_length_km)
        print(">> basin_area_km2:", basin_area_km2)
        print(f"Drainage density (Dd): {drainage_density:.6f} km/km^2")

    return float(drainage_density)


# if __name__ == "__main__":
#     lat = 22.5726
#     lon = 88.3639
#     try:
#         dd, total_len_km, basin_area_km2 = compute_drainage_density(lat, lon, verbose=True)
#         print(f"\nComputed drainage density at ({lat}, {lon}): {dd:.6f} km/km^2")
#         print("Returned components -> total_length_km:", total_len_km, ", basin_area_km2:", basin_area_km2)
#     except Exception as e:
#         print("Error computing drainage density:", str(e))
#         print("Full traceback:")
#         traceback.print_exc()
