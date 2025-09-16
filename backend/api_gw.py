"""
tamil_nadu_aquifer_working.py -- Working script for Tamil Nadu aquifer depth data
"""

import requests
import json
import math
import urllib3
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings
urllib3.disable_warnings(InsecureRequestWarning)
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Working service endpoint
SERVICE_URL = "https://arc.indiawris.gov.in/server/rest/services/SubInfoSysLCC/Litholog_Analysis_Depth_Thickness_Material/MapServer"

# Tamil Nadu layers (confirmed from service response)
TN_LAYERS = {
    "depth": 9,      # Depth in Tamilnadu (m bgl)
    "thickness": 16  # Thickness in Tamilnadu (m)
}

def transform_coordinates(lon, lat):
    """
    Transform WGS84 coordinates to Lambert Conformal Conic (service projection)
    The service uses: WGS_1984_Lambert_Conformal_Conic
    """
    try:
        from pyproj import Transformer
        
        # Service spatial reference (from the curl response)
        service_wkt = '''PROJCS["WGS_1984_Lambert_Conformal_Conic",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic"],PARAMETER["False_Easting",4000000.0],PARAMETER["False_Northing",4000000.0],PARAMETER["Central_Meridian",80.0],PARAMETER["Standard_Parallel_1",12.472944],PARAMETER["Standard_Parallel_2",35.1728055],PARAMETER["Latitude_Of_Origin",24.0],UNIT["Meter",1.0]]'''
        
        # Transform from WGS84 to service projection
        transformer = Transformer.from_crs("EPSG:4326", service_wkt, always_xy=True)
        x, y = transformer.transform(lon, lat)
        
        return x, y
    
    except ImportError:
        # Fallback: approximate transformation for Tamil Nadu region
        # This is less accurate but works without pyproj
        print("Warning: pyproj not available, using approximate transformation")
        
        # Approximate Lambert Conformal Conic transformation for Tamil Nadu
        central_meridian = 80.0
        standard_parallel_1 = 12.472944
        false_easting = 4000000.0
        false_northing = 4000000.0
        
        # Simple approximation (not geodetically accurate)
        x = false_easting + (lon - central_meridian) * 111320 * math.cos(math.radians(lat))
        y = false_northing + (lat - 24.0) * 110540
        
        return x, y

def get_tn_aquifer_depth(lat, lon, include_thickness=False, verbose=False):
    """
    Get Tamil Nadu aquifer depth using the working India-WRIS service
    """
    
    # Create session with SSL bypass
    session = requests.Session()
    session.verify = False
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    # Transform coordinates to service projection
    try:
        x, y = transform_coordinates(lon, lat)
        if verbose:
            print(f"Transformed coordinates: ({x:.2f}, {y:.2f})")
    except Exception as e:
        if verbose:
            print(f"Coordinate transformation failed: {e}")
        # Use original coordinates as fallback
        x, y = lon, lat
    
    result = {
        "latitude": lat,
        "longitude": lon,
        "aquifer_depth_m_bgl": None,
        "aquifer_thickness_m": None,
        "success": False,
        "method": "India-WRIS Tamil Nadu Layer"
    }
    
    try:
        # Query Tamil Nadu depth layer (ID: 9)
        if verbose:
            print(f"Querying Tamil Nadu aquifer depth layer...")
        
        # Method 1: Try identify operation with transformed coordinates
        identify_params = {
            "geometry": f"{x},{y}",
            "geometryType": "esriGeometryPoint",
            "sr": "102100",  # Try Web Mercator first
            "layers": f"visible:{TN_LAYERS['depth']}",
            "tolerance": 10,
            "mapExtent": f"{x-1000},{y-1000},{x+1000},{y+1000}",
            "imageDisplay": "400,400,96",
            "returnGeometry": "false",
            "f": "json"
        }
        
        identify_url = f"{SERVICE_URL}/identify"
        resp = session.get(identify_url, params=identify_params, timeout=30)
        
        if resp.status_code == 200:
            data = resp.json()
            if verbose:
                print("Response:", json.dumps(data, indent=2))
            
            results = data.get("results", [])
            if results:
                for result_item in results:
                    attributes = result_item.get("attributes", {})
                    
                    # Look for depth value
                    depth_fields = ["Pixel Value", "pixel_value", "VALUE", "Value", "DEPTH", "depth"]
                    for field in depth_fields:
                        if field in attributes and attributes[field] is not None:
                            try:
                                depth_val = float(attributes[field])
                                if depth_val > 0:  # Valid depth
                                    result["aquifer_depth_m_bgl"] = depth_val
                                    result["success"] = True
                                    if verbose:
                                        print(f"✅ Found depth: {depth_val} m bgl")
                                    break
                            except (ValueError, TypeError):
                                continue
                
                if result["success"] and include_thickness:
                    # Query thickness layer (ID: 16)
                    thickness_params = identify_params.copy()
                    thickness_params["layers"] = f"visible:{TN_LAYERS['thickness']}"
                    
                    thick_resp = session.get(identify_url, params=thickness_params, timeout=30)
                    if thick_resp.status_code == 200:
                        thick_data = thick_resp.json()
                        thick_results = thick_data.get("results", [])
                        
                        for thick_item in thick_results:
                            thick_attrs = thick_item.get("attributes", {})
                            for field in depth_fields:
                                if field in thick_attrs and thick_attrs[field] is not None:
                                    try:
                                        thick_val = float(thick_attrs[field])
                                        if thick_val > 0:
                                            result["aquifer_thickness_m"] = thick_val
                                            if verbose:
                                                print(f"✅ Found thickness: {thick_val} m")
                                            break
                                    except (ValueError, TypeError):
                                        continue
        
        # Method 2: Try with original WGS84 coordinates if first method failed
        if not result["success"]:
            if verbose:
                print("Trying with WGS84 coordinates...")
            
            wgs84_params = {
                "geometry": f"{lon},{lat}",
                "geometryType": "esriGeometryPoint",
                "sr": "4326",  # WGS84
                "layers": f"visible:{TN_LAYERS['depth']}",
                "tolerance": 15,
                "mapExtent": f"{lon-0.1},{lat-0.1},{lon+0.1},{lat+0.1}",
                "imageDisplay": "400,400,96",
                "returnGeometry": "false",
                "f": "json"
            }
            
            resp2 = session.get(identify_url, params=wgs84_params, timeout=30)
            if resp2.status_code == 200:
                data2 = resp2.json()
                
                results2 = data2.get("results", [])
                if results2:
                    for result_item in results2:
                        attributes = result_item.get("attributes", {})
                        for field in ["Pixel Value", "pixel_value", "VALUE", "Value"]:
                            if field in attributes and attributes[field] is not None:
                                try:
                                    depth_val = float(attributes[field])
                                    if depth_val > 0:
                                        result["aquifer_depth_m_bgl"] = depth_val
                                        result["success"] = True
                                        result["method"] += " (WGS84 coords)"
                                        if verbose:
                                            print(f"✅ Found depth with WGS84: {depth_val} m bgl")
                                        break
                                except (ValueError, TypeError):
                                    continue
        
        return result
    
    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"❌ Error: {str(e)}")
        return result

def test_vellore_and_surroundings():
    """Test multiple locations around Vellore"""
    
    test_locations = [
        (12.9165, 79.1325, "Vellore City"),
        (12.8956, 79.1500, "VIT Vellore Campus"),
        (12.9500, 79.1000, "Vellore North"),
        (12.8800, 79.1600, "Vellore East"),
        (13.0827, 80.2707, "Chennai (reference)")
    ]
    
    print("=== Tamil Nadu Aquifer Depth Analysis ===\n")
    print(f"{'Location':<20} {'Coordinates':<20} {'Depth (m bgl)':<15} {'Thickness (m)':<15} {'Status':<10}")
    print("-" * 85)
    
    for lat, lon, name in test_locations:
        result = get_tn_aquifer_depth(lat, lon, include_thickness=True, verbose=False)
        
        depth = f"{result['aquifer_depth_m_bgl']:.2f}" if result['aquifer_depth_m_bgl'] else "N/A"
        thickness = f"{result['aquifer_thickness_m']:.2f}" if result['aquifer_thickness_m'] else "N/A"
        status = "✅ Success" if result['success'] else "❌ Failed"
        coords = f"({lat:.3f}, {lon:.3f})"
        
        print(f"{name:<20} {coords:<20} {depth:<15} {thickness:<15} {status:<10}")
    
    # Detailed test for Vellore
    print(f"\n=== Detailed Analysis for Vellore ===")
    vellore_result = get_tn_aquifer_depth(12.9165, 79.1325, include_thickness=True, verbose=True)
    print("\nComplete result:")
    print(json.dumps(vellore_result, indent=2))

if __name__ == "__main__":
    test_vellore_and_surroundings()
