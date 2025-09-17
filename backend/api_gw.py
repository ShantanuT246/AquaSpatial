import requests
import json
import math
import urllib3
from requests.exceptions import RequestException, ConnectionError, Timeout

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SERVICE_URL = "https://arc.indiawris.gov.in/server/rest/services/SubInfoSysLCC/Litholog_Analysis_Depth_Thickness_Material/MapServer"
TN_LAYERS = {"depth": 9, "thickness": 16}
REQUEST_TIMEOUT = 30

def transform_coordinates(lon, lat):
    try:
        from pyproj import Transformer
        try:
            transformer = Transformer.from_crs("EPSG:4326", "ESRI:102024", always_xy=True)
        except Exception:
            service_wkt = '''PROJCS["WGS_1984_Lambert_Conformal_Conic",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic"],PARAMETER["False_Easting",4000000.0],PARAMETER["False_Northing",4000000.0],PARAMETER["Central_Meridian",80.0],PARAMETER["Standard_Parallel_1",12.472944],PARAMETER["Standard_Parallel_2",35.1728055],PARAMETER["Latitude_Of_Origin",24.0],UNIT["Meter",1.0]]'''
            transformer = Transformer.from_crs("EPSG:4326", service_wkt, always_xy=True)
        return transformer.transform(lon, lat)
    except Exception:
        x = 4000000.0 + (lon - 80.0) * 111320 * math.cos(math.radians(lat))
        y = 4000000.0 + (lat - 24.0) * 110540
        return x, y

def test_service_connection():
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://indiawris.gov.in/'
        })
        response = session.get(f"{SERVICE_URL}?f=json", timeout=REQUEST_TIMEOUT, verify=False)
        response.raise_for_status()
        service_info = response.json()
        print("[OK] Service is accessible. Service name: %s" % service_info.get('name', 'Unknown'))
        if 'layers' in service_info:
            for layer in service_info['layers']:
                if layer['id'] in [9,16]:
                    print("  Layer %d: %s" % (layer['id'], layer['name']))
        return True, session
    except Exception as e:
        print("[FAIL] Service connection failed: %s" % e)
        return False, None

def make_identify_request(session, params, verbose=False):
    identify_url = f"{SERVICE_URL}/identify"
    enhanced_params = {
        "geometry": params["geometry"],
        "geometryType": "esriGeometryPoint",
        "sr": "102024",
        "layers": params["layers"],
        "tolerance": 5,
        "mapExtent": params["mapExtent"],
        "imageDisplay": "400,400,96",
        "returnGeometry": "false",
        "returnFieldName": "true",
        "f": "json"
    }
    if verbose:
        print(f"  Request URL: {identify_url}")
        print(f"  Parameters: {json.dumps(enhanced_params, indent=2)}")
    try:
        response = session.get(identify_url, params=enhanced_params, timeout=REQUEST_TIMEOUT, verify=False)
        response.raise_for_status()
        if verbose:
            print(f"  Response status: {response.status_code}")
            print(f"  Response content: {response.text[:500]}...")
        return response.json()
    except Timeout:
        if verbose: print(f"[ERROR] Request timed out after {REQUEST_TIMEOUT} seconds.")
    except ConnectionError:
        if verbose: print(f"[ERROR] Connection failed. Server unreachable.")
    except RequestException as e:
        if verbose: print(f"[ERROR] HTTP error: {e}")
    except json.JSONDecodeError as e:
        if verbose: print(f"[ERROR] Invalid JSON: {e}")
    return None

def parse_identify_response(data, verbose=False):
    if not data:
        return None
    if verbose: print(f"  Response: {json.dumps(data)[:1000]}")
    if "error" in data:
        if verbose: print(f"[ERROR] {data['error']}")
        return None
    value_fields = ["Pixel Value", "pixel_value", "PIXEL_VALUE", "VALUE", "Value", "value", "DEPTH", "Depth", "depth", "THICKNESS", "Thickness", "thickness"]
    for result in data.get("results", []):
        attrs = result.get("attributes", {})
        for field in value_fields:
            if field in attrs and attrs[field] is not None:
                try:
                    val = float(attrs[field])
                    if val > 0:
                        if verbose: print(f"[OK] Value for {field}: {val}")
                        return val
                except (TypeError, ValueError):
                    continue
    if verbose: print("[INFO] No valid value found.")
    return None

def get_tn_aquifer_depth(lat, lon, include_thickness=False, verbose=False):
    service_available, session = test_service_connection()
    if not service_available:
        return {"latitude": lat, "longitude": lon, "aquifer_depth_m_bgl": None, "aquifer_thickness_m": None, "error": "Service not accessible"}
    x, y = transform_coordinates(lon, lat)
    if verbose: print(f"Transformed coordinates: ({x:.2f}, {y:.2f})")
    result = {"latitude": lat, "longitude": lon, "aquifer_depth_m_bgl": None, "aquifer_thickness_m": None, "error": None}
    buff = 5000
    params = {"geometry": f"{x},{y}", "layers": f"visible:{TN_LAYERS['depth']}", "mapExtent": f"{x-buff},{y-buff},{x+buff},{y+buff}"}
    if verbose: print(f"Querying depth layer {TN_LAYERS['depth']}...")
    data = make_identify_request(session, params, verbose)
    depth = parse_identify_response(data, verbose)
    if depth:
        result["aquifer_depth_m_bgl"] = depth
        if include_thickness:
            if verbose: print(f"Querying thickness layer {TN_LAYERS['thickness']}...")
            params["layers"] = f"visible:{TN_LAYERS['thickness']}"
            data = make_identify_request(session, params, verbose)
            thickness = parse_identify_response(data, verbose)
            if thickness:
                result["aquifer_thickness_m"] = thickness
    else:
        result["error"] = "No aquifer depth data at this location"
    return result

def test_multiple_locations():
    locations = [
        (12.9165, 79.1325, "Vellore City"),
        (13.0827, 80.2707, "Chennai"),
        (11.0168, 76.9558, "Coimbatore"),
        (10.7905, 78.7047, "Trichy"),
        (12.2958, 76.6394, "Mysore (border)")
    ]
    print("=== Tamil Nadu Aquifer Depth Analysis ===\n")
    print(f"{'Location':<20} {'Coordinates':<20} {'Depth (m bgl)':<15} {'Thickness (m)':<15} {'Status':<15}")
    print("-"*100)
    for lat, lon, name in locations:
        res = get_tn_aquifer_depth(lat, lon, include_thickness=True, verbose=False)
        depth_str = f"{res['aquifer_depth_m_bgl']:.2f}" if res['aquifer_depth_m_bgl'] else "N/A"
        thick_str = f"{res['aquifer_thickness_m']:.2f}" if res['aquifer_thickness_m'] else "N/A"
        status = "Success" if res['aquifer_depth_m_bgl'] else "No Data"
        print(f"{name:<20} ({lat:.3f}, {lon:.3f})    {depth_str:<15} {thick_str:<15} {status:<15}")

if __name__ == "__main__":
    test_multiple_locations()
