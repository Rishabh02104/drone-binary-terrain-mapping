import os
import json
import urllib.request
import urllib.parse
import numpy as np

# Bounding box calculation based on center coordinate and radius in km
def get_bbox(lat, lon, radius_km=1.5):
    # Earth radius in km
    r_earth = 6378.137
    # Coordinate offsets in radians
    d_lat = radius_km / r_earth
    d_lon = radius_km / (r_earth * np.cos(np.pi * lat / 180.0))
    
    # Offsets in degrees
    offset_lat = d_lat * (180.0 / np.pi)
    offset_lon = d_lon * (180.0 / np.pi)
    
    return {
        "min_lat": lat - offset_lat,
        "max_lat": lat + offset_lat,
        "min_lon": lon - offset_lon,
        "max_lon": lon + offset_lon
    }

# Query OpenStreetMap Overpass API
def fetch_osm_roads(lat, lon, radius_km=1.5, output_file="outputs/osm_roads.geojson"):
    bbox = get_bbox(lat, lon, radius_km)
    
    # Overpass QL Query
    query = f"""
    [out:json][timeout:25];
    (
      way["highway"]({bbox["min_lat"]:.6f},{bbox["min_lon"]:.6f},{bbox["max_lat"]:.6f},{bbox["max_lon"]:.6f});
    );
    out body;
    >;
    out skel qt;
    """
    
    print(f"Fetching road data from OpenStreetMap for bbox: lat=[{bbox['min_lat']:.4f}, {bbox['max_lat']:.4f}], lon=[{bbox['min_lon']:.4f}, {bbox['max_lon']:.4f}]...")
    
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.osm.ch/api/interpreter",
        "https://overpass.nchc.org.tw/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter"
    ]
    
    osm_data = None
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    
    for url in endpoints:
        print(f"Trying server: {url}...")
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "DroneTerrainMapper/1.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                raw_response = response.read().decode("utf-8")
                temp_data = json.loads(raw_response)
                
            elements = temp_data.get("elements", [])
            if len(elements) == 0:
                print(f"Server {url} returned 0 elements, trying fallback...")
                continue
                
            osm_data = temp_data
            print(f"Successfully fetched {len(elements)} elements from {url}.")
            break
        except Exception as e:
            print(f"Server {url} failed or timed out: {e}")
            continue
            
    if osm_data is None:
        print("Error: All Overpass API servers failed, timed out, or returned no data.")
        return False


        
    # Convert OSM JSON to GeoJSON LineStrings
    nodes = {}
    ways = []
    
    for element in osm_data.get("elements", []):
        if element["type"] == "node":
            nodes[element["id"]] = (element["lat"], element["lon"])
        elif element["type"] == "way":
            ways.append(element)
            
    features = []
    for way in ways:
        way_nodes = way.get("nodes", [])
        coordinates = []
        for node_id in way_nodes:
            if node_id in nodes:
                lat_n, lon_n = nodes[node_id]
                coordinates.append([lon_n, lat_n]) # GeoJSON uses [longitude, latitude]
                
        if len(coordinates) > 1:
            properties = {
                "osm_id": way["id"],
                "highway": way.get("tags", {}).get("highway", "unclassified"),
                "name": way.get("tags", {}).get("name", "Unnamed Road"),
                "surface": way.get("tags", {}).get("surface", "unknown")
            }
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": properties
            })
            
    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "center_lat": lat,
            "center_lon": lon,
            "radius_km": radius_km,
            "bbox": bbox
        },
        "features": features
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(geojson, f, indent=2)
        
    print(f"Successfully saved {len(features)} road segments to {output_file}.")
    return True

# Project coordinates (lat, lon) to local Cartesian coordinates (x, y) in meters
def project_coords(coords, ref_lat, ref_lon):
    r_earth = 6378137.0
    lat_rad = np.radians(ref_lat)
    
    projected = []
    for lon, lat in coords:
        dx = (lon - ref_lon) * (r_earth * np.cos(lat_rad)) * (np.pi / 180.0)
        dy = (lat - ref_lat) * r_earth * (np.pi / 180.0)
        projected.append([dx, dy])
        
    return np.array(projected)

# Find minimum distance from a point to a line segment AB
def point_to_segment_distance(p, a, b):
    ab = b - a
    ap = p - a
    ab_len_sq = np.sum(ab**2)
    
    if ab_len_sq == 0:
        return np.sqrt(np.sum(ap**2))
        
    # Project point onto segment (t is the projection ratio, clamped to [0, 1])
    t = max(0, min(1, np.dot(ap, ab) / ab_len_sq))
    projection = a + t * ab
    return np.sqrt(np.sum((p - projection)**2))

# Compute distance from point to a set of LineStrings
def point_to_geojson_distance(p, lines_projected):
    min_dist = float('inf')
    for line in lines_projected:
        if len(line) < 2:
            continue
        for i in range(len(line) - 1):
            dist = point_to_segment_distance(p, line[i], line[i+1])
            if dist < min_dist:
                min_dist = dist
    return min_dist

# Evaluate prediction against OSM road network
def evaluate_predictions(predicted_geojson_path, osm_geojson_path, distance_threshold_m=15.0):
    if not os.path.exists(predicted_geojson_path) or not os.path.exists(osm_geojson_path):
        print("Error: Files not found for comparison.")
        return None
        
    with open(predicted_geojson_path, "r") as f:
        pred_data = json.load(f)
    with open(osm_geojson_path, "r") as f:
        osm_data = json.load(f)
        
    ref_lat = osm_data["metadata"]["center_lat"]
    ref_lon = osm_data["metadata"]["center_lon"]
    
    # Extract prediction centerline points
    pred_pts = []
    for feature in pred_data.get("features", []):
        if feature["properties"].get("feature_type") == "road_centerline":
            pred_pts.extend(feature["geometry"]["coordinates"])
            
    if not pred_pts:
        print("No predicted road centerline found in prediction file.")
        return None
        
    # Extract OSM roads as lines
    osm_lines = []
    osm_nodes = []
    for feature in osm_data.get("features", []):
        coords = feature["geometry"]["coordinates"]
        osm_lines.append(coords)
        osm_nodes.extend(coords)
        
    if not osm_lines:
        print("No roads found in OSM file.")
        return None
        
    # Project points to local meters
    pred_pts_m = project_coords(pred_pts, ref_lat, ref_lon)
    osm_lines_m = [project_coords(line, ref_lat, ref_lon) for line in osm_lines]
    osm_nodes_m = project_coords(osm_nodes, ref_lat, ref_lon)
    
    print(f"Comparing {len(pred_pts)} predicted points against {len(osm_lines)} OSM road segments...")
    
    # Calculate Precision: fraction of predicted points near an OSM road
    matched_pred = 0
    for p in pred_pts_m:
        dist = point_to_geojson_distance(p, osm_lines_m)
        if dist <= distance_threshold_m:
            matched_pred += 1
            
    precision = matched_pred / len(pred_pts) if pred_pts else 0
    
    # Calculate Recall: fraction of OSM nodes near a predicted centerline point
    matched_osm = 0
    for o in osm_nodes_m:
        # Distance to closest predicted point
        dists = np.sqrt(np.sum((pred_pts_m - o)**2, axis=1))
        if len(dists) > 0 and np.min(dists) <= distance_threshold_m:
            matched_osm += 1
            
    recall = matched_osm / len(osm_nodes) if osm_nodes else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "total_predicted_coords": len(pred_pts),
        "total_osm_coords": len(osm_nodes),
        "matched_prediction_coords": matched_pred,
        "matched_osm_coords": matched_osm,
        "new_road_candidates": len(pred_pts) - matched_pred
    }
    
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download OSM road networks and compare with predicted georeferenced road GeoJSON.")
    parser.add_argument("--lat", type=float, default=12.006900, help="Center latitude (Auroville default: 12.006900)")
    parser.add_argument("--lon", type=float, default=79.810500, help="Center longitude (Auroville default: 79.810500)")
    parser.add_argument("--radius", type=float, default=1.5, help="Query radius in km")
    parser.add_argument("--pred", type=str, default="outputs/predicted_roads.geojson", help="Path to predicted road GeoJSON")
    parser.add_argument("--output", type=str, default="outputs/osm_roads.geojson", help="Path to save fetched OSM GeoJSON")
    parser.add_argument("--threshold", type=float, default=15.0, help="Matching distance threshold in meters")
    args = parser.parse_args()
    
    # Fetch OSM data
    success = fetch_osm_roads(args.lat, args.lon, args.radius, args.output)
    
    if success and os.path.exists(args.pred):
        results = evaluate_predictions(args.pred, args.output, args.threshold)
        if results:
            print("\n" + "="*40)
            print("         OSM COMPARISON METRICS")
            print("="*40)
            print(f"Map Overlay Precision: {results['precision']*100:.2f}% (Matches existing OSM roads)")
            print(f"Map Overlay Recall:    {results['recall']*100:.2f}% (OSM road coverage)")
            print(f"F1-Score:              {results['f1_score']:.4f}")
            print("-"*40)
            print(f"Detected New/Unmapped Road Coordinates: {results['new_road_candidates']}")
            print("="*40)
