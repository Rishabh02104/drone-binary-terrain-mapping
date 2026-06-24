import os
import cv2
import torch
import json
import numpy as np
from PIL import Image
from train_unet import UNet
from surface_map import extract_metrics_and_visuals, trace_skeleton

# Helper function to project pixel coordinate to GPS
def pixel_to_gps(x, y, w, h, lat_c, lon_c, heading_deg, meters_per_pixel):
    cx, cy = w / 2.0, h / 2.0
    dx = (x - cx) * meters_per_pixel
    dy = (cy - y) * meters_per_pixel  # image y increases downwards
    
    rad = np.radians(heading_deg)
    d_east = dx * np.cos(rad) + dy * np.sin(rad)
    d_north = -dx * np.sin(rad) + dy * np.cos(rad)
    
    r_earth = 6378137.0
    d_lat = d_north / r_earth * (180.0 / np.pi)
    d_lon = d_east / (r_earth * np.cos(np.radians(lat_c))) * (180.0 / np.pi)
    
    return lat_c + d_lat, lon_c + d_lon

# Custom tiled inference engine
# Custom tiled inference engine
def run_tiled_inference(image_or_path, model, device, tile_size=256, overlap=32, batch_size=16):
    is_path = isinstance(image_or_path, str)
    
    if is_path:
        # Lazy load using PIL (doesn't load pixel data into RAM immediately)
        img = Image.open(image_or_path)
        w, h = img.size
    else:
        # Image is a numpy array (BGR from OpenCV)
        h, w, _ = image_or_path.shape
        img = Image.fromarray(cv2.cvtColor(image_or_path, cv2.COLOR_BGR2RGB))
        
    stride = tile_size - 2 * overlap
    
    # Calculate number of tiles needed in each dimension
    num_tiles_y = max(1, int(np.ceil((h - tile_size) / stride)) + 1)
    num_tiles_x = max(1, int(np.ceil((w - tile_size) / stride)) + 1)
    
    # Calculate target padded dimensions
    target_padded_h = (num_tiles_y - 1) * stride + tile_size
    target_padded_w = (num_tiles_x - 1) * stride + tile_size
    
    # Create padded mask of size equal to target padded dimensions
    padded_mask = np.zeros((target_padded_h, target_padded_w), dtype=np.float32)
    
    tiles = []
    positions = []
    
    # Extract tiles using PIL crop (automatically handles padding on borders)
    for row in range(num_tiles_y):
        y_pad = row * stride
        y_start = y_pad - overlap
        y_end = y_start + tile_size
        
        for col in range(num_tiles_x):
            x_pad = col * stride
            x_start = x_pad - overlap
            x_end = x_start + tile_size
            
            # Crop tile (PIL handles negative coordinates by padding with black/0)
            tile = img.crop((x_start, y_start, x_end, y_end))
            tile_np = np.array(tile)
            
            # Ensure shape is 3D RGB (H, W, 3)
            if len(tile_np.shape) == 2:
                tile_np = np.stack([tile_np] * 3, axis=-1)
            elif tile_np.shape[2] == 4:
                tile_np = tile_np[:, :, :3]
                
            tiles.append(tile_np)
            positions.append((y_pad, x_pad))
            
    print(f"Extracted {len(tiles)} tiles of size {tile_size}x{tile_size} for sliding window inference.")
    
    # Process in batches
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i:i+batch_size]
        batch_pos = positions[i:i+batch_size]
        
        tensor_list = []
        for t in batch_tiles:
            # t is in RGB (from PIL), convert to FloatTensor
            t_tensor = torch.from_numpy(t).permute(2, 0, 1).float() / 255.0
            tensor_list.append(t_tensor)
            
        tensor_batch = torch.stack(tensor_list).to(device)
        
        with torch.no_grad():
            outputs = model(tensor_batch)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            
        if len(probs.shape) == 2:
            probs = np.expand_dims(probs, axis=0)
            
        # Paste predictions back (inner stride area only)
        for idx, (y_pad, x_pad) in enumerate(batch_pos):
            inner_prob = probs[idx, overlap : tile_size-overlap, overlap : tile_size-overlap]
            padded_mask[y_pad : y_pad+stride, x_pad : x_pad+stride] = inner_prob
            
    # Crop back to the original image dimensions
    final_prob = padded_mask[0:h, 0:w]
    road_mask = (final_prob > 0.5).astype(np.uint8) * 255
    return road_mask


# Export predicted coordinates to standard GeoJSON
def export_prediction_geojson(skeleton, road_mask, metrics, lat, lon, heading, meters_per_pixel, output_file):
    h, w = road_mask.shape
    ordered_pts = trace_skeleton(skeleton)
    
    # Sample centerline points to keep file sizes clean
    sample_step = 10
    sampled_pts = ordered_pts[::sample_step]
    if ordered_pts and ordered_pts[-1] not in sampled_pts:
        sampled_pts.append(ordered_pts[-1])
        
    gps_centerline = []
    for y, x in sampled_pts:
        lat_pt, lon_pt = pixel_to_gps(x, y, w, h, lat, lon, heading, meters_per_pixel)
        gps_centerline.append([lon_pt, lat_pt])
        
    # Extract Bounding Box
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gps_bbox = []
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        for pt in box:
            lat_pt, lon_pt = pixel_to_gps(pt[0], pt[1], w, h, lat, lon, heading, meters_per_pixel)
            gps_bbox.append([lon_pt, lat_pt])
        if len(gps_bbox) > 0:
            gps_bbox.append(gps_bbox[0]) # close polygon
            
    features = []
    if len(gps_centerline) > 1:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": gps_centerline
            },
            "properties": {
                "feature_type": "road_centerline",
                "length_m": metrics["length_m"],
                "avg_width_m": metrics["avg_width_m"],
                "min_width_m": metrics["min_width_m"],
                "max_width_m": metrics["max_width_m"]
            }
        })
        
    if len(gps_bbox) > 1:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [gps_bbox]
            },
            "properties": {
                "feature_type": "road_bounding_box",
                "area_m2": metrics["area_m2"],
                "coverage_percent": metrics["coverage_percent"]
            }
        })
        
    geojson_data = {
        "type": "FeatureCollection",
        "metadata": {
            "center_lat": lat,
            "center_lon": lon,
            "heading_deg": heading,
            "scale_m_per_px": meters_per_pixel
        },
        "features": features
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(geojson_data, f, indent=2)
    print(f"Exported prediction road geometries to {output_file}.")
    return geojson_data

# Main CLI execution
def process_large_image(image_path, model_path, lat, lon, heading, meters_per_pixel, output_mask_path, output_geojson_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running tiled inference on {device}...")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return False
        
    # Load U-Net Model
    model = UNet().to(device)
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return False
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Run sliding window
    road_mask = run_tiled_inference(image, model, device, tile_size=256, overlap=32)
    
    # Compute metrics & overlays using calibrated projection parameters
    # known_road_width and meters_per_pixel will be derived from metadata
    h, w, _ = image.shape
    
    # Calibrate meters_per_pixel based on inputs
    # If the user sets meters_per_pixel directly, we use it. Otherwise, we calculate from height & fov
    # Let's run extract_metrics_and_visuals
    # In surface_map.py, extract_metrics_and_visuals accepts height and fov_deg
    # We will pass dummy values and scale metrics manually using our meters_per_pixel
    # surface_map metrics are scaled by meters_per_pixel = Ground Width / Image Width
    # We will extract metrics using surface_map function:
    # let's calculate height & fov to match the target meters_per_pixel:
    # meters_per_pixel = (2 * height * tan(fov/2)) / image_width
    # We can just compute a height that satisfies the desired meters_per_pixel for an 84-degree FOV:
    fov_deg = 84.0
    fov_rad = np.radians(fov_deg)
    height = (meters_per_pixel * w) / (2 * np.tan(fov_rad / 2))
    
    # Run core geometric metric extractor
    _, metrics, skeleton, _, blended, combined_view = extract_metrics_and_visuals(
        image, road_mask, drone_height=height, fov_deg=fov_deg, use_calibration=True
    )
    
    # Save outputs
    cv2.imwrite(output_mask_path, combined_view)
    print(f"Saved visualization output to {output_mask_path}.")
    
    # Export georeferenced GeoJSON
    export_prediction_geojson(skeleton, road_mask, metrics, lat, lon, heading, meters_per_pixel, output_geojson_path)
    
    print("\nTiled Inference Metrics:")
    print(json.dumps(metrics, indent=2))
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tiled Inference for Large Scale Aerial Maps.")
    parser.add_argument("--image", type=str, default="images/test_image.jpg", help="Path to large image")
    parser.add_argument("--model", type=str, default="models/unet_road_model.pth", help="Path to model weights")
    parser.add_argument("--lat", type=float, default=12.006900, help="Center latitude (Auroville default: 12.006900)")
    parser.add_argument("--lon", type=float, default=79.810500, help="Center longitude (Auroville default: 79.810500)")
    parser.add_argument("--heading", type=float, default=0.0, help="Drone heading angle in degrees")
    parser.add_argument("--scale", type=float, default=0.3, help="Meters per pixel scale")
    parser.add_argument("--output_mask", type=str, default="outputs/tiled_road_mask.jpg", help="Path to save output overlay")
    parser.add_argument("--output_geojson", type=str, default="outputs/predicted_roads.geojson", help="Path to save output GeoJSON")
    args = parser.parse_args()
    
    process_large_image(
        args.image, args.model, args.lat, args.lon, args.heading, args.scale,
        args.output_mask, args.output_geojson
    )
