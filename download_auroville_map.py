import urllib.request
import numpy as np
import cv2
import os
import json

# Convert WGS84 coordinates to OSM/Slippy map tile coordinates
def latlon_to_tile(lat, lon, zoom):
    lat_rad = np.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - np.log(np.tan(lat_rad) + (1.0 / np.cos(lat_rad))) / np.pi) / 2.0 * n)
    return xtile, ytile

# Convert tile coordinates back to WGS84 coordinates (top-left of tile)
def tile_to_latlon(x, y, zoom):
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1.0 - 2.0 * y / n)))
    lat = np.degrees(lat_rad)
    return lat, lon

# Download and stitch satellite tiles of Auroville
def get_auroville_satellite_map(lat=12.006900, lon=79.810500, zoom=17, grid_size=3, output_path="images/auroville_satellite.jpg"):
    center_x, center_y = latlon_to_tile(lat, lon, zoom)
    
    # Calculate offset range to query a grid
    half_grid = grid_size // 2
    start_x = center_x - half_grid
    start_y = center_y - half_grid
    
    print(f"Auroville center tile at zoom {zoom}: X={center_x}, Y={center_y}")
    print(f"Downloading a {grid_size}x{grid_size} tile grid from Esri World Imagery...")
    
    # Grid tile stitch container (256 pixels per tile)
    tile_pixels = 256
    stitched_image = np.zeros((grid_size * tile_pixels, grid_size * tile_pixels, 3), dtype=np.uint8)
    
    # Esri World Imagery Map Server URL
    url_template = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    for row in range(grid_size):
        y_tile = start_y + row
        for col in range(grid_size):
            x_tile = start_x + col
            url = url_template.format(zoom=zoom, x=x_tile, y=y_tile)
            
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as response:
                    img_data = response.read()
                    
                # Decode image
                arr = np.asarray(bytearray(img_data), dtype=np.uint8)
                tile_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                
                if tile_img is not None:
                    # Paste into stitched container
                    stitched_image[row*tile_pixels:(row+1)*tile_pixels, col*tile_pixels:(col+1)*tile_pixels] = tile_img
                else:
                    print(f"Warning: Failed to decode tile X={x_tile}, Y={y_tile}")
            except Exception as e:
                print(f"Error downloading tile X={x_tile}, Y={y_tile}: {e}")
                
    # Save the stitched map
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, stitched_image)
    
    # Calculate exact scale (meters per pixel) at this latitude
    r_earth = 6378137.0
    # Circumference at equator: 2 * pi * r_earth
    # Scale decreases with latitude cosine
    meters_per_pixel = (np.cos(np.radians(lat)) * 2 * np.pi * r_earth) / (tile_pixels * (2.0 ** zoom))
    
    # Calculate exact georeference bounds
    top_left_lat, top_left_lon = tile_to_latlon(start_x, start_y, zoom)
    bottom_right_lat, bottom_right_lon = tile_to_latlon(start_x + grid_size, start_y + grid_size, zoom)
    
    print("\n" + "="*40)
    print("      AUROVILLE SATELLITE MAP DATA")
    print("="*40)
    print(f"Saved Map to:      {output_path}")
    print(f"Resolution:        {stitched_image.shape[1]}x{stitched_image.shape[0]} px")
    print(f"Meters per Pixel:  {meters_per_pixel:.4f} m/px")
    print(f"Top-Left Coordinate:     {top_left_lat:.6f}, {top_left_lon:.6f}")
    print(f"Bottom-Right Coordinate:  {bottom_right_lat:.6f}, {bottom_right_lon:.6f}")
    print("="*40)
    
    # Write a metadata JSON file next to it for reference
    meta = {
        "center_lat": lat,
        "center_lon": lon,
        "zoom": zoom,
        "meters_per_pixel": meters_per_pixel,
        "top_left_lat": top_left_lat,
        "top_left_lon": top_left_lon,
        "bottom_right_lat": bottom_right_lat,
        "bottom_right_lon": bottom_right_lon
    }
    with open(output_path.replace(".jpg", "_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    return True

if __name__ == "__main__":
    get_auroville_satellite_map()
