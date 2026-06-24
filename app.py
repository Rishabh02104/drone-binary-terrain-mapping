import streamlit as st
import cv2
import numpy as np
import torch
import os
import json
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Import model structures and inference functions
from surface_map import (
    SimpleCNN, 
    process_image_patch_cnn, 
    process_image_unet, 
    process_image_multiclass,
    trace_skeleton
)
from train_unet import UNet
from tiled_inference import run_tiled_inference
from osm_compare import fetch_osm_roads, evaluate_predictions


# Set page config
st.set_page_config(
    page_title="Drone Terrain Analysis & Measurement Studio",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (CSS) for premium look and feel
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
        color: #ffffff;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #3b82f6;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 14px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        color: #f8fafc;
        font-weight: 700;
    }
    .header-style {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3);
    }
    .header-title {
        color: white !important;
        font-size: 32px !important;
        font-weight: 800 !important;
        margin-bottom: 5px !important;
    }
    .header-subtitle {
        color: #bfdbfe !important;
        font-size: 16px !important;
    }
    .legend-box {
        display: inline-block;
        width: 15px;
        height: 15px;
        margin-right: 8px;
        border-radius: 3px;
        vertical-align: middle;
    }
    .legend-item {
        margin-bottom: 5px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to project pixel coordinate to GPS using telemetry
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

# Cache models
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cnn_model = None
    if os.path.exists("models/cnn_binary_model.pth"):
        cnn_model = SimpleCNN(num_classes=2).to(device)
        cnn_model.load_state_dict(torch.load("models/cnn_binary_model.pth", map_location=device))
        cnn_model.eval()
        
    unet_model = None
    if os.path.exists("models/unet_road_model.pth"):
        unet_model = UNet().to(device)
        unet_model.load_state_dict(torch.load("models/unet_road_model.pth", map_location=device))
        unet_model.eval()
        
    multiclass_model = None
    class_mapping = None
    if os.path.exists("models/cnn_multiclass_model.pth") and os.path.exists("models/multiclass_mapping.json"):
        with open("models/multiclass_mapping.json", "r") as f:
            class_mapping = json.load(f)
        multiclass_model = SimpleCNN(num_classes=len(class_mapping)).to(device)
        multiclass_model.load_state_dict(torch.load("models/cnn_multiclass_model.pth", map_location=device))
        multiclass_model.eval()
        
    return cnn_model, unet_model, multiclass_model, class_mapping, device

cnn_model, unet_model, multiclass_model, class_mapping, device = load_models()

# Premium Header
st.markdown("""
<div class="header-style">
    <div class="header-title">🛸 DRONE TERRAIN ANALYSIS & METRICS STUDIO</div>
    <div class="header-subtitle">Real-time Road Extraction, Curved Width Skeletonization, Manual Mask Labeling & GIS Export</div>
</div>
""", unsafe_allow_html=True)

# ======================================
# SIDEBAR
# ======================================
st.sidebar.header("⚙️ Configuration & Inputs")

model_option = st.sidebar.selectbox(
    "Choose Extraction Model",
    ["Patch-based CNN", "U-Net Semantic Segmentation"]
)

# Flight parameters
st.sidebar.subheader("✈️ Flight Telemetry")
altitude = st.sidebar.slider("Drone Altitude (meters)", 10, 200, 80, step=5)
fov = st.sidebar.slider("Camera Field of View (deg)", 30, 120, 84)

# Threshold and calibration parameters
st.sidebar.subheader("📐 Calibration & Confidence")
threshold = st.sidebar.slider("Road Probability Threshold", 0.1, 0.99, 0.65)
use_calibration = st.sidebar.checkbox("Use Known Road Width Calibration", True)
known_width = st.sidebar.slider("Known Road Width (meters)", 1.0, 50.0, 25.0, step=0.5)

# Tabs
tab_inf, tab_ann, tab_gis, tab_multi, tab_city = st.tabs([
    "🔍 Inference & Telemetry Analysis",
    "✏️ Manual Mask Annotation Studio",
    "🌍 GIS / GeoJSON Export",
    "🗺️ Multi-Class Terrain Map",
    "🏙️ City-Scale Mapping & OSM Compare"
])

# Shared image load helper
def get_loaded_image():
    source_img = st.radio("Image Source", ["Use Default Test Image", "Upload Custom Image"], key="src_image_radio")
    image = None
    if source_img == "Use Default Test Image":
        default_path = "images/test_image.jpg"
        if os.path.exists(default_path):
            image = cv2.imread(default_path)
        else:
            st.error("Default test image not found.")
    else:
        uploaded_file = st.file_uploader("Upload Drone Image (JPG/PNG)", type=["jpg", "png", "jpeg"], key="uploaded_file_loader")
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
    return image

# ======================================
# TAB 1: INFERENCE & TELEMETRY ANALYSIS
# ======================================
with tab_inf:
    st.subheader("Image Analysis Pipeline")
    
    col_input, col_run = st.columns([1, 2])
    
    with col_input:
        input_image = get_loaded_image()
        if input_image is not None:
            st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), caption="Source Image", use_column_width=True)
            
    with col_run:
        if input_image is None:
            st.info("Please load or upload an image to begin.")
        else:
            st.success("Image successfully loaded.")
            
            active_model = None
            if model_option == "Patch-based CNN":
                active_model = cnn_model
                model_name = "Patch CNN"
            else:
                active_model = unet_model
                model_name = "U-Net"
                
            if active_model is None:
                st.warning(f"⚠️ {model_option} model weights not found. Make sure the model exists in the models/ folder.")
            else:
                if st.button("🚀 Execute Analysis Pipeline"):
                    with st.spinner("Analyzing image..."):
                        if model_option == "Patch-based CNN":
                            road_mask, metrics, skeleton, class_blend, blended, combined_view = process_image_patch_cnn(
                                input_image, active_model, device,
                                patch_size=56, stride=16, road_threshold=threshold,
                                drone_height=altitude, fov_deg=fov,
                                use_calibration=use_calibration, known_road_width=known_width
                            )
                        else:
                            road_mask, metrics, skeleton, class_blend, blended, combined_view = process_image_unet(
                                input_image, active_model, device,
                                drone_height=altitude, fov_deg=fov,
                                use_calibration=use_calibration, known_road_width=known_width
                            )
                            
                        # Store in session state for GIS tab
                        st.session_state["last_skeleton"] = skeleton
                        st.session_state["last_road_mask"] = road_mask
                        st.session_state["last_metrics"] = metrics
                        st.session_state["img_shape"] = input_image.shape
                        
                        # Display metrics
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col4, m_col5, m_col6 = st.columns(3)
                        
                        with m_col1:
                            st.markdown(f'<div class="metric-card"><div class="metric-title">Length</div><div class="metric-value">{round(metrics["length_m"], 2)} m</div></div>', unsafe_allow_html=True)
                        with m_col2:
                            st.markdown(f'<div class="metric-card"><div class="metric-title">Average Width</div><div class="metric-value">{round(metrics["avg_width_m"], 2)} m</div></div>', unsafe_allow_html=True)
                        with m_col3:
                            st.markdown(f'<div class="metric-card"><div class="metric-title">Minimum Width</div><div class="metric-value">{round(metrics["min_width_m"], 2)} m</div></div>', unsafe_allow_html=True)
                        with m_col4:
                            st.markdown(f'<div class="metric-card"><div class="metric-title">Maximum Width</div><div class="metric-value">{round(metrics["max_width_m"], 2)} m</div></div>', unsafe_allow_html=True)
                        with m_col5:
                            st.markdown(f'<div class="metric-card"><div class="metric-title">Road Surface Area</div><div class="metric-value">{round(metrics["area_m2"], 2)} m²</div></div>', unsafe_allow_html=True)
                        with m_col6:
                            st.markdown(f'<div class="metric-card"><div class="metric-title">Road Coverage</div><div class="metric-value">{round(metrics["coverage_percent"], 2)}%</div></div>', unsafe_allow_html=True)
                            
                        # Visual outputs
                        st.subheader("Visual Mapping Outputs")
                        vis_col1, vis_col2 = st.columns(2)
                        
                        with vis_col1:
                            st.image(cv2.cvtColor(class_blend, cv2.COLOR_BGR2RGB), caption="Terrain Classification Overlay (Red = Road, Green = Non-road)", use_column_width=True)
                        with vis_col2:
                            st.image(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), caption="Final Output (Blue Centerline, Yellow Bounding Box)", use_column_width=True)
                            
                        # Downloads
                        st.subheader("💾 Export Image Data")
                        down_col1, down_col2 = st.columns(2)
                        
                        with down_col1:
                            _, encoded_mask = cv2.imencode('.png', road_mask)
                            st.download_button(
                                label="Download Road Mask (Binary PNG)",
                                data=encoded_mask.tobytes(),
                                file_name="road_mask.png",
                                mime="image/png"
                            )
                        with down_col2:
                            _, encoded_blended = cv2.imencode('.png', blended)
                            st.download_button(
                                label="Download Telemetry Annotated Image (PNG)",
                                data=encoded_blended.tobytes(),
                                file_name="annotated_output.png",
                                mime="image/png"
                            )

# ======================================
# TAB 2: MANUAL MASK ANNOTATION STUDIO
# ======================================
with tab_ann:
    st.subheader("Road Mask Labeling & Dataset Builder")
    st.markdown("""
    Create binary ground truth masks for training the U-Net model by drawing over the roads.
    Draw with **White** color for Road pixels. Paint with **Black** for Eraser.
    """)
    
    img_dir = "segmentation_dataset/Images"
    masks_dir = "segmentation_dataset/masks"
    
    if not os.path.exists(img_dir):
        st.error(f"Image directory '{img_dir}' does not exist. Please place your DJI drone images there.")
    else:
        files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.dng'))])
        
        if not files:
            st.warning("No images found in segmentation_dataset/Images.")
        else:
            annotated_files = []
            if os.path.exists(masks_dir):
                annotated_files = [os.path.splitext(f)[0] for f in os.listdir(masks_dir) if f.lower().endswith('.png')]
                
            progress = len(annotated_files)
            st.metric("Total Images Annotated", f"{progress} / {len(files)}")
            st.progress(progress / len(files))
            
            col_list, col_draw = st.columns([1, 3])
            
            with col_list:
                selected_file = st.selectbox("Select Image to Annotate", files)
                
                base_name = os.path.splitext(selected_file)[0]
                has_mask = base_name in annotated_files
                
                if has_mask:
                    st.success("✅ Already Annotated!")
                else:
                    st.info("❌ Annotation Pending")
                
                st.subheader("Brush Settings")
                stroke_width = st.slider("Brush Size", 2, 50, 15)
                tool_mode = st.radio("Tool Mode", ["Road (White Brush)", "Eraser (Black Brush)"])
                stroke_color = "#FFFFFF" if tool_mode == "Road (White Brush)" else "#000000"
                
            with col_draw:
                img_path = os.path.join(img_dir, selected_file)
                pil_img = Image.open(img_path).convert("RGB")
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 1.0)",
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_image=pil_img,
                    update_streamlit=True,
                    height=500,
                    width=750,
                    drawing_mode="freedraw",
                    key="drawable_canvas_" + selected_file,
                )
                
                if st.button("💾 Save Annotation Mask"):
                    if canvas_result.image_data is not None:
                        drawn_data = canvas_result.image_data
                        drawn_mask = (drawn_data[:, :, 0] > 0) | (drawn_data[:, :, 1] > 0) | (drawn_data[:, :, 2] > 0)
                        binary_mask = (drawn_mask * 255).astype(np.uint8)
                        
                        orig_w, orig_h = pil_img.size
                        full_res_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                        
                        os.makedirs(masks_dir, exist_ok=True)
                        dest_mask_path = os.path.join(masks_dir, f"{base_name}.png")
                        cv2.imwrite(dest_mask_path, full_res_mask)
                        
                        st.success(f"Successfully saved mask to {dest_mask_path}!")
                        st.experimental_rerun()

# ======================================
# TAB 3: GIS / GEOJSON EXPORT
# ======================================
with tab_gis:
    st.subheader("Georeferencing & Export")
    st.markdown("""
    Export drone road detections, skeleton centerlines, and bounding boxes to a standard **GeoJSON** file.
    Enter flight metadata to project pixel coordinate paths to true GPS latitude/longitude.
    """)
    
    col_coords, col_geojson = st.columns([1, 2])
    
    with col_coords:
        st.subheader("🌍 Reference Coordinates")
        mock_lat = st.number_input("Image Center Latitude", value=26.864000, format="%.6f", key="gis_lat_input")
        mock_lon = st.number_input("Image Center Longitude", value=80.978000, format="%.6f", key="gis_lon_input")
        mock_heading = st.slider("Drone Heading Angle (deg from North)", 0, 360, 0, key="gis_heading_slider")
        
    with col_geojson:
        if "last_metrics" not in st.session_state:
            st.info("Please run the analysis pipeline in Tab 1 first to generate road geometries.")
        else:
            metrics = st.session_state["last_metrics"]
            skeleton = st.session_state["last_skeleton"]
            road_mask = st.session_state["last_road_mask"]
            h, w, _ = st.session_state["img_shape"]
            meters_per_pixel = metrics["meters_per_pixel"]
            
            st.success("✅ Road Geometry Available for Geoprojection")
            
            if st.button("🗺️ Generate GeoJSON Data"):
                with st.spinner("Projecting pixel coordinates to GPS..."):
                    ordered_pts = trace_skeleton(skeleton)
                    
                    sample_step = 10
                    sampled_pts = ordered_pts[::sample_step]
                    if ordered_pts and ordered_pts[-1] not in sampled_pts:
                        sampled_pts.append(ordered_pts[-1])
                        
                    gps_centerline = []
                    for y, x in sampled_pts:
                        lat, lon = pixel_to_gps(x, y, w, h, mock_lat, mock_lon, mock_heading, meters_per_pixel)
                        gps_centerline.append([lon, lat])
                        
                    # Bounding Box
                    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    gps_bbox = []
                    if len(contours) > 0:
                        largest_contour = max(contours, key=cv2.contourArea)
                        rect = cv2.minAreaRect(largest_contour)
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        
                        for pt in box:
                            lat, lon = pixel_to_gps(pt[0], pt[1], w, h, mock_lat, mock_lon, mock_heading, meters_per_pixel)
                            gps_bbox.append([lon, lat])
                        if len(gps_bbox) > 0:
                            gps_bbox.append(gps_bbox[0])
                            
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
                            "drone_altitude_m": altitude,
                            "camera_fov_deg": fov,
                            "scale_m_per_px": meters_per_pixel,
                            "center_lat": mock_lat,
                            "center_lon": mock_lon,
                            "heading_deg": mock_heading
                        },
                        "features": features
                    }
                    
                    st.session_state["geojson_export"] = json.dumps(geojson_data, indent=2)
                    
            if "geojson_export" in st.session_state:
                st.subheader("GeoJSON Data Preview")
                st.code(st.session_state["geojson_export"][:600] + "\n... (truncated preview) ...", language="json")
                
                st.download_button(
                    label="💾 Download Full GeoJSON File",
                    data=st.session_state["geojson_export"],
                    file_name="drone_road_measurement.geojson",
                    mime="application/json"
                )

# ======================================
# TAB 4: MULTI-CLASS TERRAIN MAP
# ======================================
with tab_multi:
    st.subheader("9-Class Land Coverage Classification")
    st.markdown("""
    Classify drone image patches into 9 distinct land coverage and terrain categories:
    **Buildings**, **Cemented road**, **Coloured paver path**, **Grass Ground**, **Non-Road**, **Road**, **Sand path**, **Vegetations**, and **Solar panels**.
    """)
    
    col_m_input, col_m_run = st.columns([1, 2])
    
    with col_m_input:
        # Load image (separate file uploader so states don't clash)
        default_p = "images/test_image.jpg"
        multiclass_img = cv2.imread(default_p)
        if multiclass_img is not None:
            st.image(cv2.cvtColor(multiclass_img, cv2.COLOR_BGR2RGB), caption="Classification Image", use_column_width=True)
            
    with col_m_run:
        if multiclass_model is None or class_mapping is None:
            st.warning("⚠️ Multi-Class CNN Model weights or class mapping JSON not found. Train the model using `python train_cnn.py --mode multiclass`.")
        else:
            st.success("✅ Multi-Class CNN Model Loaded Successfully.")
            
            # Show Color Legend
            st.subheader("🎨 Terrain Class Color Legend")
            legend_cols = st.columns(3)
            
            colors_html = {
                "Buildings": ("#808080", "Buildings"),
                "Cemented road": ("#dcbebe", "Cemented Road"),
                "Coloured paver path": ("#b469ff", "Coloured Pavers"),
                "Grass Ground": ("#32dc32", "Grass Ground"),
                "Non-Road": ("#1e5096", "Non-Road (General)"),
                "Road": ("#ff0000", "Asphalt Road"),
                "Sand path": ("#64dcdc", "Sand Path"),
                "Vegetations": ("#006400", "Vegetations / Trees"),
                "solar panels": ("#0000ff", "Solar Panels")
            }
            
            for i, (name, (hex_c, label)) in enumerate(colors_html.items()):
                col_idx = i % 3
                with legend_cols[col_idx]:
                    st.markdown(f'<div class="legend-item"><div class="legend-box" style="background-color: {hex_c};"></div>{label}</div>', unsafe_allow_html=True)
                    
            if st.button("🚀 Run Multi-Class Classification"):
                with st.spinner("Running 9-class CNN inference..."):
                    multiclass_map, blended_map = process_image_multiclass(
                        multiclass_img, multiclass_model, device, class_mapping,
                        patch_size=56, stride=16
                    )
                    
                    # Display Classification Overlay
                    st.subheader("Land Classification Overlay")
                    st.image(cv2.cvtColor(blended_map, cv2.COLOR_BGR2RGB), caption="Terrain Map (50% Original / 50% Classification Layer)", use_column_width=True)
                    
                    # Calculate Coverage Statistics
                    # Extract patch classifications by colors
                    h, w, _ = multiclass_map.shape
                    total_pixels = h * w
                    
                    st.subheader("📊 Land Coverage Distribution")
                    
                    # Compute percentage of each color
                    color_mapping_rgb = {
                        "Buildings": [128, 128, 128],
                        "Cemented road": [220, 220, 220],
                        "Coloured paver path": [255, 105, 180], # BGR to RGB conversion handling
                        "Grass Ground": [50, 220, 50],
                        "Non-Road": [150, 80, 30],
                        "Road": [255, 0, 0],
                        "Sand path": [220, 220, 100],
                        "Vegetations": [0, 100, 0],
                        "solar panels": [0, 0, 255]
                    }
                    
                    # Flat classification array matching BGR colors
                    flat_map = multiclass_map.reshape(-1, 3)
                    
                    # Count matching patches (by unique colors)
                    unique_colors, counts = np.unique(flat_map, axis=0, return_counts=True)
                    
                    # Print progress bars for classes present
                    stats = {}
                    for class_name, rgb in color_mapping_rgb.items():
                        bgr = rgb[::-1]
                        # Find matching count
                        match_count = 0
                        for uc, count in zip(unique_colors, counts):
                            if np.array_equal(uc, bgr):
                                match_count = count
                                break
                        percent = (match_count / total_pixels) * 100
                        if percent > 0.05: # only show if present
                            stats[class_name] = percent
                            
                    # Sort stats by percentage
                    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
                    
                    for name, pct in sorted_stats:
                        hex_c = colors_html[name][0]
                        st.markdown(f"**{name}**: {round(pct, 2)}%")
                        st.progress(pct / 100.0)
                        
                    # Save / Download classification image
                    _, encoded_blend = cv2.imencode('.png', blended_map)
                    st.download_button(
                        label="Download Terrain Coverage Map (PNG)",
                        data=encoded_blend.tobytes(),
                        file_name="multiclass_terrain_map.png",
                        mime="image/png"
                    )

# ======================================
# TAB 5: CITY-SCALE MAPPING & OSM COMPARE
# ======================================
with tab_city:
    st.subheader("Auroville Spiral Road Network Mapping & OSM Verification")
    st.markdown("""
    Run sliding-window tiled inference on large city-scale aerial maps of **Auroville, Tamil Nadu** (Spiral layout) and validate predicted centerlines against the live **OpenStreetMap (OSM)** database.
    """)
    
    col_c1, col_c2 = st.columns([1, 2])
    with col_c1:
        st.subheader("🌍 Target Coordinates (Auroville Center)")
        city_lat = st.number_input("Center Latitude", value=12.006900, format="%.6f", key="city_lat_val")
        city_lon = st.number_input("Center Longitude", value=79.810500, format="%.6f", key="city_lon_val")
        city_radius = st.slider("Query Bounding Box Radius (km)", 0.5, 3.0, 1.5, step=0.1, key="city_radius_val")
        city_scale = st.number_input("Map Scale (meters per pixel)", value=0.30, format="%.4f", key="city_scale_val")
        city_heading = st.slider("Drone Heading (degrees)", 0, 360, 0, key="city_heading_val")
        
        # Select image
        source_city = st.radio("Select Map Image Source", ["Use Current Test Image", "Upload Large Custom Image"], key="src_city_radio")
        city_img = None
        if source_city == "Use Current Test Image":
            city_img = cv2.imread("images/test_image.jpg")
        else:
            uploaded_city = st.file_uploader("Upload Large Map Image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="city_upload")
            if uploaded_city is not None:
                file_bytes = np.asarray(bytearray(uploaded_city.read()), dtype=np.uint8)
                city_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
        if city_img is not None:
            st.image(cv2.cvtColor(city_img, cv2.COLOR_BGR2RGB), caption="Large Map Input", use_column_width=True)
            
    with col_c2:
        if unet_model is None:
            st.warning("⚠️ U-Net Model weights not found. Please train U-Net using Tab 1/CLI first.")
        else:
            st.success("✅ U-Net Semantic Segmentation Model Ready.")
            
            if st.button("🚀 Run Tiled Inference & Fetch OSM Roads", key="run_city_btn"):
                if city_img is None:
                    st.error("Please load a map image first.")
                else:
                    with st.spinner("Executing sliding-window tiled U-Net inference on map..."):
                        # Run sliding window U-Net inference
                        road_mask = run_tiled_inference(city_img, unet_model, device, tile_size=256, overlap=32)
                        
                        # Calculate scale parameters based on inputs
                        h, w, _ = city_img.shape
                        fov_deg = 84.0
                        fov_rad = np.radians(fov_deg)
                        height = (city_scale * w) / (2 * np.tan(fov_rad / 2))
                        
                        _, metrics, skeleton, _, blended, combined_view = extract_metrics_and_visuals(
                            city_img, road_mask, drone_height=height, fov_deg=fov_deg, use_calibration=True
                        )
                        
                        # Display combined overlay
                        st.subheader("Model-Predicted Road Segmentation & Centerline")
                        st.image(cv2.cvtColor(combined_view, cv2.COLOR_BGR2RGB), caption="Prediction Overlay (Blue=Centerline, Yellow=Bounding Box)", use_column_width=True)
                        
                        # Write local files
                        pred_geojson_path = "outputs/predicted_roads.geojson"
                        osm_geojson_path = "outputs/osm_roads.geojson"
                        
                        # Export georeferenced GeoJSON
                        from tiled_inference import export_prediction_geojson
                        export_prediction_geojson(skeleton, road_mask, metrics, city_lat, city_lon, city_heading, city_scale, pred_geojson_path)
                        
                    with st.spinner("Fetching live road database from OpenStreetMap..."):
                        # Fetch OSM data
                        fetch_success = fetch_osm_roads(city_lat, city_lon, city_radius, osm_geojson_path)
                        
                    if fetch_success:
                        # Compare vectors
                        with st.spinner("Calculating Precision & Recall against OSM database..."):
                            comp_metrics = evaluate_predictions(pred_geojson_path, osm_geojson_path, distance_threshold_m=15.0)
                            
                        if comp_metrics:
                            st.subheader("📊 Comparison & Accuracy Analytics")
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("Map Precision (Matches OSM)", f"{comp_metrics['precision']*100:.1f}%")
                            with col_m2:
                                st.metric("Map Recall (OSM Coverage)", f"{comp_metrics['recall']*100:.1f}%")
                            with col_m3:
                                st.metric("F1-Score", f"{comp_metrics['f1_score']:.3f}")
                                
                            st.info(f"💡 **Unmapped Road Candidates**: Detected **{comp_metrics['new_road_candidates']}** coordinates that are not present in the current OpenStreetMap database. These represent potential new roads or unmapped structures!")
                            
                            # Download Buttons
                            with open(pred_geojson_path, "r") as f:
                                pred_data = f.read()
                            with open(osm_geojson_path, "r") as f:
                                osm_data = f.read()
                                
                            st.subheader("💾 Export GIS Layers")
                            col_d1, col_d2 = st.columns(2)
                            with col_d1:
                                st.download_button(
                                    label="💾 Download AI Predicted Road GeoJSON",
                                    data=pred_data,
                                    file_name="auroville_predicted_roads.geojson",
                                    mime="application/json"
                                )
                            with col_d2:
                                st.download_button(
                                    label="💾 Download OSM Road Database GeoJSON",
                                    data=osm_data,
                                    file_name="auroville_osm_roads.geojson",
                                    mime="application/json"
                                )

