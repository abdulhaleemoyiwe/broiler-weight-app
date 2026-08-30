import cv2
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from ultralytics import YOLO

# --- Page Configuration & CSS Styling ---
st.set_page_config(
    page_title="Broiler Weight predictor", 
    page_icon="🐔",  
    layout="centered"
)


st.markdown(f'<link rel="manifest" href="https://raw.githubusercontent.com/abdulhaleemoyiwe/broiler-weight-app/refs/heads/main/manifest.json">', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Main background and font styling */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* 1. Clear, edge-to-edge camera feed (removes gray borders) */
    [data-testid="stCameraInput"] video {
        width: 100% !important;
        height: auto !important;
        object-fit: cover !important; 
        border-radius: 8px !important;
    }

    /* 2. Style the default overlay button to be green */
    [data-testid="stCameraInput"] button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        border: none !important;
    }

    [data-testid="stCameraInput"] button:hover {
        background-color: #45a049 !important;
    }

    /* Output dashboard styling */
    .prediction-box {
        background-color: #1E2127;
        border-left: 5px solid #4CAF50;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        text-align: center;
    }
    .weight-text {
        font-size: 42px;
        font-weight: 900;
        color: #4CAF50;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- System Constants ---
MARKER_REAL_SIZE_CM = 10.0  
TARGET_MARKER_ID = 0       
Z_REF = 100.0              
BASE_CM_PER_PIXEL_AT_1M = 0.045 

# --- Caching Models ---
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

@st.cache_resource
def load_regression_model():
    try:
        # Load the single best pipeline from my latest training
        rf_pipeline = joblib.load('Random_Forest_model.pkl')
        return rf_pipeline
    except Exception as e:
        st.error(f"⚠️ Could not load Random Forest model. Check the file name. Error: {e}")
        return None

# Load models into memory
model = load_yolo_model()
rf_model = load_regression_model()

# --- App Header ---
st.title("🐔 Computer Vision Broiler Weight predictor")
st.markdown("Place the 10cm ArUco marker next to the chicken for automatic scale calibration.")

# User Controls
z_actual = st.number_input(
    "Manual Camera Distance Fallback (cm):",
    min_value=10.0,
    max_value=300.0,
    value=100.0,
    help="Only used if the ArUco marker is hidden or missed."
)

st.markdown("### 📷 Live Camera Feed")
img_file_buffer = st.camera_input("Take Snapshot (Button inside feed)")

if img_file_buffer is not None:
    # 1. Load image from camera
    bytes_data = img_file_buffer.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # 2. Run YOLO Instance Segmentation 
    with st.spinner("Processing image and running segmentation..."):
        results = model(frame, conf=0.7)

    if results[0].masks is not None:
        # Extract Mask and Geometric Contours
        mask_coords = results[0].masks.xy[0]
        contour = mask_coords.astype(np.int32)

        # Raw pixel calculations
        raw_psa = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        (_, _), (w, h), _ = rect
        raw_l_max = max(w, h)
        raw_l_min = min(w, h)

        # 3. ArUco Marker Detection
        marker_detected = False
        cm_per_pixel = 0.0
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            corners, ids, _ = detector.detectMarkers(gray_frame)
        except AttributeError:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray_frame, aruco_dict, parameters=aruco_params
            )

        if ids is not None and TARGET_MARKER_ID in ids.flatten():
            marker_detected = True
            id_index = np.where(ids.flatten() == TARGET_MARKER_ID)[0][0]
            target_corners = corners[id_index][0]
            pixel_width = np.linalg.norm(target_corners[0] - target_corners[1])
            cm_per_pixel = MARKER_REAL_SIZE_CM / pixel_width
            calculation_method = "ArUco Optical Scale (High Accuracy)"
        else:
            cm_per_pixel = BASE_CM_PER_PIXEL_AT_1M * (z_actual / Z_REF)
            calculation_method = "Manual Height Fallback"

        # 4. Convert to Physical Metrics & Calculate New Proxies
        final_psa = raw_psa * (cm_per_pixel**2)
        final_l_max = raw_l_max * cm_per_pixel
        final_l_min = raw_l_min * cm_per_pixel
        volume_proxy = final_psa * final_l_max
        psa_pow_1_5 = final_psa ** 1.5

        # --- Visualizations ---
        annotated = frame.copy()
        cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 3)  
        if marker_detected:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

        # --- Dashboard Output ---
        st.markdown("---")
        st.subheader("Step 1: AI Segmentation Mask")
        
        # --- NEW SAFETY CHECK BLOCK ---
        if annotated is not None:
            try:
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Calibrated via: {calculation_method}",
                    use_column_width=True
                )
            except Exception as e:
                st.error(f"⚠️ Could not process image colors. The AI generated a false positive shape. Error: {e}")
        else:
            st.warning("⚠️ Could not generate the annotated image. Please try adjusting the camera angle.")
        # ------------------------------

        st.subheader("Step 2: Extracted Proxy Parameters")
        col1, col2, col3 = st.columns(3)
        col1.metric("Surface Area (PSA)", f"{final_psa:.1f} cm²")
        col2.metric("Max Length", f"{final_l_max:.1f} cm")
        col3.metric("Volume Proxy", f"{volume_proxy:.1f} cm³")

        # --- ML Prediction ---
        if rf_model:
            # Format exactly as the new pipeline expects using a DataFrame
            input_df = pd.DataFrame([{
                'projected_surface_area_cm2': final_psa,
                'min_axial_length_cm': final_l_min,
                'max_axial_length_cm': final_l_max,
                'volume_proxy_cm3': volume_proxy,
                'psa_pow_1_5': psa_pow_1_5
            }])
            
            pred_weight = rf_model.predict(input_df)[0]

            st.markdown("### Step 3: Final Weight Estimation")
            st.markdown(
                f"""
                <div class="prediction-box">
                    <p style="font-size:18px; margin-bottom:0; color:#AAAAAA;">Random Forest Regression Output</p>
                    <p class="weight-text">{pred_weight:.1f} grams</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("Model not loaded. Please check the backend.")

    else:
        st.error("⚠️ No object detected by the YOLO model. Adjust the camera angle and try again.")
