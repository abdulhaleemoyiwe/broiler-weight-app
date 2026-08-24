import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# Custom CSS to remove gray letterboxing borders from camera input
st.markdown(
    """
    <style>
    /* Make camera container full width and remove background padding */
    [data-testid="stCameraInput"] > div {
        background-color: transparent !important;
        width: 100% !important;
    }
    /* Stretch video feed to eliminate side pillarboxes */
    [data-testid="stCameraInput"] video {
        object-fit: cover !important;
        width: 100% !important;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- System Constants ---
CSV_FILE = "regression_data.csv"
MARKER_REAL_SIZE_CM = 10.0  # Updated: 10cm x 10cm ArUco Marker
TARGET_MARKER_ID = 0       # Updated: Specifically target ID 0
Z_REF = 100.0              # Baseline calibration height in cm (1 meter)

# Baseline calibration pixel ratio
BASE_CM_PER_PIXEL_AT_1M = 0.045 


@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()

st.title("Chicken Feature Extraction (ArUco & Fallback)")

# User Controls
actual_weight = st.number_input(
    "Enter Actual Scale Weight (g):", min_value=0.0, step=1.0
)
z_actual = st.number_input(
    "Camera Distance to Platform (cm):",
    min_value=10.0,
    max_value=300.0,
    value=100.0,
)

img_file_buffer = st.camera_input("Take Snapshot")

if img_file_buffer is not None:
    # 1. Load image from camera
    bytes_data = img_file_buffer.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # 2. Run YOLO Instance Segmentation (Only detect if 70% confident or higher)
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

        # 3. ArUco Marker Detection (DICT_4X4_50, Target ID 0)
        marker_detected = False
        cm_per_pixel = 0.0
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            # For newer OpenCV versions (4.7+)
            aruco_dict = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_4X4_50
            )
            aruco_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            corners, ids, _ = detector.detectMarkers(gray_frame)
        except AttributeError:
            # For older OpenCV versions
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray_frame, aruco_dict, parameters=aruco_params
            )

        # Check if any markers were found AND if Target ID 0 is among them
        if ids is not None and TARGET_MARKER_ID in ids.flatten():
            marker_detected = True
            st.success(
                f"✅ ArUco Marker (ID {TARGET_MARKER_ID}) detected! Using 10cm optical scale calibration."
            )

            # Locate index for ID 0
            id_index = np.where(ids.flatten() == TARGET_MARKER_ID)[0][0]
            target_corners = corners[id_index][0]

            # Calculate pixel width of the 10cm marker top edge (top-left to top-right)
            top_left = target_corners[0]
            top_right = target_corners[1]
            pixel_width = np.linalg.norm(top_left - top_right)

            # Establish dynamic real-world scale (10cm / width in pixels)
            cm_per_pixel = MARKER_REAL_SIZE_CM / pixel_width
            calculation_method = "ArUco Marker (10cm)"
        else:
            # FALLBACK METHOD: Manual Height
            st.warning(
                f"⚠️ Marker ID 0 hidden or missed. Falling back to manual height ({z_actual} cm)."
            )

            # Scale baseline ratio by current camera height
            cm_per_pixel = BASE_CM_PER_PIXEL_AT_1M * (z_actual / Z_REF)
            calculation_method = "Manual Fallback"

        # 4. Convert all pixel measurements to exact centimeters (cm & cm²)
        final_psa = raw_psa * (cm_per_pixel**2)
        final_l_max = raw_l_max * cm_per_pixel
        final_l_min = raw_l_min * cm_per_pixel

        # --- Visualizations ---
        annotated = frame.copy()
        cv2.drawContours(
            annotated, [contour], -1, (0, 255, 0), 2
        )  # Draw object mask

        if marker_detected:
            # Draw green square around detected ArUco marker ID 0
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

        # Display Segmented Image
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption=f"Segmentation Mode: {calculation_method}",
        )

        # --- Download Button for Segmented Image ---
        _, img_buffer = cv2.imencode(".jpg", annotated)
        st.download_button(
            label="📸 Download Segmented Image",
            data=img_buffer.tobytes(),
            file_name="segmented_capture.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

        # --- Display Final Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("PSA (cm²)", f"{final_psa:.1f}")
        col2.metric("L_max (cm)", f"{final_l_max:.1f}")
        col3.metric("L_min (cm)", f"{final_l_min:.1f}")

        # --- Save to CSV ---
        if st.button("Save Data for Regression", use_container_width=True):
            new_data = {
                "PSA_cm2": final_psa,
                "L_max_cm": final_l_max,
                "L_min_cm": final_l_min,
                "Distance_cm": z_actual,
                "Method": calculation_method,
                "Weight_g": actual_weight,
            }
            df = pd.DataFrame([new_data])

            if not os.path.isfile(CSV_FILE):
                df.to_csv(CSV_FILE, index=False)
            else:
                df.to_csv(CSV_FILE, mode="a", header=False, index=False)

            st.success(
                f"Data Logged Successfully! Total records: {len(pd.read_csv(CSV_FILE))}"
            )

        # --- Preview & Download Full CSV Dataset ---
        if os.path.exists(CSV_FILE):
            st.markdown("---")
            st.subheader("📊 Logged Regression Dataset")
            df_all = pd.read_csv(CSV_FILE)

            # Display dataset table on screen
            st.dataframe(df_all, use_container_width=True)

            # Download button for CSV data
            csv_bytes = df_all.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Dataset (CSV)",
                data=csv_bytes,
                file_name="regression_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        st.error(
            "No object detected by YOLO model. Adjust placement and retry."
        )
