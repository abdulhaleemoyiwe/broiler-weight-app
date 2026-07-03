import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_cropper import st_cropper
from PIL import Image

# --- CONFIGURATION ---
PIXELS_PER_CM = 18.5 

st.set_page_config(page_title="Broiler Morphometrics", layout="centered")

# --- CUSTOM CLEAN WEB LAYOUT CSS ---
st.markdown("""
    <style>
    [data-testid="stCameraInput"] video + div svg {
        display: none !important;
    }
    [data-testid="stCameraInput"] > div,
    [data-testid="stCameraInput"] div div,
    [data-testid="stCameraInput"] [style*="backdrop-filter"] {
        background-color: transparent !important;
        background: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        box-shadow: none !important;
    }
    [data-testid="stCameraInput"] { position: relative; }
    [data-testid="stCameraInput"]::after {
        content: "ALIGN CHICKEN IN THIS BOX (1 METER)";
        position: absolute;
        top: 15%; left: 15%; right: 15%; bottom: 25%;
        border: 4px dashed #00BFFF;
        color: #00BFFF;
        font-weight: bold; font-size: 1.1rem;
        display: flex; align-items: flex-start; justify-content: center;
        padding-top: 10px; text-align: center;
        pointer-events: none; z-index: 99;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🐔 Automated Broiler Morphometrics")
st.write("1-Meter Calibrated Contour Profiling Layout")

# --- CAMERA INPUT ---
st.info("Step 1: Align the chicken inside the clear blue dashed box at exactly 1 meter distance.")
img_file = st.camera_input("Capture Broiler Image")

if img_file is not None:
    raw_pil_image = Image.open(img_file)
    st.divider()
    
    # --- POST-CAPTURE FEATURE SELECTION ---
    st.subheader("Step 2: Select Morphometric Target")
    feature_type = st.selectbox(
        "Select the feature to measure:",
        ["Body Length", "Wingspan", "Shank Length"]
    )
    
    st.write(f"Drag the red box tightly around the **{feature_type}**, then click the button below.")

    # NO LAG: realtime_update=False lets you move the box smoothly
    cropped_feature = st_cropper(
        raw_pil_image, 
        realtime_update=False, 
        box_color='#FF0000', 
        aspect_ratio=None
    )
    
    # Clean activation trigger button to stop backend lagging
    if st.button("Apply Crop & Run Measurement", type="secondary"):
        if cropped_feature:
            feature_cv2 = np.array(cropped_feature)
            feature_cv2 = cv2.cvtColor(feature_cv2, cv2.COLOR_RGB2BGR)
            
            # --- LOW-LIGHT NOISE FILTERING & MASK SEGMENTATION ---
            gray = cv2.cvtColor(feature_cv2, cv2.COLOR_BGR2GRAY)
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((5,5), np.uint8)
            clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # --- PARAMETER 1: TRUE SURFACE AREA ---
            total_mask_pixels = np.sum(clean_mask == 255)
            measured_area_cm2 = total_mask_pixels / (PIXELS_PER_CM ** 2)
            
            # --- PARAMETER 2: CONTOUR BOUNDING BOX (TOP-DOWN / SIDE-TO-SIDE LENGTH) ---
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Isolate the largest white object in the crop box
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Dynamic feature logic: Wingspan/Body are horizontal width, Shank is vertical height
                if feature_type == "Shank Length":
                    chosen_pixel_dimension = h
                    # Draw a straight vertical tracking line down the center of the leg
                    pt1, pt2 = (int(x + w//2), int(y)), (int(x + w//2), int(y + h))
                else:
                    chosen_pixel_dimension = w
                    # Draw a straight horizontal line across the body or wing span
                    pt1, pt2 = (int(x), int(y + h//2)), (int(x + w), int(y + h//2))
                
                automated_length_cm = chosen_pixel_dimension / PIXELS_PER_CM
                
                # Visual rendering layout
                visual_output = feature_cv2.copy()
                cv2.rectangle(visual_output, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green bounding box
                cv2.line(visual_output, pt1, pt2, (255, 0, 0), 3) # Blue measurement axis
                
                st.write("### Processing & Segmentation Results")
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    display_img = cv2.cvtColor(visual_output, cv2.COLOR_BGR2RGB)
                    st.image(display_img, caption=f"Measured {feature_type} Window")
                with col_img2:
                    st.image(clean_mask, caption="Segmented Profile Mask (Used for Area)")
                
                # Clean metrics readout
                c1, c2 = st.columns(2)
                c1.metric(label="Calculated Profile Length", value=f"{automated_length_cm:.2f} cm")
                c2.metric(label="True Surface Area", value=f"{measured_area_cm2:.2f} cm²")
                
                # --- DATA LOGGER AND EXPORT ---
                st.divider()
                st.subheader("Step 3: Log Data")
                actual_weight = st.number_input("Actual Scale Weight (grams) for Database", min_value=0.0, step=0.1)
                
                data_to_save = pd.DataFrame([{
                    "Date/Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Feature Measured": feature_type,
                    "Length (cm)": round(automated_length_cm, 2),
                    "Surface Area (cm2)": round(measured_area_cm2, 2),
                    "Actual Weight (g)": actual_weight
                }])
                
                csv = data_to_save.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="💾 Download Morphometrics Entry (CSV)",
                    data=csv,
                    file_name=f"broiler_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    type="primary"
                )
            else:
                st.warning("Could not separate chicken shape from background. Ensure the target is clearly visible inside the box.")
