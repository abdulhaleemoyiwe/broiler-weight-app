import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_cropper import st_cropper
from PIL import Image

# --- CONFIGURATION ---
# Calibration constant: Pixels per CM at exactly 1 meter distance
PIXELS_PER_CM = 18.5 

st.set_page_config(page_title="Broiler Morphometrics", layout="centered")

# --- CUSTOM CSS: LIVE BLUE OVERLAY ON CAMERA ---
st.markdown("""
    <style>
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

st.title("🐔 Broiler Morphometrics")
st.write("1-Meter Calibrated Data Collection")

# --- CAMERA INPUT WITH LIVE BLUE BOX ---
st.info("Step 1: Use the live blue box below to align the chicken at exactly 1 meter distance.")
img_file = st.camera_input("Capture Broiler Image")

if img_file is not None:
    raw_pil_image = Image.open(img_file)
    st.divider()
    
    # --- POST-CAPTURE FEATURE SELECTION ---
    st.subheader("Step 2: Isolate Measurement Target")
    feature_type = st.selectbox(
        "Select the anatomical feature to measure:",
        ["Chicken Body", "Shank (Leg)"]
    )
    
    st.write(f"Drag the red box to tightly enclose the **{feature_type}** to filter out background noise.")

    # INTERACTIVE BOUNDING BOX
    cropped_feature = st_cropper(raw_pil_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    
    if cropped_feature:
        feature_cv2 = np.array(cropped_feature)
        feature_cv2 = cv2.cvtColor(feature_cv2, cv2.COLOR_RGB2BGR)
        
        # --- LOW-LIGHT NOISE FILTERING & EDGE DETECTION ---
        gray = cv2.cvtColor(feature_cv2, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5,5), np.uint8)
        clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x_f, y_f, w_f, h_f = cv2.boundingRect(cnt)
            
            # Draw the final green measurement box
            cv2.rectangle(feature_cv2, (x_f, y_f), (x_f + w_f, y_f + h_f), (0, 255, 0), 3)
            
            # --- AUTOMATIC CALCULATIONS IN CM ---
            measured_length_cm = w_f / PIXELS_PER_CM
            measured_width_cm = h_f / PIXELS_PER_CM
            measured_area_cm2 = cv2.contourArea(cnt) / (PIXELS_PER_CM**2)
            
            st.write("### Processing Results")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(feature_cv2, channels="BGR", caption=f"Measured {feature_type}")
            with col_img2:
                st.image(clean_mask, caption="Low-Light Filter Mask")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Length", f"{measured_length_cm:.2f} cm")
            c2.metric("Width", f"{measured_width_cm:.2f} cm")
            c3.metric("Surface Area", f"{measured_area_cm2:.2f} cm²")
            
            # --- ELECTRONIC SCALE INPUT & DATA EXPORT ---
            st.divider()
            st.subheader("Step 3: Log Data")
            actual_weight = st.number_input("Actual Scale Weight (grams) for Database", min_value=0.0, step=0.1)
            
            # Create the spreadsheet row
            data_to_save = pd.DataFrame([{
                "Date/Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Feature Measured": feature_type,
                "Length (cm)": round(measured_length_cm, 2),
                "Width (cm)": round(measured_width_cm, 2),
                "Area (cm2)": round(measured_area_cm2, 2),
                "Actual Weight (g)": actual_weight
            }])
            
            # Generate CSV file for download
            csv = data_to_save.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="💾 Download Measurement Data (CSV)",
                data=csv,
                file_name=f"broiler_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary"
            )
            
            st.caption("Clicking the button above will save this measurement directly to your phone as a spreadsheet file.")
                
        else:
            st.warning("Could not detect the object. Please adjust the red crop box.")
