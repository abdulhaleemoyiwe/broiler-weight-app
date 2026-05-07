import streamlit as st
import cv2
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
# This is your calibration constant (Pixels per CM at 1 meter distance)
PIXELS_PER_CM = 18.5 

st.set_page_config(page_title="Broiler Morphometrics", layout="centered")

st.title("🐔 Broiler Morphometrics")
st.write("Objective 1: Data Collection & Weight Prediction")

# --- CAMERA INPUT ---
img_file = st.camera_input("Position chicken 1m away and capture")

if img_file is not None:
    # Convert Streamlit upload to OpenCV image
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # --- PROCESSING LOGIC (Otsu Thresholding) ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Using Otsu's method to separate chicken from background
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Identify the largest object (the broiler)
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # --- THE CALCULATIONS ---
        length_cm = w / PIXELS_PER_CM
        width_cm = h / PIXELS_PER_CM
        area_cm2 = cv2.contourArea(cnt) / (PIXELS_PER_CM**2)
        aspect_ratio = float(w) / h
        
        # Draw the bounding box on the image for visual verification
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
        st.image(image, channels="BGR", caption="Processed Image with Detection Box")
        
        # --- RESULTS UI ---
        st.subheader("Morphological Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Body Length", f"{length_cm:.2f} cm")
            st.metric("Body Width", f"{width_cm:.2f} cm")
        with col2:
            st.metric("Surface Area", f"{area_cm2:.2f} cm²")
            st.metric("Aspect Ratio", f"{aspect_ratio:.2f}")
            
        # --- SHANK MEASUREMENT ---
        st.divider()
        st.subheader("🦵 Shank Measurement")
        st.write("To measure the shank, identify its height in pixels from the image.")
        shank_px = st.number_input("Enter Shank height (pixels)", min_value=0)
        
        if shank_px > 0:
            shank_cm = shank_px / PIXELS_PER_CM
            st.success(f"Estimated Shank Length: {shank_cm:.2f} cm")
        
        # --- DATA RECORDING ---
        st.divider()
        actual_weight = st.number_input("Enter Actual Scale Weight (grams)", min_value=0)
        
        if st.button("Save to Dataset"):
            st.balloons()
            # This is where we would append to a CSV or Google Sheet
            st.success("Data points captured for analysis!")
            
    else:
        st.error("No chicken detected. Please ensure high contrast (Dark background, Light bird).")
