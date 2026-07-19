import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_cropper import st_cropper
from PIL import Image
import io

# --- FALLBACK CALIBRATION (If marker is not found) ---
FALLBACK_PIXELS_PER_CM = 18.5 

st.set_page_config(page_title="Broiler Morphometrics V2", layout="centered")

# --- CUSTOM CSS: CLEAN CAMERA VIEWPORT ---
st.markdown("""
    <style>
    [data-testid="stCameraInput"] video + div svg { display: none !important; }
    [data-testid="stCameraInput"], [data-testid="stCameraInput"] > div,
    [data-testid="stCameraInput"] div div {
        background-color: transparent !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stCameraInput"] video {
        width: 100% !important;
        height: auto !important;
        object-fit: contain !important;
    }
    [data-testid="stCameraInput"] { position: relative; }
    [data-testid="stCameraInput"]::after {
        content: "PLACE 10x10cm WHITE SQUARE NEXT TO BIRD";
        position: absolute;
        top: 15%; left: 10%; right: 10%; bottom: 20%;
        border: 4px dashed #00BFFF;
        color: #00BFFF;
        font-weight: bold; font-size: 1.1rem;
        display: flex; align-items: flex-start; justify-content: center;
        padding-top: 12px; text-align: center;
        pointer-events: none; 
        z-index: 99999 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🐔 Broiler Morphometrics (Marker Edition)")
st.write("Auto-Calibrating using 10x10cm Reference Marker & Morphological Filtering")

# --- STEP 1: CAMERA INPUT ---
st.info("Step 1: Place the 10x10cm white paper flat on the ground next to the bird.")
img_file = st.camera_input("Capture Broiler Image")

if img_file is not None:
    raw_pil_image = Image.open(img_file)
    st.divider()
    
    # --- STEP 2: INTERACTIVE CROP (MUST INCLUDE MARKER AND BIRD) ---
    st.subheader("Step 2: Box BOTH the Bird and the Marker")
    feature_type = st.selectbox(
        "Select the feature to measure:",
        ["Body Length", "Wingspan", "Shank Length"]
    )
    
    st.write("Drag the red box tightly so it contains **both the chicken and the 10x10 square**.")

    cropped_feature = st_cropper(
        raw_pil_image, 
        realtime_update=True, 
        box_color='#FF0000', 
        aspect_ratio=None
    )
    
    if cropped_feature:
        feature_cv2 = np.array(cropped_feature)
        feature_cv2 = cv2.cvtColor(feature_cv2, cv2.COLOR_RGB2BGR)
        
        # --- IMAGE PROCESSING ---
        h_orig, w_orig = feature_cv2.shape[:2]
        gray = cv2.cvtColor(feature_cv2, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Otsu Segmentation
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # NEW: Morphological Opening Filter (Removes small floor noise/dust)
        kernel_open = np.ones((5,5), np.uint8)
        opened_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        
        # Close filter (fills holes in the bird)
        kernel_close = np.ones((5,5), np.uint8)
        clean_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # --- FIND CONTOURS ---
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        marker_contour = None
        chicken_contour = None
        pixels_per_cm = None
        max_area = 0
        
        # --- MARKER DETECTION LOGIC ---
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500: # Ignore tiny specks
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            # If it has 4 corners and is roughly square-shaped
            if len(approx) == 4 and 0.80 <= aspect_ratio <= 1.20:
                marker_contour = cnt
                pixels_per_cm = w / 10.0 # 10 cm square
            else:
                if area > max_area:
                    max_area = area
                    chicken_contour = cnt

        visual_output = feature_cv2.copy()
        
        # Check if calibration worked
        if pixels_per_cm is not None:
            st.success("✅ 10x10 Marker Detected! Dynamic scaling active.")
            cv2.drawContours(visual_output, [marker_contour], -1, (255, 0, 0), 3) # Blue box for marker
            used_pixels_per_cm = pixels_per_cm
            scale_status = "Dynamic (Marker)"
        else:
            st.warning("⚠️ Marker not found. Using fallback 1-meter fixed distance calibration.")
            used_pixels_per_cm = FALLBACK_PIXELS_PER_CM
            scale_status = "Fixed (Fallback)"
            chicken_contour = max(contours, key=cv2.contourArea) if contours else None

        # --- CHICKEN MEASUREMENT LOGIC ---
        if chicken_contour is not None:
            hull = cv2.convexHull(chicken_contour)
            rect = cv2.minAreaRect(hull)
            box_pts = cv2.boxPoints(rect)
            box_pts = np.int32(box_pts)
            
            rect_width, rect_height = rect[1]
            chosen_pixel_dimension = max(rect_width, rect_height)
            
            automated_length_cm = chosen_pixel_dimension / used_pixels_per_cm
            measured_area_cm2 = cv2.contourArea(hull) / (used_pixels_per_cm ** 2)
            
            cv2.drawContours(visual_output, [box_pts], 0, (0, 255, 0), 2)  
            cv2.drawContours(visual_output, [hull], 0, (0, 255, 255), 1) 
            
            st.write("### Processing & Segmentation Results")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                display_img = cv2.cvtColor(visual_output, cv2.COLOR_BGR2RGB)
                st.image(display_img, caption=f"Analyzed View ({scale_status})")
            with col_img2:
                display_mask = cv2.resize(clean_mask, (w_orig, h_orig))
                st.image(display_mask, caption="Cleaned Filter Mask (Morph Open Applied)")
            
            c1, c2 = st.columns(2)
            c1.metric(label="True Rotated Axis Length", value=f"{automated_length_cm:.2f} cm")
            c2.metric(label="Convex Hull Surface Area", value=f"{measured_area_cm2:.2f} cm²")
            
            # --- STEP 3: DATABASE DATA LOGGER ---
            st.divider()
            st.subheader("Step 3: Log Data")
            actual_weight = st.number_input("Actual Scale Weight (grams) for Database", min_value=0.0, step=0.1)
            
            current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            data_to_save = pd.DataFrame([{
                "Date/Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Feature Measured": feature_type,
                "Length (cm)": round(automated_length_cm, 2),
                "Surface Area (cm2)": round(measured_area_cm2, 2),
                "Actual Weight (g)": actual_weight,
                "Calibration Method": scale_status
            }])
            
            csv = data_to_save.to_csv(index=False).encode('utf-8')
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.download_button(
                    label="💾 Download CSV Entry",
                    data=csv,
                    file_name=f"broiler_metrics_v2_{current_timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )
            with col_btn2:
                raw_image_buffer = io.BytesIO()
                raw_pil_image.save(raw_image_buffer, format="JPEG")
                raw_image_bytes = raw_image_buffer.getvalue()
                
                st.download_button(
                    label="📸 Download Clean Image",
                    data=raw_image_bytes,
                    file_name=f"broiler_raw_v2_{current_timestamp}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
        else:
            st.warning("No clear broiler silhouette detected inside the selection region. Readjust the red crop box.")
