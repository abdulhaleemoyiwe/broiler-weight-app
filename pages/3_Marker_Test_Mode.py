import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_cropper import st_cropper
from PIL import Image
import io

st.set_page_config(page_title="Marker Calibration Test Mode", layout="centered")

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
        content: "PLACE 10x10cm MARKER & TEST OBJECT INSIDE FRAME";
        position: absolute;
        top: 15%; left: 10%; right: 10%; bottom: 20%;
        border: 4px dashed #FF4B4B;
        color: #FF4B4B;
        font-weight: bold; font-size: 1.1rem;
        display: flex; align-items: flex-start; justify-content: center;
        padding-top: 12px; text-align: center;
        pointer-events: none; 
        z-index: 99999 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧪 Marker Calibration Test Mode")
st.write("Test the 10x10cm reference marker accuracy on random test objects before deploying to live chickens.")

# --- STEP 1: CAMERA INPUT ---
st.info("Step 1: Place your 10x10cm white square marker flat on the ground right next to your test object.")
img_file = st.camera_input("Capture Test Frame")

if img_file is not None:
    raw_pil_image = Image.open(img_file)
    st.divider()
    
    # --- STEP 2: INTERACTIVE CROP ---
    st.subheader("Step 2: Box BOTH the Marker and the Test Object")
    st.write("Drag the red box tightly so it encloses **both the 10x10 square and the object** you want to measure.")

    cropped_feature = st_cropper(
        raw_pil_image, 
        realtime_update=True, 
        box_color='#FF0000', 
        aspect_ratio=None
    )
    
    if cropped_feature:
        feature_cv2 = np.array(cropped_feature)
        feature_cv2 = cv2.cvtColor(feature_cv2, cv2.COLOR_RGB2BGR)
        
        # --- IMAGE PROCESSING PIPELINE ---
        h_orig, w_orig = feature_cv2.shape[:2]
        gray = cv2.cvtColor(feature_cv2, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Otsu Thresholding
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological Opening (Removes floor dust / background noise)
        kernel_open = np.ones((5,5), np.uint8)
        opened_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        
        # Morphological Closing (Fills internal gaps)
        kernel_close = np.ones((5,5), np.uint8)
        clean_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # --- FIND CONTOURS ---
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        marker_contour = None
        pixels_per_cm = None
        
        # --- MARKER DETECTION ALGORITHM ---
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300: # Filter out tiny noise specks
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            # Check for 4 corners and square-like proportions
            if len(approx) == 4 and 0.80 <= aspect_ratio <= 1.20:
                marker_contour = cnt
                pixels_per_cm = w / 10.0 # 10 cm baseline scale reference
                break

        visual_output = feature_cv2.copy()
        
        if pixels_per_cm is not None:
            st.success("✅ 10x10 Marker Detected Successfully! Dynamic calibration active.")
            cv2.drawContours(visual_output, [marker_contour], -1, (255, 0, 0), 3) # Blue boundary for marker
            scale_status = "Dynamic (Marker Calibrated)"
            
            # --- FIXED OBJECT MEASUREMENT LOGIC ---
            valid_object_contour = None
            max_obj_area = 0
            
            # Get bounding box of the detected marker to exclude it cleanly
            mx, my, mw, mh = cv2.boundingRect(marker_contour)
            
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                
                # Check if this contour IS the marker by seeing if their bounding boxes overlap heavily
                if abs(cx - mx) < 15 and abs(cy - my) < 15 and abs(cw - mw) < 15:
                    continue
                
                area = cv2.contourArea(cnt)
                if area > max_obj_area and area > 200:
                    max_obj_area = area
                    valid_object_contour = cnt
            
            if valid_object_contour is not None:
                hull = cv2.convexHull(valid_object_contour)
                rect = cv2.minAreaRect(hull)
                box_pts = cv2.boxPoints(rect)
                box_pts = np.int32(box_pts)
                
                rect_width, rect_height = rect[1]
                chosen_pixel_dimension = max(rect_width, rect_height)
                
                automated_length_cm = chosen_pixel_dimension / pixels_per_cm
                measured_area_cm2 = cv2.contourArea(hull) / (pixels_per_cm ** 2)
                
                cv2.drawContours(visual_output, [box_pts], 0, (0, 255, 0), 2)  
                cv2.drawContours(visual_output, [hull], 0, (0, 255, 255), 1) 
                
                st.write("### Test Analysis Results")
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    display_img = cv2.cvtColor(visual_output, cv2.COLOR_BGR2RGB)
                    st.image(display_img, caption="Analyzed Object View")
                with col_img2:
                    display_mask = cv2.resize(clean_mask, (w_orig, h_orig))
                    st.image(display_mask, caption="Clean Filter Mask")
                
                c1, c2 = st.columns(2)
                c1.metric(label="Calculated Object Length", value=f"{automated_length_cm:.2f} cm")
                c2.metric(label="Calculated Surface Area", value=f"{measured_area_cm2:.2f} cm²")
                
                # --- STEP 3: DATA LOGGER ---
                st.divider()
                st.subheader("Step 3: Log Test Data")
                actual_length = st.number_input("Actual Physical Length of Object (measured with ruler in cm):", min_value=0.0, step=0.1)
                
                current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                data_to_save = pd.DataFrame([{
                    "Date/Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Feature Measured": "Test Object",
                    "Camera Length (cm)": round(automated_length_cm, 2),
                    "Surface Area (cm2)": round(measured_area_cm2, 2),
                    "Actual Physical Length (cm)": actual_length,
                    "Calibration Method": scale_status
                }])
                
                csv = data_to_save.to_csv(index=False).encode('utf-8')
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.download_button(
                        label="💾 Download Test CSV Entry",
                        data=csv,
                        file_name=f"test_marker_metric_{current_timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary"
                    )
                with col_btn2:
                    raw_image_buffer = io.BytesIO()
                    raw_pil_image.save(raw_image_buffer, format="JPEG")
                    raw_image_bytes = raw_image_buffer.getvalue()
                    
                    st.download_button(
                        label="📸 Download Test Image",
                        data=raw_image_bytes,
                        file_name=f"test_marker_image_{current_timestamp}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
            else:
                st.warning("⚠️ 10x10 Marker found, but no target object was detected inside the crop box. Expand your red crop box slightly to fully enclose your test object.")
        else:
            st.error("❌ 10x10 Marker not found. Ensure the white square paper is flat, well-lit, and fully enclosed within your red crop box.")
