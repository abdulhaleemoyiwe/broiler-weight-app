import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_cropper import st_cropper
from PIL import Image
import io

# --- CONFIGURATION & CALIBRATION ---
PIXELS_PER_CM = 18.5 

st.set_page_config(page_title="Broiler Morphometrics", layout="centered")

# --- CUSTOM CSS: CLEAN CAMERA VIEWPORT & 1-METER GUIDANCE BOX ---
st.markdown("""
    <style>
    /* 1. Hide default Streamlit camera target reticle lines */
    [data-testid="stCameraInput"] video + div svg {
        display: none !important;
    }
    
    /* 2. Strip out all dark masking bars, shading, and blur filters */
    [data-testid="stCameraInput"], 
    [data-testid="stCameraInput"] > div,
    [data-testid="stCameraInput"] div div,
    [data-testid="stCameraInput"] [style*="backdrop-filter"],
    [data-testid="stCameraInput"] [style*="background-color"] {
        background-color: transparent !important;
        background: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        box-shadow: none !important;
    }

    /* 3. Ensure crisp video presentation without side letterboxing */
    [data-testid="stCameraInput"] video {
        width: 100% !important;
        height: auto !important;
        object-fit: contain !important;
        background-color: transparent !important;
    }

    /* 4. Overlay bright blue 1-Meter Calibration Guide Box on top layer */
    [data-testid="stCameraInput"] { position: relative; }
    [data-testid="stCameraInput"]::after {
        content: "ALIGN CHICKEN IN THIS BOX (1 METER)";
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

st.title("🐔 Automated Broiler Morphometrics")
st.write("1-Meter Calibrated Rotated Contour & Convex Hull Profiling")

# --- STEP 1: CAMERA INPUT ---
st.info("Step 1: Align the chicken inside the clear blue dashed box at exactly 1 meter distance.")
img_file = st.camera_input("Capture Broiler Image")

if img_file is not None:
    raw_pil_image = Image.open(img_file)
    st.divider()
    
    # --- STEP 2: FEATURE SELECTION & INTERACTIVE CROP ---
    st.subheader("Step 2: Select Morphometric Target")
    feature_type = st.selectbox(
        "Select the feature to measure:",
        ["Body Length", "Wingspan", "Shank Length"]
    )
    
    st.write(f"Drag the red box tightly around the **{feature_type}**.")

    cropped_feature = st_cropper(
        raw_pil_image, 
        realtime_update=True, 
        box_color='#FF0000', 
        aspect_ratio=None
    )
    
    if cropped_feature:
        feature_cv2 = np.array(cropped_feature)
        feature_cv2 = cv2.cvtColor(feature_cv2, cv2.COLOR_RGB2BGR)
        
        # --- PERFORMANCE MATRIX DOWNSAMPLING ---
        h_orig, w_orig = feature_cv2.shape[:2]
        max_dimension = 350  
        scale_factor = 1.0
        
        if max(h_orig, w_orig) > max_dimension:
            scale_factor = max_dimension / max(h_orig, w_orig)
            processing_img = cv2.resize(feature_cv2, (int(w_orig * scale_factor), int(h_orig * scale_factor)))
        else:
            processing_img = feature_cv2.copy()
            
        adjusted_pixels_per_cm = PIXELS_PER_CM * scale_factor
        
        # --- LOW-LIGHT NOISE FILTERING & OTSU SEGMENTATION ---
        gray = cv2.cvtColor(processing_img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((5,5), np.uint8)
        clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # --- BIOLOGICAL ROTATED AXIS & CONVEX HULL EXTRACTION ---
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            hull_scaled = cv2.convexHull(largest_contour)
            rect_scaled = cv2.minAreaRect(hull_scaled)
            box_pts_scaled = cv2.boxPoints(rect_scaled)
            
            rect_width, rect_height = rect_scaled[1]
            chosen_pixel_dimension = max(rect_width, rect_height)
            
            automated_length_cm = chosen_pixel_dimension / adjusted_pixels_per_cm
            measured_area_cm2 = cv2.contourArea(hull_scaled) / (adjusted_pixels_per_cm ** 2)
            
            dist01 = np.linalg.norm(box_pts_scaled[0] - box_pts_scaled[1])
            dist12 = np.linalg.norm(box_pts_scaled[1] - box_pts_scaled[2])
            
            if dist01 > dist12:
                mid1 = (box_pts_scaled[0] + box_pts_scaled[3]) / 2.0
                mid2 = (box_pts_scaled[1] + box_pts_scaled[2]) / 2.0
            else:
                mid1 = (box_pts_scaled[0] + box_pts_scaled[1]) / 2.0
                mid2 = (box_pts_scaled[2] + box_pts_scaled[3]) / 2.0
                
            hull_display = (hull_scaled / scale_factor).astype(np.int32)
            box_display = (box_pts_scaled / scale_factor).astype(np.int32)
            pt1 = (int(mid1[0] / scale_factor), int(mid1[1] / scale_factor))
            pt2 = (int(mid2[0] / scale_factor), int(mid2[1] / scale_factor))
            
            visual_output = feature_cv2.copy()
            cv2.drawContours(visual_output, [box_display], 0, (0, 255, 0), 2)  
            cv2.drawContours(visual_output, [hull_display], 0, (0, 255, 255), 1) 
            cv2.line(visual_output, pt1, pt2, (255, 0, 0), 3)                 
            
            st.write("### Processing & Segmentation Results")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                display_img = cv2.cvtColor(visual_output, cv2.COLOR_BGR2RGB)
                st.image(display_img, caption=f"Rotated Biological Axis ({feature_type})")
            with col_img2:
                display_mask = cv2.resize(clean_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                st.image(display_mask, caption="Convex Hull Filter Mask (Used for Area)")
            
            c1, c2 = st.columns(2)
            c1.metric(label="True Rotated Axis Length", value=f"{automated_length_cm:.2f} cm")
            c2.metric(label="Convex Hull Surface Area", value=f"{measured_area_cm2:.2f} cm²")
            
            # --- STEP 3: DATABASE DATA LOGGER ---
            st.divider()
            st.subheader("Step 3: Log Data")
            actual_weight = st.number_input("Actual Scale Weight (grams) for Database", min_value=0.0, step=0.1)
            
            # Formulate perfectly paired layout names
            current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            data_to_save = pd.DataFrame([{
                "Date/Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Feature Measured": feature_type,
                "Length (cm)": round(automated_length_cm, 2),
                "Surface Area (cm2)": round(measured_area_cm2, 2),
                "Actual Weight (g)": actual_weight
            }])
            
            csv = data_to_save.to_csv(index=False).encode('utf-8')
            
            # Layout the data download elements nicely side-by-side
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.download_button(
                    label="💾 Download CSV Entry",
                    data=csv,
                    file_name=f"broiler_metrics_{current_timestamp}.csv",
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
                    file_name=f"broiler_raw_{current_timestamp}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
        else:
            st.warning("No clear broiler silhouette detected inside the selection region. Readjust the red crop box.")
