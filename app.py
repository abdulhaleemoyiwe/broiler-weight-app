import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_cropper import st_cropper
from PIL import Image

# --- CONFIGURATION ---
PIXELS_PER_CM = 18.5 

st.set_page_config(page_title="Broiler Morphometrics", layout="wide")

# --- CUSTOM CSS: INSTANT RESPONSIVE LAYOUT ---
st.markdown("""
    <style>
    /* 1. Only hide the central overlay reticle grid, leaving utility icons intact */
    [data-testid="stCameraInput"] video + div svg {
        display: none !important;
    }
    
    /* 2. Strip out all dark shading bars and blur filters from the viewport layout */
    [data-testid="stCameraInput"] > div,
    [data-testid="stCameraInput"] div div,
    [data-testid="stCameraInput"] [style*="backdrop-filter"] {
        background-color: transparent !important;
        background: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        box-shadow: none !important;
    }

    /* 3. Base styling for our 1-Meter Guide Box */
    [data-testid="stCameraInput"] { position: relative; }
    [data-testid="stCameraInput"]::after {
        content: "ALIGN CHICKEN IN THIS BOX (1 METER)";
        position: absolute;
        color: #00BFFF;
        font-weight: bold; font-size: 1.1rem;
        display: flex; align-items: flex-start; justify-content: center;
        padding-top: 10px; text-align: center;
        pointer-events: none; z-index: 99;
        border: 4px dashed #00BFFF;
    }

    /* 4. DYNAMIC ORIENTATION RULES */
    @media (orientation: portrait) {
        [data-testid="stCameraInput"]::after {
            top: 15%; left: 15%; right: 15%; bottom: 25%;
        }
    }
    
    @media (orientation: landscape) {
        [data-testid="stCameraInput"] video {
            width: 100% !important;
            height: auto !important;
            max-height: 70vh !important;
            object-fit: cover !important;
        }
        [data-testid="stCameraInput"]::after {
            top: 10%; left: 25%; right: 25%; bottom: 15%;
            font-size: 1.3rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.title("🐔 Automated Broiler Morphometrics")
st.write("1-Meter Calibrated Contour Profiling Layout")

# --- CAMERA INPUT ---
st.info("Step 1: Rotate your device sideways for Landscape View, then align the broiler.")
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
    
    st.write(f"Drag the red box tightly around the **{feature_type}**.")

    # FIX: Set realtime_update=True so changes are calculated instantly without button hangouts
    cropped_feature = st_cropper(
        raw_pil_image, 
        realtime_update=True, 
        box_color='#FF0000', 
        aspect_ratio=None
    )
    
    if cropped_feature:
        feature_cv2 = np.array(cropped_feature)
        feature_cv2 = cv2.cvtColor(feature_cv2, cv2.COLOR_RGB2BGR)
        
        # --- BACKGROUND PERFORMANCE BOOST ---
        # Scale down large images temporarily to keep calculations lightning fast on mobile
        h_orig, w_orig = feature_cv2.shape[:2]
        max_dimension = 300  # Lightweight pixel boundary
        scale_factor = 1.0
        
        if max(h_orig, w_orig) > max_dimension:
            scale_factor = max_dimension / max(h_orig, w_orig)
            processing_img = cv2.resize(feature_cv2, (int(w_orig * scale_factor), int(h_orig * scale_factor)))
        else:
            processing_img = feature_cv2.copy()
            
        adjusted_pixels_per_cm = PIXELS_PER_CM * scale_factor
        
        # --- INSTANT LOW-LIGHT NOISE FILTERING & MASK SEGMENTATION ---
        gray = cv2.cvtColor(processing_img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 7, 50, 50)
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # --- PARAMETER 1: INSTANT SURFACE AREA ---
        total_mask_pixels = np.sum(clean_mask == 255)
        measured_area_cm2 = total_mask_pixels / (adjusted_pixels_per_cm ** 2)
        
        # --- PARAMETER 2: CONTOUR BOUNDING BOX ---
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x_scaled, y_scaled, w_scaled, h_scaled = cv2.boundingRect(largest_contour)
            
            # Map back up to original coordinates for crisp high-res display rendering
            x = int(x_scaled / scale_factor)
            y = int(y_scaled / scale_factor)
            w = int(w_scaled / scale_factor)
            h = int(h_scaled / scale_factor)
            
            if feature_type == "Shank Length":
                chosen_pixel_dimension = h
                pt1, pt2 = (int(x + w//2), int(y)), (int(x + w//2), int(y + h))
            else:
                chosen_pixel_dimension = w
                pt1, pt2 = (int(x), int(y + h//2)), (int(x + w), int(y + h//2))
            
            automated_length_cm = chosen_pixel_dimension / PIXELS_PER_CM
            
            # Draw tracking output cleanly
            visual_output = feature_cv2.copy()
            cv2.rectangle(visual_output, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            cv2.line(visual_output, pt1, pt2, (255, 0, 0), 3) 
            
            st.write("### Processing & Segmentation Results")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                display_img = cv2.cvtColor(visual_output, cv2.COLOR_BGR2RGB)
                st.image(display_img, caption=f"Measured {feature_type} Window")
            with col_img2:
                display_mask = cv2.resize(clean_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                st.image(display_mask, caption="Segmented Profile Mask (Used for Area)")
            
            # Instant layout metrics
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
            st.warning("No clear broiler profile detected inside the selection region. Readjust the red crop box.")
