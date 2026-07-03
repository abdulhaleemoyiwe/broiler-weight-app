import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_cropper import st_cropper
from PIL import Image
from skimage.morphology import skeletonize

# --- CONFIGURATION ---
PIXELS_PER_CM = 18.5 

st.set_page_config(page_title="Broiler Morphometrics", layout="centered")

# --- CUSTOM CLEAN WEB LAYOUT CSS ---
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

    /* 3. Render the clear bright blue 1-Meter Guide Box perfectly on top */
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
st.write("1-Meter Calibrated Length & Surface Area Segmentation")

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
    
    st.write(f"Drag the red box to tightly enclose the **{feature_type}**.")

    # INTERACTIVE BOUNDING BOX (Returns cropped PIL image)
    cropped_feature = st_cropper(raw_pil_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    
    if cropped_feature:
        # Convert PIL to numpy array safely
        feature_cv2 = np.array(cropped_feature)
        # Convert from RGB (PIL format) to BGR (OpenCV format)
        feature_cv2 = cv2.cvtColor(feature_cv2, cv2.COLOR_RGB2BGR)
        
        # --- LOW-LIGHT NOISE FILTERING & MASK SEGMENTATION ---
        gray = cv2.cvtColor(feature_cv2, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5,5), np.uint8)
        clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # --- METRIC 1: SURFACE AREA CALCULATION FROM EXTRACTED MASK ---
        total_mask_pixels = np.sum(clean_mask == 255)
        measured_area_cm2 = total_mask_pixels / (PIXELS_PER_CM ** 2)
        
        # --- METRIC 2: SKELETONIZED LENGTH VECTOR EXTRACTION ---
        binary_input = clean_mask // 255
        skeleton = skeletonize(binary_input)
        skeleton_output = (skeleton * 255).astype(np.uint8)
        
        y_indices, x_indices = np.where(skeleton_output > 0)
        
        if len(x_indices) > 2:
            coords = np.column_stack((x_indices, y_indices))
            dist_matrix = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
            max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
            
            # CRITICAL FIX: Explicitly cast coordinates to pure integers to prevent drawing crash
            pt1 = (int(coords[max_idx[0]][0]), int(coords[max_idx[0]][1]))
            pt2 = (int(coords[max_idx[1]][0]), int(coords[max_idx[1]][1]))
            
            # Map tracking layout onto frame safely
            visual_output = feature_cv2.copy()
            cv2.line(visual_output, pt1, pt2, (255, 0, 0), 3)       # Blue topological span vector
            cv2.circle(visual_output, pt1, 6, (0, 255, 0), -1)      # Endpoint node 1
            cv2.circle(visual_output, pt2, 6, (0, 255, 0), -1)      # Endpoint node 2
            
            # Math conversion
            pixel_distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            automated_length_cm = pixel_distance / PIXELS_PER_CM
            
            st.write("### Processing & Segmentation Results")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                # Convert back to RGB for crisp, true-color Streamlit display presentation
                display_img = cv2.cvtColor(visual_output, cv2.COLOR_BGR2RGB)
                st.image(display_img, caption=f"Measured {feature_type} Axis")
            with col_img2:
                st.image(clean_mask, caption="Low-Light Filter Mask (Used for Area)")
            
            # Metrics display parameters
            c1, c2 = st.columns(2)
            c1.metric(label="Calculated Length", value=f"{automated_length_cm:.2f} cm")
            c2.metric(label="Surface Area", value=f"{measured_area_cm2:.2f} cm²")
            
            # --- DATABASE DATA LOGGER ---
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
            st.warning("Could not isolate a solid skeletal vector. Readjust the red crop box closer to the chicken's edge.")
