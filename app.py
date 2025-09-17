# app.py

import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
import imageio # For creating GIFs
import time
from streamlit_image_coordinates import image_coordinates # The new component

# Import the main function from your refactored script
from sticker_generator import generate_sticker_frames

# --- 1. SET UP THE PAGE AND CONSTANTS ---
st.set_page_config(layout="wide", page_title="GIF & Sticker Maker")

MODEL_PATHS = {
    "sam": "weights/sam_hq_vit_h.pth",
    "cotracker": "weights/cotracker_stride_4_wind_8.pth"
}

# --- 2. HELPER FUNCTION TO DRAW POINTS ---
def draw_points_on_image(image, points):
    """Draws circles on the image at the given point coordinates."""
    img_with_points = image.copy()
    for (x, y) in points:
        cv2.circle(img_with_points, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(img_with_points, (x, y), radius=5, color=(0, 0, 0), thickness=1) # Black outline
    return img_with_points

# --- 3. INITIALIZE SESSION STATE ---
# Session state is used to store variables between reruns of the script.
if "points" not in st.session_state:
    st.session_state.points = []
if "first_frame" not in st.session_state:
    st.session_state.first_frame = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "gif_path" not in st.session_state:
    st.session_state.gif_path = None

# --- 4. BUILD THE STREAMLIT INTERFACE ---

st.title("✂️ Automatic GIF & Sticker Maker")
st.markdown("Powered by **SAM-PT**. Upload a video, click on an object, and generate a transparent GIF.")

# Check for model weights
models_exist = os.path.exists(MODEL_PATHS["sam"]) and os.path.exists(MODEL_PATHS["cotracker"])
if not models_exist:
    st.error(
        """
        **ERROR: Model weights not found!**
        Please make sure the `weights/sam_hq_vit_h.pth` and `weights/cotracker_stride_4_wind_8.pth` files exist.
        """
    )
    st.stop()


col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Your Video")
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

    # When a new video is uploaded, process its first frame
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location to get a stable path
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # If it's a new video, reset the state
        if st.session_state.video_path != video_path:
            st.session_state.video_path = video_path
            st.session_state.points = []
            st.session_state.gif_path = None
            
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                st.session_state.first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

with col2:
    st.subheader("2. Select Points on the Object")
    if st.session_state.first_frame is not None:
        # Create an image with points drawn on it for display
        image_to_display = draw_points_on_image(st.session_state.first_frame, st.session_state.points)

        # Use the image_coordinates component to get clicks
        coords = image_coordinates(image_to_display, key="local")
        
        # When a user clicks, the script reruns and `coords` will have a value
        if coords:
            # Check to avoid adding the same point multiple times on quick reruns
            if coords not in st.session_state.points:
                st.session_state.points.append(coords)
                # Force a rerun to redraw the image with the new point
                st.rerun()

        # Display the number of points selected
        st.write(f"**Selected Points:** {len(st.session_state.points)}")
        if st.session_state.points:
             if st.button("Clear Last Point"):
                st.session_state.points.pop()
                st.rerun()
    else:
        st.info("Upload a video to select points.")

st.divider()

st.subheader("3. Generate and Download")
if st.button("Generate Sticker GIF", disabled=(st.session_state.first_frame is None or not st.session_state.points)):
    with st.spinner("Processing started... This may take a few minutes."):
        st.success("Model inference is running...")
        
        # Call the main function
        original_frames, generated_masks = generate_sticker_frames(
            video_path=st.session_state.video_path,
            points_coords=st.session_state.points,
            model_paths=MODEL_PATHS,
        )
        
        st.info("Inference complete. Creating transparent GIF...")

        # Create transparent frames for the GIF
        transparent_frames = []
        for frame_np, mask_np in zip(original_frames, generated_masks):
            frame_rgba = np.concatenate([frame_np, np.full((frame_np.shape[0], frame_np.shape[1], 1), 255, dtype=np.uint8)], axis=-1)
            frame_rgba[:, :, 3] = mask_np * 255
            transparent_frames.append(frame_rgba)

        # Save the GIF
        gif_path = "output_sticker.gif"
        imageio.mimsave(gif_path, transparent_frames, fps=10, loop=0)
        st.session_state.gif_path = gif_path

if st.session_state.gif_path:
    st.image(st.session_state.gif_path, caption="Generated Sticker GIF")
    with open(st.session_state.gif_path, "rb") as file:
        st.download_button(
            label="Download GIF",
            data=file,
            file_name="sticker.gif",
            mime="image/gif"
        )