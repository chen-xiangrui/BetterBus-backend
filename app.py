import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO

# Set up page configuration
st.set_page_config(
    page_title="My YOLO App",
    page_icon="🚀"
)

# Page title
st.title('My YOLO App')

# Description
st.markdown('This is an application for object detection using YOLO')

# Function to convert file buffer to cv2 image
def create_opencv_image_from_stringio(img_stream, cv2_img_flag=1):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

# Load the YOLO model
model_path = 'best.pt'
model = YOLO(model_path)

# File uploader
img_files = st.file_uploader(label="Choose image files",
                             type=['png', 'jpg', 'jpeg'],
                             accept_multiple_files=True)

# Process each uploaded image
for n, img_file_buffer in enumerate(img_files):
    if img_file_buffer is not None:
        # Convert image file buffer to OpenCV image
        open_cv_image = create_opencv_image_from_stringio(img_file_buffer)

        # Pass image to the model to get the detection result
        results = model(open_cv_image, conf=0.25)
        for result in results:
            im0 = result.plot()
        
        # Show result image using st.image()
        if im0 is not None:
            st.image(im0, channels="BGR",
                     caption=f'Detection Results ({n+1}/{len(img_files)})')

# Footer
st.markdown("""
  <p style='text-align: center; font-size:16px; margin-top: 32px'>
    AwesomePython @2020
  </p>
""", unsafe_allow_html=True)
