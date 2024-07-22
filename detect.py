# from flask import Flask, request, jsonify
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np
# import easyocr
# import base64
# from io import BytesIO
# from gtts import gTTS
# import os

# app = Flask(__name__)

# # Load pretrained YOLOv8s model
# model_path = 'best.pt'
# model = YOLO(model_path)

# # Translate text on image to string
# def ocr_to_string(image):
#     image_PIL_form = image
#     image = np.array(image)
#     reader = easyocr.Reader(['en'])
#     result = reader.readtext(image)
#     result_list = [text_tuple[1] for text_tuple in result]  # convert tuple to list
#     result_string = ' '.join(result_list)  # concat list string elements into a string
#     list_of_bus_numbers = ['A1', 'A2', 'D1', 'D2', 'K', 'E', 'BTC', '96', '95', '151', 
#                            'L', '153', '154', '156', '170', '186', '48', '67', '183', 
#                            '188', '33', '10', '200', '201']
    
#     # Run detection model on bus number object
#     model_path_cropped = 'best-cropped.pt'
#     model_cropped = YOLO(model_path_cropped)
#     results_cropped = model_cropped(image_PIL_form, conf=0.40)
#     image_cropped = None
#     bus_number = 'Bus not found'
#     for result_cropped in results_cropped:
#         orig_img = result_cropped.orig_img
        
#         for i, bbox in enumerate(result_cropped.boxes.xyxy):
#             xmin, ymin, xmax, ymax = map(int, bbox)
            
#             # Crop and do image processing for the detected object
#             image_cropped = Image.fromarray(orig_img).crop((xmin, ymin, xmax, ymax))
    
#             # Attempt 1 to identify bus number through bus number object
#             image_cropped = np.array(image_cropped)
#             reader_cropped = easyocr.Reader(['en'])
#             result_cropped = reader_cropped.readtext(image_cropped)
#             result_list_cropped = [text_tuple[1] for text_tuple in result_cropped]  # convert tuple to list
#             result_string_cropped = ' '.join(result_list_cropped)  # concat list string elements into a string 
#             bus_number = find_substring(result_string_cropped, list_of_bus_numbers)
            
#         if bus_number != 'Bus not found':
#             return bus_number
    
#     # Attempt 2 to identify bus number through bus object
#     return find_substring(result_string, list_of_bus_numbers)

# def find_substring(main_string, substrings):
#     for substring in substrings:
#         if substring in main_string:
#             return substring
#     return 'Bus not found'

# def process_results(results):
#     # Process each result
#     for result in results:
#         img = result.orig_img

#         for i, bbox in enumerate(result.boxes.xyxy):  
#             xmin, ymin, xmax, ymax = map(int, bbox)
#             ymax -= (1 / 4) * (ymax - ymin)
#             # Crop the detected object
#             cropped_img = Image.fromarray(img).crop((xmin, ymin, xmax, ymax))
            
#             return ocr_to_string(cropped_img)

# @app.route('/process-frame', methods=['POST'])
# def process_frame():
#     try:
#         data = request.json
#         frame_data = data['frame']
        
#         # Convert base64 image to PIL Image
#         image = Image.open(BytesIO(base64.b64decode(frame_data)))
        
#         # Process the frame with YOLO model
#         results = model(image, conf=0.70)
#         bus_result = process_results(results)
        
#         return jsonify({'result': bus_result}), 200
    
#     except Exception as e:
#         print(f"Error processing frame: {str(e)}")
#         return jsonify({'error': 'Error processing frame'}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

# import streamlit as st
# from flask import Flask, request, jsonify
# from threading import Thread
# import base64
# from io import BytesIO
# from PIL import Image
# import torch
# from ultralytics import YOLO

# # Initialize Flask app
# flask_app = Flask(__name__)

# # Load your YOLO model
# model = YOLO('best.pt')

# # Streamlit UI
# st.title("YOLO Object Detection")
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     if image.mode != 'RGB':
#         image = image.convert('RGB')

#     results = model(image, conf=0.7)
#     st.image(image, caption='Uploaded Image', use_column_width=True)
#     st.write("Detecting objects...")

#     for result in results:
#         img = result.orig_img
#         for i, bbox in enumerate(result.boxes.xyxy):
#             xmin, ymin, xmax, ymax = map(int, bbox)
#             ymax -= (1 / 4) * (ymax - ymin)
#             cropped_img = Image.fromarray(img).crop((xmin, ymin, xmax, ymax))
#             st.write(cropped_img)

# # Flask API endpoint
# @flask_app.route('/process-frame', methods=['POST'])
# def process_frame():
#     try:
#         data = request.json
#         frame_data = data['frame']
#         image = Image.open(BytesIO(base64.b64decode(frame_data)))
#         if image.mode != 'RGB':
#             image = image.convert('RGB')

#         results = model(image, conf=0.7)
#         for result in results:
#             img = result.orig_img
#             for i, bbox in enumerate(result.boxes.xyxy):
#                 xmin, ymin, xmax, ymax = map(int, bbox)
#                 ymax -= (1 / 4) * (ymax - ymin)
#                 cropped_img = Image.fromarray(img).crop((xmin, ymin, xmax, ymax))
#                 # For simplicity, return the bounding box coordinates
#                 return jsonify({
#                     "xmin": xmin,
#                     "ymin": ymin,
#                     "xmax": xmax,
#                     "ymax": ymax
#                 })

#         return jsonify({"message": "No objects detected"}), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Run Flask app in a separate thread
# def run_flask():
#     flask_app.run(port=8501)

# thread = Thread(target=run_flask)
# thread.start()






# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np
# import easyocr
# import base64
# from io import BytesIO
# from threading import Thread

# app = Flask(__name__)
# CORS(app)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# # Load pretrained YOLOv8s model
# model_path = 'best.pt'
# model = YOLO(model_path)

# # Translate text on image to string
# def ocr_to_string(image):
#     image_PIL_form = image
#     image = np.array(image)
#     reader = easyocr.Reader(['en'])
#     result = reader.readtext(image)
#     result_list = [text_tuple[1] for text_tuple in result]  # convert tuple to list
#     result_string = ' '.join(result_list)  # concat list string elements into a string
#     list_of_bus_numbers = ['A1', 'A2', 'D1', 'D2', 'K', 'E', 'BTC', '96', '95', '151', 
#                            'L', '153', '154', '156', '170', '186', '48', '67', '183', 
#                            '188', '33', '10', '200', '201']
    
#     # Run detection model on bus number object
#     model_path_cropped = 'best-cropped.pt'
#     model_cropped = YOLO(model_path_cropped)
#     results_cropped = model_cropped(image_PIL_form, conf=0.40)
#     image_cropped = None
#     bus_number = 'Bus not found'
#     for result_cropped in results_cropped:
#         orig_img = result_cropped.orig_img
        
#         for i, bbox in enumerate(result_cropped.boxes.xyxy):
#             xmin, ymin, xmax, ymax = map(int, bbox)
            
#             # Crop and do image processing for the detected object
#             image_cropped = Image.fromarray(orig_img).crop((xmin, ymin, xmax, ymax))
    
#             # Attempt 1 to identify bus number through bus number object
#             image_cropped = np.array(image_cropped)
#             reader_cropped = easyocr.Reader(['en'])
#             result_cropped = reader_cropped.readtext(image_cropped)
#             result_list_cropped = [text_tuple[1] for text_tuple in result_cropped]  # convert tuple to list
#             result_string_cropped = ' '.join(result_list_cropped)  # concat list string elements into a string 
#             bus_number = find_substring(result_string_cropped, list_of_bus_numbers)
            
#         if bus_number != 'Bus not found':
#             return bus_number
    
#     # Attempt 2 to identify bus number through bus object
#     return find_substring(result_string, list_of_bus_numbers)

# def find_substring(main_string, substrings):
#     for substring in substrings:
#         if substring in main_string:
#             return substring
#     return 'Bus not found'

# def process_results(results):
#     # Process each result
#     for result in results:
#         img = result.orig_img

#         for i, bbox in enumerate(result.boxes.xyxy):  
#             xmin, ymin, xmax, ymax = map(int, bbox)
#             ymax -= (1 / 4) * (ymax - ymin)
#             # Crop the detected object
#             cropped_img = Image.fromarray(img).crop((xmin, ymin, xmax, ymax))
            
#             return ocr_to_string(cropped_img)
#     return 'Bus not found'

# @app.route('/process-frame', methods=['POST'])
# def process_frame():
#     try:
#         data = request.json
#         frame_data = data['frame']
#         # Convert base64 image to PIL Image
#         image = Image.open(BytesIO(base64.b64decode(frame_data)))
#         # Process the frame with YOLO model
#         results = model(image, conf=0.70)
#         bus_result = process_results(results)
#         return jsonify({'result': bus_result}), 200
    
#     except Exception as e:
#         print(f"Error processing frame: {str(e)}")
#         return jsonify({'error': 'Error processing frame'}), 500

# def run_flask():
#     app.run(debug=True, use_reloader=False)

# thread = Thread(target=run_flask)
# thread.start()





# import streamlit as st
# import base64
# from io import BytesIO
# from PIL import Image
# import numpy as np
# import easyocr
# from ultralytics import YOLO

# # Load pretrained YOLOv8s model
# model_path = 'best.pt'
# model = YOLO(model_path)

# # Load YOLOv8s cropped model
# model_path_cropped = 'best-cropped.pt'
# model_cropped = YOLO(model_path_cropped)

# def ocr_to_string(image):
#     image_PIL_form = image
#     image = np.array(image)
#     reader = easyocr.Reader(['en'])
#     result = reader.readtext(image)
#     result_list = [text_tuple[1] for text_tuple in result]
#     result_string = ' '.join(result_list)
#     list_of_bus_numbers = ['A1', 'A2', 'D1', 'D2', 'K', 'E', 'BTC', '96', '95', '151', 
#                            'L', '153', '154', '156', '170', '186', '48', '67', '183', 
#                            '188', '33', '10', '200', '201']

#     results_cropped = model_cropped(image_PIL_form, conf=0.40)
#     image_cropped = None
#     bus_number = 'Bus not found'
#     for result_cropped in results_cropped:
#         orig_img = result_cropped.orig_img
#         for i, bbox in enumerate(result_cropped.boxes.xyxy):
#             xmin, ymin, xmax, ymax = map(int, bbox)
#             image_cropped = Image.fromarray(orig_img).crop((xmin, ymin, xmax, ymax))
#             image_cropped = np.array(image_cropped)
#             reader_cropped = easyocr.Reader(['en'])
#             result_cropped = reader_cropped.readtext(image_cropped)
#             result_list_cropped = [text_tuple[1] for text_tuple in result_cropped]
#             result_string_cropped = ' '.join(result_list_cropped)
#             bus_number = find_substring(result_string_cropped, list_of_bus_numbers)
#         if bus_number != 'Bus not found':
#             return bus_number

#     return find_substring(result_string, list_of_bus_numbers)

# def find_substring(main_string, substrings):
#     for substring in substrings:
#         if substring in main_string:
#             return substring
#     return 'Bus not found'

# def process_image(image):
#     results = model(image, conf=0.70)
#     for result in results:
#         img = result.orig_img
#         for i, bbox in enumerate(result.boxes.xyxy):  
#             xmin, ymin, xmax, ymax = map(int, bbox)
#             ymax -= (1 / 4) * (ymax - ymin)
#             cropped_img = Image.fromarray(img).crop((xmin, ymin, xmax, ymax))
#             return ocr_to_string(cropped_img)
#     return 'Bus not found'

# st.title('Bus Number Detection')

# uploaded_file = st.file_uploader("Choose an image...", type="jpg")
# if uploaded_file:
#     image = Image.open(uploaded_file)
#     bus_result = process_image(image)
#     st.write({"result": bus_result})




import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
import easyocr
from ultralytics import YOLO
import base64
import json

# Load pretrained YOLOv8s models
model_path = 'best.pt'
model = YOLO(model_path)

model_path_cropped = 'best-cropped.pt'
model_cropped = YOLO(model_path_cropped)

def ocr_to_string(image):
    image_PIL_form = image
    image = np.array(image)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    result_list = [text_tuple[1] for text_tuple in result]
    result_string = ' '.join(result_list)
    list_of_bus_numbers = ['A1', 'A2', 'D1', 'D2', 'K', 'E', 'BTC', '96', '95', '151', 
                           'L', '153', '154', '156', '170', '186', '48', '67', '183', 
                           '188', '33', '10', '200', '201']

    results_cropped = model_cropped(image_PIL_form, conf=0.40)
    image_cropped = None
    bus_number = 'Bus not found'
    for result_cropped in results_cropped:
        orig_img = result_cropped.orig_img
        for i, bbox in enumerate(result_cropped.boxes.xyxy):
            xmin, ymin, xmax, ymax = map(int, bbox)
            image_cropped = Image.fromarray(orig_img).crop((xmin, ymin, xmax, ymax))
            image_cropped = np.array(image_cropped)
            reader_cropped = easyocr.Reader(['en'])
            result_cropped = reader_cropped.readtext(image_cropped)
            result_list_cropped = [text_tuple[1] for text_tuple in result_cropped]
            result_string_cropped = ' '.join(result_list_cropped)
            bus_number = find_substring(result_string_cropped, list_of_bus_numbers)
        if bus_number != 'Bus not found':
            return bus_number

    return find_substring(result_string, list_of_bus_numbers)

def find_substring(main_string, substrings):
    for substring in substrings:
        if substring in main_string:
            return substring
    return 'Bus not found'

def process_image(image):
    results = model(image, conf=0.70)
    for result in results:
        img = result.orig_img
        for i, bbox in enumerate(result.boxes.xyxy):  
            xmin, ymin, xmax, ymax = map(int, bbox)
            ymax -= (1 / 4) * (ymax - ymin)
            cropped_img = Image.fromarray(img).crop((xmin, ymin, xmax, ymax))
            return ocr_to_string(cropped_img)
    return 'Bus not found'

# Streamlit UI
st.title('Bus Number Detection')

# File upload component
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# API simulation via file upload or sidebar input
if uploaded_file:
    image = Image.open(uploaded_file)
    bus_result = process_image(image)
    st.json({"result": bus_result})

# API simulation via sidebar
st.sidebar.header('API Input Simulation')
base64_data = st.sidebar.text_area('Base64 Image Data', '')
if base64_data:
    try:
        frame_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(frame_data))
        bus_result = process_image(image)
        st.sidebar.json({"result": bus_result})
    except Exception as e:
        st.sidebar.json({"error": str(e)})
