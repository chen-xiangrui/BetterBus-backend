from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np
import easyocr
import base64
from io import BytesIO
from gtts import gTTS
import os
import pygame

app = Flask(__name__)

# Load pretrained YOLOv8s model
model_path = 'best.pt'
model = YOLO(model_path)

# Initialize Pygame mixer for audio playback
pygame.mixer.init()

# Function to play audio
def play_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_file = 'bus_audio.mp3'
    tts.save(audio_file)
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

# Translate text on image to string
def ocr_to_string(image):
    image_PIL_form = image
    image = np.array(image)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    result_list = [text_tuple[1] for text_tuple in result]  # convert tuple to list
    result_string = ' '.join(result_list)  # concat list string elements into a string
    list_of_bus_numbers = ['A1', 'A2', 'D1', 'D2', 'K', 'E', 'BTC', '96', '95', '151', 
                           'L', '153', '154', '156', '170', '186', '48', '67', '183', 
                           '188', '33', '10', '200', '201']
    
    # Run detection model on bus number object
    model_path_cropped = 'best-cropped.pt'
    model_cropped = YOLO(model_path_cropped)
    results_cropped = model_cropped(image_PIL_form, conf=0.40)
    image_cropped = None
    bus_number = 'Bus not found'
    for result_cropped in results_cropped:
        orig_img = result_cropped.orig_img
        
        for i, bbox in enumerate(result_cropped.boxes.xyxy):
            xmin, ymin, xmax, ymax = map(int, bbox)
            
            # Crop and do image processing for the detected object
            image_cropped = Image.fromarray(orig_img).crop((xmin, ymin, xmax, ymax))
    
            # Attempt 1 to identify bus number through bus number object
            image_cropped = np.array(image_cropped)
            reader_cropped = easyocr.Reader(['en'])
            result_cropped = reader_cropped.readtext(image_cropped)
            result_list_cropped = [text_tuple[1] for text_tuple in result_cropped]  # convert tuple to list
            result_string_cropped = ' '.join(result_list_cropped)  # concat list string elements into a string 
            bus_number = find_substring(result_string_cropped, list_of_bus_numbers)
            
        if bus_number != 'Bus not found':
            return bus_number
    
    # Attempt 2 to identify bus number through bus object
    return find_substring(result_string, list_of_bus_numbers)

def find_substring(main_string, substrings):
    for substring in substrings:
        if substring in main_string:
            return substring
    return 'Bus not found'

def process_results(results):
    # Process each result
    for result in results:
        img = result.orig_img

        for i, bbox in enumerate(result.boxes.xyxy):  
            xmin, ymin, xmax, ymax = map(int, bbox)
            ymax -= (1 / 4) * (ymax - ymin)
            # Crop the detected object
            cropped_img = Image.fromarray(img).crop((xmin, ymin, xmax, ymax))
            
            return ocr_to_string(cropped_img)

@app.route('/process-frame', methods=['POST'])
def process_frame():
    try:
        data = request.json
        frame_data = data['frame']
        
        # Convert base64 image to PIL Image
        image = Image.open(BytesIO(base64.b64decode(frame_data)))
        
        # Process the frame with YOLO model
        results = model(image, conf=0.70)
        bus_result = process_results(results)
        
        if bus_result != 'Bus not found':
            play_audio(bus_result)
        
        return jsonify({'result': bus_result}), 200
    
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({'error': 'Error processing frame'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
