from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import numpy as np
import easyocr
import cv2
from gtts import gTTS
import pygame

# Define the path to the pretrained model and the source directory
model_path = Path('best.pt')

# Load pretrained YOLOv8s model
model = YOLO(model_path)

# translate text on image to string
def ocr_to_string(image):
    image_PIL_form = image
    image = np.array(image)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    result_list = [text_tuple[1] for text_tuple in result] # convert tuple to list
    result_string = ' '.join(result_list) # concat list string elements into a string
    list_of_bus_numbers = ['A1', 'A2', 'D1', 'D2', 'K', 'E', 'BTC', '96', '95', '151', 
                           'L', '153', '154', '156', '170', '186', '48', '67', '183', 
                           '188', '33', '10', '200', '201']
    
    # run detection model on bus number object
    model_path_cropped = Path('best-cropped.pt')
    model_cropped = YOLO(model_path_cropped)
    results_cropped = model_cropped(source=image_PIL_form, conf = 0.40)
    image_cropped = None
    bus_number = 'Bus not found'
    for result_cropped in results_cropped:
        orig_img = result_cropped.orig_img
        
        for i, bbox in enumerate(result_cropped.boxes.xyxy):
            xmin, ymin, xmax, ymax = map(int, bbox)
            
            # Crop and do image processing for the detected object
            image_cropped = Image.fromarray(orig_img).crop((xmin, ymin, xmax, ymax))
    
            # attempt 1 to identify bus number through bus number object
            image_cropped = np.array(image_cropped)
            reader_cropped = easyocr.Reader(['en'])
            result_cropped = reader_cropped.readtext(image_cropped)
            result_list_cropped = [text_tuple[1] for text_tuple in result_cropped] # convert tuple to list
            result_string_cropped = ' '.join(result_list_cropped) # concat list string elements into a string 
            bus_number = find_substring(result_string_cropped, list_of_bus_numbers)
            
        if bus_number != 'Bus not found':
            return bus_number
    
        # attempt 2 to identify bus number through bus object
        return find_substring(result_string, list_of_bus_numbers)
    
    # case where no bus object is detected (??)
    return 'Bus not found'

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
        
# Open primary camera
cap = cv2.VideoCapture(0)
count = 0
current_audio_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.70)
    count = count + 1
    bus_result = process_results(results)
        
    if bus_result != '' and bus_result != 'Bus not found' and bus_result is not None and (count > current_audio_count + 100): 
        language = 'en'
        
        try:
            audio_obj = gTTS(text=bus_result, lang=language, slow=False)
            audio_file = 'audio/bus-audio.mp3'
            audio_obj.save(audio_file)
            # Initialize the mixer module
            pygame.mixer.init()
            # Load the mp3 file
            pygame.mixer.music.load(audio_file)
            # Play the loaded mp3 file
            pygame.mixer.music.play()
        except Exception as e:
            continue

    # Break the loop on 'q' key press 
    # in app, it will be the back button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Close camera
cap.release()
cv2.destroyAllWindows()