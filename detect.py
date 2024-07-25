# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from ultralytics import YOLO
# from PIL import Image
# import pytesseract
# import base64
# from io import BytesIO
# import uvicorn
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust this in production for security
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load pretrained YOLOv8s model
# model_path = 'best.pt'
# model = YOLO(model_path)

# def get_coordinates(results):
#     coordinates_list = []
#     # Process each result
#     for result in results:
#         for i, bbox in enumerate(result.boxes.xyxy):  
#             xmin, ymin, xmax, ymax = map(int, bbox)
#             coordinates_list.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
#     return coordinates_list

# class FrameData(BaseModel):
#     frame: str

# @app.post("/process-frame")
# async def process_frame(data: FrameData):
#     try:
#         frame_data = data.frame
        
#         # Convert base64 image to PIL Image
#         image = Image.open(BytesIO(base64.b64decode(frame_data)))
        
#         # Process the frame with YOLO model
#         results = model(image, conf=0.70)
#         coordinates = get_coordinates(results)
        
#         detected_bus_numbers = ['A1', 'A2', 'D1', 'D2', 'K', 'E', 'BTC', '96', '95', '151', 'L', '153', '154', '156', '170', '186', '48', '67', '183', '188', '33', '10', '200', '201']
        
#         for coord in coordinates:
#             # Crop the image based on coordinates
#             cropped_image = image.crop((coord['xmin'], coord['ymin'], coord['xmax'], coord['ymax']))
            
#             # Perform OCR on the cropped image
#             detected_text = pytesseract.image_to_string(cropped_image)
#             print('Detected string: ' + detected_text)
            
#             # Check if the detected text contains any of the specified bus numbers
#             for bus_number in detected_bus_numbers:
#                 if bus_number in detected_text:
#                     print('Managed to detect bus number: ' + bus_number)
#                     return {"bus_number": bus_number}
        
#         return {"bus_number": "Bus not found"}
    
#     except Exception as e:
#         print(f"Error processing frame: {str(e)}")
#         raise HTTPException(status_code=500, detail="Error processing frame")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np
# import easyocr
# import base64
# from io import BytesIO
# import uvicorn
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust this in production for security
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load pretrained YOLOv8s model
# model_path = 'best.pt'
# model = YOLO(model_path)

# def get_coordinates(results):
#     coordinates_list = []
#     # Process each result
#     for result in results:
#         for i, bbox in enumerate(result.boxes.xyxy):  
#             xmin, ymin, xmax, ymax = map(int, bbox)
#             coordinates_list.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
#     return coordinates_list

# def ocr_to_string(image):
#     reader = easyocr.Reader(['en'])
#     print('easyocr ran successfully')
#     result = reader.readtext(np.array(image))
#     result_list = [text_tuple[1] for text_tuple in result]  # Convert tuple to list
#     return ' '.join(result_list)  # Concatenate list string elements into a string

# def find_substring(main_string, substrings):
#     for substring in substrings:
#         if substring in main_string:
#             return substring
#     return 'Bus not found'

# class FrameData(BaseModel):
#     frame: str

# @app.post("/process-frame")
# async def process_frame(data: FrameData):
#     try:
#         frame_data = data.frame
        
#         # Convert base64 image to PIL Image
#         image = Image.open(BytesIO(base64.b64decode(frame_data)))
#         print('image converted to PIL format')
        
#         # Process the frame with YOLO model
#         results = model(image, conf=0.70)
#         print('model ran successfully')
#         coordinates = get_coordinates(results)
        
#         detected_bus_numbers = ['A1', 'A2', 'D1', 'D2', 'K', 'E', 'BTC', '96', '95', '151', 'L', '153', '154', '156', '170', '186', '48', '67', '183', '188', '33', '10', '200', '201']
        
#         for coord in coordinates:
#             # Crop the image based on coordinates
#             cropped_image = image.crop((coord['xmin'], coord['ymin'], coord['xmax'], coord['ymax']))
            
#             # Perform OCR on the cropped image
#             detected_text = ocr_to_string(cropped_image)
#             print('Detected string: ' + detected_text)
            
#             # Check if the detected text contains any of the specified bus numbers
#             bus_number = find_substring(detected_text, detected_bus_numbers)
#             if bus_number != 'Bus not found':
#                 print('Managed to detect bus number: ' + bus_number)
#                 return {"bus_number": bus_number}
        
#         return {"bus_number": "Bus not found"}
    
#     except Exception as e:
#         print(f"Error processing frame: {str(e)}")
#         raise HTTPException(status_code=500, detail="Error processing frame")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests
import base64
from io import BytesIO
import uvicorn

app = FastAPI()

# Load pretrained YOLOv8s model
model_path = 'best.pt'
model = YOLO(model_path)

# API key for OCR.space
OCR_API_KEY = 'K84211199288957'

def ocr_to_string(image):
    # Convert PIL image to bytes
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    
    # Prepare request
    headers = {
        'apikey': OCR_API_KEY,
    }
    files = {
        'file': ('image.jpg', img_bytes),
    }
    data = {
        'language': 'eng',
        'isOverlayRequired': False,
    }
    
    # Send request to OCR.space API
    response = requests.post('https://api.ocr.space/parse/image', headers=headers, files=files, data=data)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="OCR.space API request failed")
    
    result = response.json()
    
    # Extract text from the result
    if 'ParsedResults' in result and len(result['ParsedResults']) > 0:
        parsed_text = result['ParsedResults'][0]['ParsedText']
    else:
        parsed_text = ''
    
    list_of_bus_numbers = ['A1', 'A2', 'D1', 'D2', 'K', 'E', 'BTC', '96', '95', '151', 
                           'L', '153', '154', '156', '170', '186', '48', '67', '183', 
                           '188', '33', '10', '200', '201']
    
    # Run detection model on bus number object
    model_path_cropped = 'best-cropped.pt'
    model_cropped = YOLO(model_path_cropped)
    results_cropped = model_cropped(image, conf=0.40)
    bus_number = 'Bus not found'
    
    for result_cropped in results_cropped:
        orig_img = result_cropped.orig_img
        
        for i, bbox in enumerate(result_cropped.boxes.xyxy):
            xmin, ymin, xmax, ymax = map(int, bbox)
            
            # Crop and do image processing for the detected object
            image_cropped = Image.fromarray(orig_img).crop((xmin, ymin, xmax, ymax))
    
            # Attempt 1 to identify bus number through bus number object
            buffered_cropped = BytesIO()
            image_cropped.save(buffered_cropped, format="JPEG")
            img_bytes_cropped = buffered_cropped.getvalue()
            
            files_cropped = {
                'file': ('image.jpg', img_bytes_cropped),
            }
            response_cropped = requests.post('https://api.ocr.space/parse/image', headers=headers, files=files_cropped, data=data)
            
            if response_cropped.status_code == 200:
                result_cropped = response_cropped.json()
                if 'ParsedResults' in result_cropped and len(result_cropped['ParsedResults']) > 0:
                    parsed_text_cropped = result_cropped['ParsedResults'][0]['ParsedText']
                else:
                    parsed_text_cropped = ''
                    
                bus_number = find_substring(parsed_text_cropped, list_of_bus_numbers)
                
            if bus_number != 'Bus not found':
                return bus_number
    
    # Attempt 2 to identify bus number through bus object
    return find_substring(parsed_text, list_of_bus_numbers)

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

class FrameData(BaseModel):
    frame: str
    
@app.post("/process-frame")
async def process_frame(data: FrameData):
    try:
        frame_data = data.frame
        
        # Convert base64 image to PIL Image
        image = Image.open(BytesIO(base64.b64decode(frame_data)))
        
        # Process the frame with YOLO model
        results = model(image, conf=0.70)
        bus_result = process_results(results)
        
        return {"result": bus_result}
    
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing frame")

if __name__ == "main":
    uvicorn.run(app, host="0.0.0.0", port=8501)