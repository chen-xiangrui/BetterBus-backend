from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import base64
from io import BytesIO
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pretrained YOLOv8s model
model_path = 'best.pt'
model = YOLO(model_path)

def get_coordinates(results):
    coordinates_list = []
    # Process each result
    for result in results:
        for i, bbox in enumerate(result.boxes.xyxy):  
            xmin, ymin, xmax, ymax = map(int, bbox)
            coordinates_list.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    return coordinates_list

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
        coordinates = get_coordinates(results)
        
        return {"coordinates": coordinates}
    
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)