from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
from car_damage_detection import CarDamageDetectionCNN

app = FastAPI(title="Car Damage Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
detector = CarDamageDetectionCNN()

@app.get("/")
def read_root():
    return {"message": "Car Damage Detection API is running"}

@app.post("/predict")
async def predict_damage(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = detector.predict(contents)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)