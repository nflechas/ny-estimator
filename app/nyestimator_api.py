import logging
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.random_forest import NYEstimatorModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path.cwd().parent / "models/original_simple_classifier.pkl"
logger.info("Initializing ModelService...")
MODEL = NYEstimatorModel(MODEL_PATH)
logger.info("ModelService initialized.")


# Define request and response models for FastAPI
class PredictionRequest(
    BaseModel
):  # Sample values are left for illustration purposes and easier debugging
    id: int = 1001
    accommodates: int = 4
    room_type: str = "Entire home/apt"
    beds: int = 2
    bedrooms: int = 1
    bathrooms: int = 2
    neighbourhood: str = "Brooklyn"
    tv: int = 1
    elevator: int = 1
    internet: int = 0
    latitude: float = 40.71383
    longitude: float = -73.9658


class PredictionResponse(BaseModel):
    id: int = 1001
    price_category: str = "High"


# Initialize FastAPI app
app = FastAPI()


@app.post("/predict", response_model=PredictionResponse)
async def predict(self, request: PredictionRequest):
    try:
        print(f"trying to predit on {request}")
        prediction = MODEL.predict([request.dict()])[0]
        return PredictionResponse(
            id=prediction["id"], price_category=prediction["price_category"]
        )
    except Exception as e:
        logger.error(f"Prediction request failed: {e}")
        logger.error(traceback.format_exc())

        raise HTTPException(status_code=500, detail=str(e))
