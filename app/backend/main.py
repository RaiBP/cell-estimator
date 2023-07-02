import base64
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path

from image_loader import (
    ImageLoader,
    prepare_phase_img,
    prepare_amplitude_img,
    encode_b64,
)
from image_segmentator import FastSAMImageSegmentator

# Setting up logger
logging.basicConfig(level=logging.INFO)


# Initializing image loader for dataset
logging.info("Initializing image loader.")
dataset_path = Path(
    "/home/fidelinus/tum/applied_machine_intelligence/final_project/data/real_world_sample01.pre"
)
image_loader = ImageLoader.from_file(dataset_path)
logging.info(f"Image loader initialized with {len(image_loader)} images.")

# Initializing image segmentator
logging.info("Initializing image segmentator.")
image_segmentator = FastSAMImageSegmentator()
logging.info("Image segmentator initialized.")


class Polygon(BaseModel):
    points: List[float] | None


class PolygonWithPredictions(BaseModel):
    polygon: Polygon
    class_id: int
    confidence: float


class ImageId(BaseModel):
    image_id: int


class ImagesWithPredictions(BaseModel):
    amplitude_img_data: str
    phase_img_data: str
    predictions: List[PolygonWithPredictions]


class DatasetInfo(BaseModel):
    file: str
    classes: List[str] = ["rbc", "wbc", "plt", "agg", "oof"]
    classes_description: List[str] = [
        "Red Blood Cell",
        "White Blood Cell",
        "Platelet",
        "Aggregation",
        "Out of Focus",
    ]
    num_images: int


app = FastAPI()

origins = ["http://localhost:3000", "https://localhost:3000"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/dataset_info")
async def get_dataset_info():
    return DatasetInfo(file=dataset_path.name, num_images=len(image_loader))


@app.post("/images")
async def get_images(image_id: ImageId):
    image_id = image_id.image_id % len(image_loader)
    if image_id not in image_loader:
        logging.warning(f"Image with id {image_id} not found.")
        return {"message": "Image not found"}

    amplitude_image, phase_image = image_loader.get_images(image_id)
    amplitude_image, phase_image = prepare_amplitude_img(
        amplitude_image
    ), prepare_phase_img(phase_image)

    image_segmentator.set_image(amplitude_image, image_id)
    polygons = image_segmentator.segment()
    predictions = [
        PolygonWithPredictions(
            polygon=Polygon(points=polygon),
            class_id=0,
            confidence=1.0,
        )
        for polygon in polygons
    ]

    amplitude_image_b64 = encode_b64(amplitude_image)
    phase_img_b64 = encode_b64(phase_image)

    logging.info(f"Sending images with id {image_id} to client.")

    return ImagesWithPredictions(
        amplitude_img_data=amplitude_image_b64,
        phase_img_data=phase_img_b64,
        predictions=predictions,
    )
