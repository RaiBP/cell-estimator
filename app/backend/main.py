import logging
import os
from feature_extraction.feature_extractor import FeatureExtractor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path

from segmentation import utils as segmentation_utils
from pipeline import config as pipeline_config 
from classification.utils import create_classification_model
from segmentation.utils import create_segmentation_model

from pprint import pprint

from image_loader import (
    ImageLoader,
    prepare_phase_img,
    prepare_amplitude_img,
    encode_b64,
)

# Setting up logger
logging.basicConfig(level=logging.INFO)

# Initialization values. All of these can be latter changed via POST methods
# Initializing image loader for dataset
logging.info("Initializing image loader.")
data_folder = Path(os.environ["DATA_FOLDER"])
#data_folder = Path("/home/fidelinus/tum/applied_machine_intelligence/final_project/data")
dataset_path = data_folder / "real_world_sample01.pre"
image_loader = ImageLoader.from_file(dataset_path)
logging.info(f"Image loader initialized with {len(image_loader)} images.")

# Initializing image segmentator
logging.info("Initializing image segmentator.")
segmentation_method = pipeline_config["image_segmentator"]["method"]
# image_to_segment = pipeline_config["image_segmentator"]["image_to_segment"]
image_segmentator = create_segmentation_model(segmentation_method)
logging.info("Image segmentator initialized.")

logging.info("Initializing feature extractor.")
feature_extractor = FeatureExtractor()
logging.info("Feature extractor initialized.")

logging.info("Initializing classifier.")
classification_method = pipeline_config["classifier"]["method"]
classifier = create_classification_model(classification_method)
logging.info("Classifier initialized.")


class Polygon(BaseModel):
    points: List[float] | None


class PolygonWithPredictions(BaseModel):
    polygon: Polygon
    class_id: str
    confidence: dict
    features: dict


class ImageQuery(BaseModel):
    image_id: int
    image_type: int


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


@app.post("/select_dataset")
async def select_dataset(dataset_filename: str):
    """
    Method for changing the dataset file from which to load the images
    """
    global image_loader, data_folder, dataset_path
    logging.info("Initializing image loader with new dataset.")
    dataset_path = data_folder / dataset_filename
    image_loader = ImageLoader.from_file(dataset_path)
    logging.info(f"Image loader initialized with {len(image_loader)} images.")
    return DatasetInfo(file=dataset_path.name, num_images=len(image_loader))


@app.post("/select_segmentator")
async def select_segmentator(segmentation_method: str):
    """
    Method for initializing a new segmentator of type indicated by 'segmentation_method'
    """
    global image_segmentator
    logging.info(f"Initializing new segmentator of type {segmentation_method}.")
    image_segmentator = create_segmentation_model(segmentation_method)
    message = f"New segmentator of type {segmentation_method} initialized."
    logging.info(message)
    return {'message': message}


@app.post("/select_classifier")
async def select_classifier(classification_method: str):
    """
    Method for initializing a new classifier of type indicated by 'classification_method'
    """
    global classifier
    logging.info(f"Initializing new classifier of type {classification_method}.")
    classifier = create_classification_model(classification_method)
    message = f"New classifier of type {classification_method} initialized."
    logging.info(message)
    return {'message': message}


@app.post("/images")
async def get_images(image_query: ImageQuery):

    image_id = image_query.image_id
    image_type = image_query.image_type

    image_id = image_id % len(image_loader)
    if image_id not in image_loader:
        logging.warning(f"Image with id {image_id} not found.")
        return {"message": "Image not found"}

    amplitude_image, phase_image = image_loader.get_images(image_id)

    amplitude_image_str = prepare_amplitude_img(amplitude_image)
    phase_image_str = prepare_phase_img(phase_image)

    if segmentation_method in ["fastsam", "sam"]:
        image_to_be_segmented = amplitude_image_str if image_type == 0 else phase_image_str
    else:
        image_to_be_segmented = amplitude_image if image_type == 0 else phase_image

    try:
        masks = image_segmentator.segment_image(image_to_be_segmented)
        contours = [segmentation_utils.get_mask_contour(m) for m in masks]
        contours = [segmentation_utils.normalize_contour(c) for c in contours]
        contours = segmentation_utils.flatten_contours(contours)
    except Exception as e:
        logging.error(f"Error while segmenting image with id {image_id}: {e}")
        contours = []
        masks = []
    logging.info(f"Found {len(masks)} masks in image with id {image_id}")
    try:
        features = feature_extractor.extract_features(phase_image, amplitude_image, masks)
        features_records = features.to_dict('records')
    except Exception as e:
        logging.error(f"Error while extracting features from image with id {image_id}: {e}")
        features = None
    try:
        labels, probabilities = classifier.classify(features)
        print(labels)
        print(probabilities)
    except Exception as e:
        logging.error(f"Error while classifying image with id {image_id}: {e}")
        probabilities = []
        labels = []


    predictions = [
        PolygonWithPredictions(
            polygon=Polygon(points=polygon),
            class_id=label,
            confidence=prob,
            features=mask_features
        )
        for polygon, label, prob, mask_features in zip(contours, labels, probabilities, features_records)
    ]

    logging.info(f"Sending image with id {image_id} and {len(predictions)} predictions to client.")

    amplitude_image_b64 = encode_b64(amplitude_image_str)
    phase_img_b64 = encode_b64(phase_image_str)

    return ImagesWithPredictions(
        amplitude_img_data=amplitude_image_b64,
        phase_img_data=phase_img_b64,
        predictions=predictions,
    )
