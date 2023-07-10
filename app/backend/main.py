import os
import base64
import logging
from feature_extraction.feature_extractor import FeatureExtractor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

from segmentation import utils as segmentation_utils
from pipeline import config as pipeline_config 
from classification.utils import create_classification_model
from segmentation.utils import create_segmentation_model
from backend_functions import *

from image_loader import (
    ImageLoader,
    prepare_phase_img,
    prepare_amplitude_img,
    encode_b64,
)

# Setting up logger
logging.basicConfig(level=logging.INFO)


# Initializing image loader for dataset
logging.info("Initializing image loader.")
data_folder = Path(os.environ["DATA_FOLDER"])
dataset_path = data_folder / "real_world_sample01.pre"
image_loader = ImageLoader.from_file(dataset_path)
logging.info(f"Image loader initialized with {len(image_loader)} images.")

# Initializing image segmentator
logging.info("Initializing image segmentator.")
segmentation_method = pipeline_config["image_segmentator"]["method"]
image_to_segment = pipeline_config["image_segmentator"]["image_to_segment"]
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


class ImageId(BaseModel):
    image_id: int


class ImagesGivenID(BaseModel):
    amplitude_img_data: str
    phase_img_data: str


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

class PredictionsList(BaseModel):
    predictions: List[str]


class ListOfLists(BaseModel):
    coordinates: List[List[int]]


app = FastAPI()

# Variable to store the features calculated from feature extraction
shared_features = None

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


@app.post("/segment")
async def segment_image(image_id: ImageId): 
    image_id = image_id.image_id % len(image_loader)
    if image_id not in image_loader:
        logging.warning(f"Image with id {image_id} not found.")
        return {"message": "Image not found"}

    amplitude_image, phase_image = image_loader.get_images(image_id)

    amplitude_image_str = prepare_amplitude_img(amplitude_image)
    phase_image_str = prepare_phase_img(phase_image)

    if segmentation_method in ["fastsam", "sam"]:
        image_to_be_segmented = amplitude_image_str if image_to_segment == "amplitude" else phase_image_str
    else:
        image_to_be_segmented = amplitude_image if image_to_segment == "amplitude" else phase_image

    try:
        masks = image_segmentator.segment_image(image_to_be_segmented)
        contours = [segmentation_utils.get_mask_contour(m) for m in masks]
        contours = [segmentation_utils.normalize_contour(c) for c in contours]
        contours = segmentation_utils.flatten_contours(contours)
        logging.info(f"Masks of image {image_id} calculated succesfully.")
    except Exception as e:
        logging.error(f"Error while segmenting image with id {image_id}: {e}")
        contours = []
        masks = []
    logging.info(f"Found {len(masks)} masks in image with id {image_id}")


    predictions = [
        PolygonWithPredictions(
            polygon=Polygon(points=polygon),
            class_id=labels[idx],
            confidence=probabilities[idx],
        )
        for idx, polygon in enumerate(contours)
    ]

    logging.info(f"Sending image with id {image_id} and {len(predictions)} predictions to client.")

    amplitude_image_b64 = encode_b64(amplitude_image_str)
    phase_img_b64 = encode_b64(phase_image_str)

    return ImagesWithPredictions(
        amplitude_img_data=amplitude_image_b64,
        phase_img_data=phase_img_b64,
        predictions=predictions,
    )


@app.post("/send_images")
async def send_images(image_id: ImageId):
    """
    Method for sending phase and amplitude images indexed by "image_id" in file given by "dataset_path".
    """
    global image_loader, logging
    
    amplitude_image_b64, phase_image_b64 = load_image(image_id, image_loader, logging)
    
    return ImagesGivenID(
        amplitude_img_data=amplitude_image_b64,
        phase_img_data=phase_image_b64,
    )



@app.post("/images")
async def get_images(image_id: ImageId):
    global shared_features
    image_id = image_id.image_id % len(image_loader)
    if image_id not in image_loader:
        logging.warning(f"Image with id {image_id} not found.")
        return {"message": "Image not found"}

    amplitude_image, phase_image = image_loader.get_images(image_id)

    amplitude_image_str = prepare_amplitude_img(amplitude_image)
    phase_image_str = prepare_phase_img(phase_image)

    if segmentation_method in ["fastsam", "sam"]:
        image_to_be_segmented = amplitude_image_str if image_to_segment == "amplitude" else phase_image_str
    else:
        image_to_be_segmented = amplitude_image if image_to_segment == "amplitude" else phase_image

    try:
        masks = image_segmentator.segment_image(image_to_be_segmented)
        contours = [segmentation_utils.get_mask_contour(m) for m in masks]
        contours = [segmentation_utils.normalize_contour(c) for c in contours]
        contours = segmentation_utils.flatten_contours(contours)
        logging.info(f"Masks of image {image_id} calculated succesfully.")
    except Exception as e:
        logging.error(f"Error while segmenting image with id {image_id}: {e}")
        contours = []
        masks = []
    logging.info(f"Found {len(masks)} masks in image with id {image_id}")
    try:
        features = feature_extractor.extract_features(phase_image, amplitude_image, masks)
        shared_features = features
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
            class_id=labels[idx],
            confidence=probabilities[idx],
        )
        for idx, polygon in enumerate(contours)
    ]

    logging.info(f"Sending image with id {image_id} and {len(predictions)} predictions to client.")

    amplitude_image_b64 = encode_b64(amplitude_image_str)
    phase_img_b64 = encode_b64(phase_image_str)

    return ImagesWithPredictions(
        amplitude_img_data=amplitude_image_b64,
        phase_img_data=phase_img_b64,
        predictions=predictions,
    )


@app.post("/process_lists")
def get_lists(lists: List[ListOfLists]):
    masks_pre = []
    # Process the received arrays
    for list in lists:
        # Access the NumPy array using array.data
        shape_coordinates = np.array(list.coordinates)
        image_shape = (384, 512)
        msk = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(msk, [shape_coordinates], 0, 255, -1)
        masks_pre.append(msk)

    logging.info("Masks created successfully")
    return {"message": "Masks created successfully"}



@app.post("/process_predictions")
async def process_strings_endpoint(predictions: PredictionsList):
    global shared_features

    predictions = predictions.predictions
    predictions_enc = np.array([string.encode('UTF-8') for string in predictions])

    # Load saved training data and concatenate with the new data
    file_path = os.path.join('classification/training_data', 'training_data.csv')
    new_df = pd.read_csv(file_path)

    y_saved = new_df['Labels'].str[2:-1].values
    y_saved = np.array([item.encode() for item in y_saved])
    X_saved = new_df.drop(['Labels'], axis=1)

    shared_features = shared_features.drop(["MaskID"], axis = 1)

    X_updated= pd.concat([X_saved, shared_features], axis=0)
    y_updated = np.concatenate((y_saved, predictions_enc))

    # Active learning
    classifier._active_learning(X_updated, y_updated)
    shared_features = None

    # Save the DataFrame to a CSV file inside the folder
    y_updated = y_updated.tolist()
    y_updated = [f"b'{item.decode()}'" for item in y_updated]
    X_updated['Labels'] = y_updated
    X_updated.to_csv(file_path, index=False)
    logging.info("Training data updated succesfully")

    logging.info("Predictions processed succesfully")
    return {"message": "Predictions processed succesfully"}
