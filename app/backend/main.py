import logging
import os
from feature_extraction.feature_extractor import FeatureExtractor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
import cv2

from segmentation import utils as segmentation_utils
from pipeline import config as pipeline_config 
from classification.utils import create_classification_model
from segmentation.utils import create_segmentation_model, list_segmentation_methods

from image_loader import (
    ImageLoader,
    prepare_phase_img,
    prepare_amplitude_img,
    encode_b64,
)


class PipelineManager:
    def __init__(self, dataset_path: Path, segmentation_method: str, classification_method: str, feature_extractor: FeatureExtractor):
        self.set_dataset(dataset_path)
        self.set_segmentation_method(segmentation_method)
        self.set_classification_method(classification_method)
        self.set_feature_extractor(feature_extractor)
        self.shared_features = None

    def set_dataset(self, dataset_path: Path):
        logging.info("Initializing image loader with new dataset.")
        self.image_loader = ImageLoader.from_file(dataset_path)
        logging.info(f"Image loader initialized with {len(self.image_loader)} images.")

    def set_segmentation_method(self, segmentation_method: str):
        logging.info(f"Initializing new segmentator of type {segmentation_method}.")
        self.image_segmentator = create_segmentation_model(segmentation_method)
        logging.info(f"New segmentator of type {segmentation_method} initialized.")

    def set_classification_method(self, classification_method: str):
        logging.info(f"Initializing new classifier of type {classification_method}.")
        self.classifier = create_classification_model(classification_method)
        logging.info(f"New classifier of type {classification_method} initialized.")

    def set_feature_extractor(self, feature_extractor: FeatureExtractor):
        logging.info(f"Initializing new feature extractor of type {feature_extractor}.")
        self.feature_extractor = feature_extractor
        logging.info(f"New feature extractor of type {feature_extractor} initialized.")

    def set_shared_features(self, shared_features: pd.DataFrame):
        self.shared_features = shared_features

    def get_current_segmentation_method(self) -> str:
        return self.image_segmentator.name()

# Setting up logger
logging.basicConfig(level=logging.INFO)

logging.info("Initializing image loader.")
# data_folder = Path(os.environ["DATA_FOLDER"])
data_folder = Path("/home/fidelinus/tum/applied_machine_intelligence/final_project/data")
dataset_path = data_folder / "real_world_sample01.pre"

# Initializing image segmentator
segmentation_method = pipeline_config["image_segmentator"]["method"]
image_to_segment = pipeline_config["image_segmentator"]["image_to_segment"]

feature_extractor = FeatureExtractor()

classification_method = pipeline_config["classifier"]["method"]

logging.info("Initializing pipeline manager.")
pipeline_manager = PipelineManager(dataset_path, segmentation_method, classification_method, feature_extractor)

class ImageSegmentationMethod(str, Enum):
    cellpose = "cellpose"
    threshold = "threshold"
    fastsam = "fastsam"
    sam = "sam"

class SegmentationMethod(BaseModel):
    method: ImageSegmentationMethod

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
    logging.info("Initializing image loader with new dataset.")
    dataset_path = data_folder / dataset_filename
    pipeline_manager.set_dataset(dataset_path)
    return DatasetInfo(file=dataset_path.name, num_images=len(pipeline_manager.image_loader))


@app.post("/select_segmentator")
async def select_segmentator(segmentation_method: SegmentationMethod):
    """
    Method for initializing a new segmentator of type indicated by 'segmentation_method'
    """
    pipeline_manager.set_segmentation_method(segmentation_method.method)
    message = f"New segmentator of type {pipeline_manager.get_current_segmentation_method()} initialized."
    return {'message': message}


@app.post("/select_classifier")
async def select_classifier(classification_method: str):
    """
    Method for initializing a new classifier of type indicated by 'classification_method'
    """
    pipeline_manager.set_classification_method(classification_method)
    return {'message': message}

@app.get("/get_segmentation_methods")
async def get_segmentation_methods():
    return {"segmentation_methods": list_segmentation_methods()}


@app.post("/images")
async def get_images(image_query: ImageQuery):

    image_loader = pipeline_manager.image_loader
    image_segmentator = pipeline_manager.image_segmentator
    classifier = pipeline_manager.classifier

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
        logging.info(f"Masks of image {image_id} calculated succesfully.")
    except Exception as e:
        logging.error(f"Error while segmenting image with id {image_id}: {e}")
        contours = []
        masks = []
    logging.info(f"Found {len(masks)} masks in image with id {image_id}")
    try:
        features = feature_extractor.extract_features(phase_image, amplitude_image, masks)
        pipeline_manager.set_shared_features(features)
        # shared_features = features
        features_records = features.to_dict('records')
    except Exception as e:
        logging.error(f"Error while extracting features from image with id {image_id}: {e}")
        features = None
        features_records = {}
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

    predictions = predictions.predictions
    predictions_enc = np.array([string.encode('UTF-8') for string in predictions])

    # Load saved training data and concatenate with the new data
    file_path = os.path.join('classification/training_data', 'training_data.csv')
    new_df = pd.read_csv(file_path)

    y_saved = new_df['Labels'].str[2:-1].values
    y_saved = np.array([item.encode() for item in y_saved])
    X_saved = new_df.drop(['Labels'], axis=1)

    pipeline_manager.set_shared_features(pipeline_manager.shared_features.drop(["MaskID"], axis = 1))

    X_updated= pd.concat([X_saved, pipeline_manager.shared_features], axis=0)
    y_updated = np.concatenate((y_saved, predictions_enc))

    # Active learning
    pipeline_manager.classifier._active_learning(X_updated, y_updated)
    pipeline_manager.set_shared_features(None)

    # Save the DataFrame to a CSV file inside the folder
    y_updated = y_updated.tolist()
    y_updated = [f"b'{item.decode()}'" for item in y_updated]
    X_updated['Labels'] = y_updated
    X_updated.to_csv(file_path, index=False)
    logging.info("Training data updated succesfully")

    logging.info("Predictions processed succesfully")
    return {"message": "Predictions processed succesfully"}
