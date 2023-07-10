import logging
import os
from feature_extraction.feature_extractor import FeatureExtractor
from fastapi import FastAPI
from fastapi.responses import FileResponse
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
from backend import PipelineManager

from pprint import pprint
from segmentation.utils import create_segmentation_model, list_segmentation_methods

from image_loader import (
    ImageLoader,
    prepare_phase_img,
    prepare_amplitude_img,
    encode_b64,
)

# Setting up logger
logging.basicConfig(level=logging.INFO)

# Initialization values. All of these can be latter changed via POST methods
user_data_folder = Path(os.environ["USER_DATA_FOLDER"])
user_dataset = "user.pre"
user_dataset_path = user_data_folder / user_dataset

# Initializing image loader for dataset
logging.info("Initializing image loader.")
data_folder = Path(os.environ["DATA_FOLDER"])
#data_folder = Path("/home/fidelinus/tum/applied_machine_intelligence/final_project/data")
dataset = "real_world_sample01.pre"
dataset_path = data_folder / dataset
# data_folder = Path(os.environ["DATA_FOLDER"])
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

img_dims = image_loader.get_image_dimensions()
manager = PipelineManager(dataset, segmentation_method, classification_method, user_dataset_path, img_dims)


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

class DataForPlotting(BaseModel):
    features_names: List[str]
    feature_1_values: List[float]
    feature_2_values: List[float]
    cell_types: List[str]
    feature_1_training_values: List[float]
    feature_2_training_values: List[float]
    cell_types_training: List[str]

class PredictionsList(BaseModel):
    predictions: List[str]

class FeaturesList(BaseModel):
    features: List[str]


class ListsOfCoordinates(BaseModel):
    coordinates: List[List[int]]


app = FastAPI()

# Global help variables
shared_features = None
features_to_plot = None
corrected_predictions = None

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
    try:
        logging.info("Initializing image loader with new dataset.")
        dataset_path = data_folder / dataset_filename
        image_loader = ImageLoader.from_file(dataset_path)
        # save backend state
        manager.img_dims = image_loader.get_image_dimensions()
        manager.dataset_id = dataset_filename
        logging.info(f"Image loader initialized with {len(image_loader)} images.")
    except Exception as e:
        logging.error(f"Could not read dataset with filename {dataset_filename}: {e}")
        return {'message': "Dataset was not changed due to error"}
 
    return DatasetInfo(file=dataset_path.name, num_images=len(image_loader))


@app.post("/select_segmentator")
async def select_segmentator(segmentation_method: SegmentationMethod):
    """
    Method for initializing a new segmentator of type indicated by 'segmentation_method'
    """
    global image_segmentator
    try:
        logging.info(f"Initializing new segmentator of type {segmentation_method}.")
        image_segmentator = create_segmentation_model(segmentation_method)
        message = f"New segmentator of type {segmentation_method} initialized."
        # save backend state
        manager.segmentation_method = segmentation_method
        logging.info(message)
    except Exception as e:
        logging.error(f"Could not initialize segmentator of type {segmentation_method}: {e}")
        return {'message': "Segmentator was not changed due to error"}
    return {'message': message}


@app.post("/select_classifier")
async def select_classifier(classification_method: str):
    """
    Method for initializing a new classifier of type indicated by 'classification_method'
    """
    global classifier
    try:
        logging.info(f"Initializing new classifier of type {classification_method}.")
        classifier = create_classification_model(classification_method)
        message = f"New classifier of type {classification_method} initialized."
        # save backend state
        manager.classification_method = classification_method
        logging.info(message)
    except Exception as e:
        logging.error(f"Could not initialize classifier of type {classification_method}: {e}")
        return {'message': "Classifier was not changed due to error"}
    return {'message': message}

@app.get("/get_segmentation_methods")
async def get_segmentation_methods():
    return {"segmentation_methods": list_segmentation_methods()}


@app.post("/images")
async def get_images(image_query: ImageQuery):
    global shared_features
    image_id = image_query.image_id
    image_type = image_query.image_type

    image_id = image_id % len(image_loader)
    image_id = 2

    if image_id not in image_loader:
        logging.warning(f"Image with id {image_id} not found.")
        return {"message": "Image not found"}

    manager.image_id = image_id

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
        shared_features = features
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

    manager.save_masks(masks, labels)
    manager.cell_counter += len(masks)
    manager.image_counter += 1

    return ImagesWithPredictions(
        amplitude_img_data=amplitude_image_b64,
        phase_img_data=phase_img_b64,
        predictions=predictions,
    )


@app.get('/download_masks_and_labels')
async def download_masks_and_labels_route():
    return FileResponse(user_dataset_path, media_type='application/octet-stream', filename=user_dataset)


@app.post("/process_lists_of_coordinates")
def get_lists_of_coordinates(lists: List[ListsOfCoordinates]):
    """
    Method for converting the Polygon's coordinates to masks
    """
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


@app.post("/process_corrected_predictions")
async def get_corrected_predictions(predictions: PredictionsList):
    """
    Method for performing active learning based on the user-corrected predictions
    """
    global shared_features
    global corrected_predictions

    corrected_predictions = predictions.predictions
    predictions_enc = np.array([string.encode('UTF-8') for string in corrected_predictions])

    # Load saved training data and concatenate with the new data
    file_path = os.path.join('classification/training_data', 'training_data.csv')
    new_df = pd.read_csv(file_path)

    y_saved = new_df['Labels'].str[2:-1].values
    y_saved = np.array([item.encode() for item in y_saved])
    X_saved = new_df.drop(['Labels'], axis=1)

    shared_features = shared_features.drop(["MaskID"], axis = 1)

    X_updated= pd.concat([X_saved, shared_features], axis=0)
    y_updated = np.concatenate((y_saved, predictions_enc))

    # Active learning is performed here
    classifier._active_learning(X_updated, y_updated)

    # Save the DataFrame to a CSV file inside the folder
    y_updated = y_updated.tolist()
    y_updated = [f"b'{item.decode()}'" for item in y_updated]
    X_updated['Labels'] = y_updated
    X_updated.to_csv(file_path, index=False)
    logging.info("Training data updated succesfully")

    logging.info("Predictions processed succesfully")
    return {"message": "Predictions processed succesfully"}


@app.post("/features_and_data_to_plot")
async def get_features_and_data_to_plot(features: FeaturesList):
    """
    Method for saving the features that will be used for plotting
    and sending the data that will be plotted
    """
    global shared_features
    global corrected_predictions


    if features is not None:
        features_to_plot = features.features

    file_path = os.path.join('classification/training_data', 'training_data.csv')
    training_features = pd.read_csv(file_path)

    feature_1_values = shared_features[features_to_plot[0]].tolist()
    feature_2_values = shared_features[features_to_plot[1]].tolist()
    cell_types = corrected_predictions

    feature_1_training_values = training_features[features_to_plot[0]].tolist()
    feature_2_training_values = training_features[features_to_plot[1]].tolist()
    cell_types_training = training_features['Labels'].str[2:-1].tolist()

    return DataForPlotting(features_names=features_to_plot, 
                           feature_1_values=feature_1_values,
                           feature_2_values=feature_2_values,
                           cell_types = cell_types,
                           feature_1_training_values=feature_1_training_values,
                           feature_2_training_values=feature_2_training_values,
                           cell_types_training = cell_types_training)
