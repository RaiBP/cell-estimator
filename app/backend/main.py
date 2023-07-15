import logging
import os
from re import A
from shutil import which
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

from segmentation.utils import create_segmentation_model, list_segmentation_methods
from classification.utils import list_classification_methods

from image_loader import (
    ImageLoader,
    prepare_phase_img,
    prepare_amplitude_img,
    encode_b64,
)

# Setting up logger
logging.basicConfig(level=logging.INFO)

training_data_folder = Path(os.environ["TRAINING_DATA_FOLDER"])
training_data_filename = "training_data_user.csv"
training_data_path = training_data_folder / training_data_filename

# Initialization values. All of these can be latter changed via POST methods
user_data_folder = Path(os.environ["USER_DATA_FOLDER"])
#user_data_folder = Path("/home/larintzos/Group06/notebooks/h5py_data")
user_dataset = "user_data.pre"
user_dataset_path = user_data_folder / user_dataset

# Initializing image loader for dataset
logging.info("Initializing image loader.")
data_folder = Path(os.environ["DATA_FOLDER"])
#data_folder = Path("/mnt/w")
dataset = "sample01.pre"
dataset_path = data_folder / dataset
image_loader = ImageLoader.from_file(dataset_path)
logging.info(f"Image loader initialized with {len(image_loader)} images.")

# Initializing image segmentator
segmentation_method = pipeline_config["image_segmentator"]["method"]
image_to_segment = pipeline_config["image_segmentator"]["image_to_segment"]

feature_extractor = FeatureExtractor()

classification_method = pipeline_config["classifier"]["method"]


logging.info("Initializing pipeline manager.")
manager = PipelineManager(logging, dataset_path, segmentation_method, classification_method, feature_extractor, user_dataset_path)

class ImageSegmentationMethod(str, Enum):
    cellpose = "cellpose"
    threshold = "threshold"
    fastsam = "fastsam"
    sam = "sam"

class CellClassificationMethod(str, Enum):
    svc = "svc"
    rfc = "rfc"
    knn = "knn"
    tsc = "tsc"

class Dataset(BaseModel):
    filename: str

class SegmentationMethod(BaseModel):
    method: ImageSegmentationMethod

class ClassificationMethod(BaseModel):
    method: CellClassificationMethod

class Polygon(BaseModel):
    points: List[float] | None


class PolygonWithPredictions(BaseModel):
    polygon: Polygon
    class_id: str
    features: dict

# I leave this class just to use the get_images method (which is deprecated)
class PolygonWithPredictions_LEGACY(BaseModel):
    polygon: Polygon
    class_id: str 
    confidence: dict
    features: dict

class ImagesWithPredictions_LEGACY(BaseModel):
    amplitude_img_data: str
    phase_img_data: str
    predictions: List[PolygonWithPredictions_LEGACY]

class Predictions(BaseModel):
    class_id: str
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

class ImagesWithPolygons(BaseModel):
    amplitude_img_data: str
    phase_img_data: str
    polygons: List[Polygon]

class ListOfPolygons(BaseModel):
    polygons: List[Polygon]


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

class Polygons(BaseModel):
    points: List[List[int]]


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

@app.get("/datasets")
async def get_datasets():
    return {
        # are we sure all the datasets will be in .pre file format?
        "datasets": [dataset.name for dataset in data_folder.glob("*.pre")],
    }

@app.get("/dataset_info")
async def get_dataset_info():
    return DatasetInfo(file=dataset_path.name, num_images=len(image_loader))


@app.post("/select_dataset")
async def select_dataset(dataset: Dataset):
    """
    Method for changing the dataset file from which to load the images
    """
    global manager, data_folder, logging
    try:
        logging.info("Initializing image loader with new dataset.")
        dataset_path = data_folder / dataset.filename
        # save backend state
        manager.set_dataset(dataset_path)
        num_imgs = len(manager.image_loader)
        logging.info(f"Image loader initialized with {num_imgs} images.")
    except Exception as e:
        logging.error(f"Could not read dataset with filename {dataset.filename}: {e}")
        return {'message': "Dataset was not changed due to error"} 
    return DatasetInfo(file=dataset_path.name, num_images=num_imgs)


@app.post("/select_segmentator")
async def select_segmentator(segmentation_method: SegmentationMethod):
    """
    Method for initializing a new segmentator of type indicated by 'segmentation_method'
    """
    global manager, logging
    try:
        manager.set_segmentation_method(segmentation_method.method)
    except Exception as e:
        logging.error(f"Could not initialize segmentator of type {segmentation_method}: {e}")
        return {'message': "Segmentator was not changed due to error"}
    return {'message': f"New segmentator of type {manager.get_current_segmentation_method()} initialized."}


@app.post("/select_classifier")
async def select_classifier(classification_method: ClassificationMethod):
    """
    Method for initializing a new classifier of type indicated by 'classification_method'
    """
    global manager, logging
    try:
        manager.set_classification_method(classification_method.method)
        message = f"New classifier of type {classification_method.method} initialized."
    except Exception as e:
        logging.error(f"Could not initialize classifier of type {classification_method.method}: {e}")
        return {'message': "Classifier was not changed due to error"}
    return {'message': message}


@app.get("/get_segmentation_methods")
async def get_segmentation_methods():
    return {"segmentation_methods": list_segmentation_methods()}


@app.get("/get_classification_methods")
async def get_classification_methods():
    return {"classification_methods": list_classification_methods()}


@app.post("/set_image")
async def set_image(image_query: ImageQuery):
    """
    Method to set an image as the current image in the backend manager. All subsequent operations will be 
    performed on this image until this method is called again and sets a different image.
    Note that this method looks for features, masks and labels that are already saved in the server and returns them 
    to the frontend
    """
    global manager, logging, training_data_path
    image_loader = manager.image_loader

    image_id = image_query.image_id
    image_id = image_id % len(image_loader)

    image_type = image_query.image_type

    if image_id not in image_loader:
        logging.warning(f"Image with id {image_id} not found.")
        return {"error": "Image not found"}
    else:
        manager.set_amplitude_phase_images(image_id)

        manager.set_image_id(image_id)
        manager.set_image_type(image_type)

        features = manager.get_saved_features(image_id, manager.get_dataset_id(), training_data_path)
        masks, labels = manager.get_saved_masks_and_labels(image_id, manager.get_dataset_id())
        
        manager.set_masks(masks)
        manager.set_shared_features(features)
        manager.set_predictions(labels)
        if features is not None:
            features_records = features.to_dict('records')
        else:
            features_records = {}
        logging.info (f"Image with id {image_id} from dataset {manager.dataset_id} is set as active image.") 


        contours = [segmentation_utils.get_mask_contour(m) for m in masks]
        contours = [segmentation_utils.normalize_contour(c) for c in contours]
        contours = segmentation_utils.flatten_contours(contours)

        amplitude_image_str, phase_image_str = manager.get_amplitude_phase_images_str()

        amplitude_image_b64 = encode_b64(amplitude_image_str)
        phase_img_b64 = encode_b64(phase_image_str)

        if masks is None or features is None or labels is None:
            return ImagesWithPredictions(
                amplitude_img_data=amplitude_image_b64,
                phase_img_data=phase_img_b64,
                predictions=[]
            )

        else:
            predictions = [
                    PolygonWithPredictions(
                        polygon=Polygon(points=polygon),
                        class_id=label,
                        features=mask_features
                    )
                    for polygon, label, mask_features in zip(contours, labels, features_records)
                ]

            
            return ImagesWithPredictions(
                amplitude_img_data=amplitude_image_b64,
                phase_img_data=phase_img_b64,
                predictions=predictions,
            )


@app.get("/segment")
async def segment():
    global manager, logging

    image_segmentator = manager.image_segmentator
    image_id = manager.image_id
    image_type = manager.image_type

    amplitude_image, phase_image = manager.get_amplitude_phase_images()
    amplitude_image_str, phase_image_str = manager.get_amplitude_phase_images_str()

    segmentation_method = manager.get_current_segmentation_method()

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
        manager.set_masks(masks)

    except Exception as e:
        logging.error(f"Error while segmenting image with id {image_id}: {e}")
        contours = []
        masks = []
        manager.set_masks([])

    logging.info(f"Found {len(masks)} masks in image with id {image_id}")

    polygons = [Polygon(points=polygon) for polygon in contours]

    logging.info(f"Sending image with id {image_id} and {len(polygons)} masks to client.")


    return ListOfPolygons(
        polygons=polygons
    )


@app.post("/classify")
async def classify(polygons: List[Polygon], use_backend_masks: bool):
    global manager, logging

    if use_backend_masks:
        try:
            masks = manager.get_masks()
            assert(masks is not None)
        except Exception as e:
            logging.error(f"No masks found in backend manager: {e}")
            logging.info("Using provided polygon list instead.")    
            masks = manager.get_masks_from_polygons(polygons)
            manager.set_masks(masks)
    else:
        masks = manager.get_masks_from_polygons(polygons)
        manager.set_masks(masks)

    amplitude_image, phase_image = manager.get_amplitude_phase_images()
    image_id = manager.image_id
    classifier = manager.classifier

    try:
        features = feature_extractor.extract_features(phase_image, amplitude_image, masks)
    except Exception as e:
        logging.error(f"Error while extracting features from image with id {image_id}: {e}")
        features = None
        features_records = {}
    if features is not None:
        try:
            labels, probabilities = classifier.classify(features)
            entropies = classifier.calculate_entropy(labels, probabilities)
            proba_per_label = classifier.calculate_probability_per_label(labels, probabilities)
            features_records = features.to_dict('records')
            features['LabelsEntropy'] = entropies
            features = classifier.add_class_probabilities_columns(features, proba_per_label)

            manager.set_shared_features(features)
            manager.set_predictions(labels)
        except Exception as e:
            logging.error(f"Error while classifying image with id {image_id}: {e}")
            labels = []
            features_records = {}
            manager.set_shared_features(None)
            manager.set_predictions(None)
    else:
        labels = []
        features_records = {}
        manager.set_shared_features(None)
        manager.set_predictions(None)

    if (not features_records) or (not labels):
        # if any of these is empty, then predictions is also empty
        predictions = []
    else:
        predictions = [
            Predictions(
                class_id=label,
                features=mask_features
            )
            for label, mask_features in zip(labels, features_records)]

    logging.info(f"Sending {len(predictions)} predictions to client for image with id {image_id}.")
    return predictions


@app.post("/save_masks_and_labels")
async def save_masks_and_labels(predictions: List[str]):
    """
    Method to save the masks and labels once the user has confirmed that they are correct.
    They are saved to a h5 file found in 'user_dataset_path' (defined at the start of this file).
    Note that the classify method must have been called before this one, in order for there to be 
    masks saved in the manager. The classify method basically "saves" the masks to the manager.
    i.e. if the user changes the masks after classifying, he will have to call the classify method again.
    """
    global manager, logging, user_dataset, training_data_path
    masks = manager.get_masks()
    try: 
        assert masks is not None
    except:
        logging.error(f"Error: No masks found in the backend manager")
        return {"error": "Masks not found."}

    try:
        assert len(masks) == len(predictions)
    except:
        logging.error(f"Error: given list of labels do not coincide in length with given masks")
        return {"error": "Masks and labels are not equal in number"}

    image_id = manager.image_id
    manager.set_predictions(predictions)

    try:
        manager.save_masks(masks, predictions)
        logging.info(f"Masks and labels saved as {user_dataset}")
    except Exception as e:
        logging.error(f"Error while saving masks and labels of image with id {image_id}: {e}")
        return {"error": f"Masks and labels of image with id {image_id} could not be saved"}

    try:
        # the backend has the features from all the masks
        features = manager.get_shared_features()

        # for this to pass, we must have the features saved in the manager beforehand
        # the features will be in the manager either because they were saved there
        # during the classify method or during the set_image method
        assert features is not None
    
        # we don't save the mask IDs, as they are irrelevant for the classification models
        features = features.drop(["MaskID"], axis = 1)

        dataset_id = manager.get_dataset_id()

        training_df = pd.read_csv(training_data_path)

        is_match_present = (training_df['DatasetID'] == dataset_id) & (training_df['ImageID'] == image_id)
        if any(is_match_present):
            # if we already have features by the image ID, we delete those and write the new ones
            training_df = training_df[~is_match_present]

        features['DatasetID'] = dataset_id
        features['ImageID'] = image_id
        features['Labels'] = predictions

        new_training_df = pd.concat([training_df, features], ignore_index=True)

        new_training_df.to_csv(training_data_path, index=False)
        logging.info(f"Training data updated succesfully and saved as {training_data_path.name}")
    except Exception as e:
        logging.error(f"Error while saving training data of image with id {image_id}: {e}")
    return {"message": "User data has been saved"}
    

@app.post("/images")
async def get_images(image_query: ImageQuery):
    global manager, logging

    image_loader = manager.image_loader
    image_segmentator = manager.image_segmentator
    classifier = manager.classifier

    image_id = image_query.image_id
    image_type = image_query.image_type

    image_id = image_id % len(image_loader)

    if image_id not in image_loader:
        logging.warning(f"Image with id {image_id} not found.")
        return {"message": "Image not found"}

    manager.set_image_id(image_id)

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
        manager.set_shared_features(features)
        # shared_features = features
        features_records = features.to_dict('records')
    except Exception as e:
        logging.error(f"Error while extracting features from image with id {image_id}: {e}")
        features = None
        features_records = {}
    try:
        labels, probabilities = classifier.classify(features)
    except Exception as e:
        logging.error(f"Error while classifying image with id {image_id}: {e}")
        probabilities = []
        labels = []


    predictions = [
        PolygonWithPredictions_LEGACY(
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

    return ImagesWithPredictions_LEGACY(
        amplitude_img_data=amplitude_image_b64,
        phase_img_data=phase_img_b64,
        predictions=predictions,
    )


@app.get('/download_masks_and_labels')
async def download_masks_and_labels_route():
    return FileResponse(user_dataset_path, media_type='application/octet-stream', filename=user_dataset)


@app.post("/receive_predictions")
async def receive_predictions(predictions_list: PredictionsList):
    """
    Method for receiving predictions from frontend and saving them in the PipelineManager
    """
    global manager
    predictions = predictions_list.predictions
    manager.set_predictions(predictions)
    logging.info("Predictions saved succesfully")
    return {"message": "Predictions saved succesfully"}


@app.get("/retrain_model")
async def retrain_model():
    """
    Method for performing active learning based on the user-corrected predictions
    Training data is taken from csv in 'training_data_path'. To add new data, please
    use method 'save_masks_and_labels'
    """
    global manager, training_data_path
    try:
        # Load saved training data and concatenate with the new data
        training_data = pd.read_csv(training_data_path)
        classification_method = manager.get_current_classification_model()
        
        if classification_method == 'tsc':
            y = training_data['Labels'].str.strip("b'")
        else:
            y = training_data['Labels'].str[2:-1].values
            y = np.array([item.encode() for item in y])

        X = training_data.drop('Labels', axis=1)

        model_filename = f"{manager.get_current_classification_model()}_user_model.pkl"

        # Active learning
        manager.classifier.fit(X, y, model_filename=model_filename)
        logging.info(f"Model retrained succesfully on {manager.cell_counter} data points and saved as {model_filename}")
    except Exception as e:
        logging.error(f"Error while retraining model: {e}")
        return {"error": "Error while retraining model"}
    return {"message": "Model retrained succesfully"}


@app.post("/features_and_data_to_plot")
async def get_features_and_data_to_plot(features: FeaturesList):
    """
    Method for saving the features that will be used for plotting
    and sending the data that will be plotted
    IMPORTANT: the POST method "receive_predictions" must be called first
    """
    if features is not None:
        features_to_plot = features.features

    file_path = os.path.join('classification/data', 'training_data_base.csv')
    training_features = pd.read_csv(file_path)
    
    shared_features = manager.get_shared_features()
    corrected_predictions = manager.get_predictions()

    assert shared_features is not None and corrected_predictions is not None

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
