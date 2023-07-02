from feature_extraction import * 
from cellpose_segmentation import CellposeSegmentation
from threshold_segmentation import ThresholdSegmentation
from sam_segmentation import SAMSegmentation
from cellpose import models
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class Pipeline:
    def __init__(self, phase_img, amplitude_img, segmentation_algorithm, classification_model):
        self.phase_img = phase_img
        self.amplitude_img = amplitude_img
        self.segmentation_algorithm = segmentation_algorithm
        self.classification_model = classification_model

    def process_data(self):
        # Segmentation
        if self.segmentation_algorithm == 'cellpose':
            # model_type='cyto' or model_type='nuclei'
            seg_model = models.Cellpose(gpu=False, model_type='cyto')
            seg = CellposeSegmentation(self.phase_img, self.amplitude_img, seg_model)
        elif self.segmentation_algorithm == 'thresholding':
            seg = ThresholdSegmentation(self.phase_img, self.amplitude_img)
        elif self.segmentation_algorithm == 'sam':
            sam_checkpoint = "sam_vit_b_01ec64.pth"
            model_type = "vit_b"
            device = "cuda"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            seg_model = SamAutomaticMaskGenerator(sam)
            seg = SAMSegmentation(self.phase_img, self.amplitude_img, seg_model)
 
        masks = seg.segment()
        
        # probably here need an if statement
        masks_array = []
        for idx, _ in enumerate(self.phase_img):
            masks1 = image_to_masks(masks[idx])
            masks_array.append(masks1)

        # Feature extraction
        fe = FeatureExtractor(self.phase_img, self.amplitude_img, masks_array)
        extracted_features = fe.extract_features_multiple_masks()

        # Classification
        if self.classification_model == 'SVC':
            class_model = load_model("models", "best_svc_model.pkl")
        elif self.classification_model == 'RFC':
            class_model = load_model("models", "best_rfc_model.pkl")
        elif self.classification_model == 'KNN':
            class_model = load_model("models", "best_knn_model.pkl")
        elif self.classification_model == 'NB':
            class_model = load_model("models", "best_nb_model.pkl")
                
        classifier = Classification(extracted_features, class_model)

        y_pred, prob = classifier.classify()

        return y_pred, prob
