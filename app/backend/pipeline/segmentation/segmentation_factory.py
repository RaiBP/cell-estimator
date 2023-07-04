from threshold_segmentation import ThresholdSegmentation
from sam_segmentation import SAMSegmentation
from cellpose_segmentation import CellposeSegmentation


class SegmentationFactory:
    def __init__(self):
        pass 


    @staticmethod
    def create_model(selector):
        if selector == "cellpose":
            return CellposeSegmentation()   
        elif selector == "threshold":
            return ThresholdSegmentation()
        elif selector == "sam":
            return SAMSegmentation()
        else:
            raise ValueError("Invalid segmentation model")

