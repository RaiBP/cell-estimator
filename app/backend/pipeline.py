from segmentation.threshold_segmentator import ThresholdImageSegmentator
from segmentation.fastsam_segmentator import FastSAMImageSegmentator
#from segmentation.mobile_sam_segmentator import MobileSAMImageSegmentator
from segmentation.sam_segmentator import SAMImageSegmentator
from segmentation.cellpose_segmentator import CellPoseImageSegmentator

config = {
    'image_segmentator': {
        'class': ThresholdImageSegmentator,
        # Put other parameters here in the future
    },
}
