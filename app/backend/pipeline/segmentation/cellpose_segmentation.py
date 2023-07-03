from segmentation import Segmentation
from cellpose import utils, models

class CellposeSegmentation(Segmentation):
    def __init__(self, model_type="cyto", use_gpu=True, diameter=None, channels=[0,0], flow_threshold=0.4, do_3D=False, cellprob_threshold=0.0, min_size=-1, augment=True, net_avg=True, resample=True, use_phase = True):

        self.model = models.Cellpose(gpu=use_gpu, model_type=model_type)
        self.diameter = diameter
        self.channels = channels
        self.flow_threshold = flow_threshold
        self.do_3D = do_3D
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size
        self.augment = augment
        self.net_avg = net_avg
        self.resample = resample
        self.use_phase = use_phase


    def _segment_single_image(self, phase, amplitude):
        img = phase if self.use_phase else amplitude

        masks_img, _, _, _ = self.model.eval(img, diameter=self.diameter, channels=self.channels,
                                         flow_threshold=self.flow_threshold, do_3D=self.do_3D, cellprob_threshold=self.cellprob_threshold, min_size=self.min_size, 
                                         augment=self.augment, net_avg=self.net_avg, resample=self.resample)
        return masks_img
    

    def _list_of_outlines(self, masks):
        return utils.outlines_list(masks) 
