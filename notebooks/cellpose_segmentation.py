from segmentation import Segmentation
from cellpose import utils

class CellposeSegmentation(Segmentation):
    def __init__(self, phase_array, amplitude_array, model, diameter=None, channels=[0,0], flow_threshold=0.4, do_3D=False, cellprob_threshold=-1.5, use_phase = True):
        super().__init__(phase_array, amplitude_array)
        self.model = model
        self.diameter = diameter
        self.channels = channels
        self.flow_threshold = flow_threshold
        self.do_3D = do_3D
        self.cellprob_threshold = cellprob_threshold
        self.use_phase = use_phase

    def _segment_single_image(self, phase, amplitude):
        # Implement Cellpose-based segmentation here
        # Use self.img and self.model to perform the segmentation
        # Return the segmented image

        imgs = phase if self.use_phase else amplitude

        masks_imgs, flows_imgs, styles_imgs, diams_imgs = self.model.eval(imgs, diameter=self.diameter, channels=self.channels,
                                                flow_threshold=self.flow_threshold, do_3D=self.do_3D, cellprob_threshold=self.cellprob_threshold)
        return masks_imgs
    

    def _list_of_outlines(self, masks):
        return utils.outlines_list(masks)
    