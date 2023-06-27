from abc import ABC, abstractmethod
import numpy as np

class Segmentation(ABC):
    def __init__(self, phase_array, amplitude_array):
        # assert that there is the same number of phase and amplitude images
        assert(np.shape(amplitude_array)[0] == np.shape(phase_array)[0])

        self.batch = list(zip(phase_array, amplitude_array))
        self.masks = None


    def segment(self):
        self.masks = []
        for phase, amplitude in self.batch:
            mask_single_image = self._segment_single_image(phase, amplitude)
            self.masks.append(mask_single_image)
        return self.masks

    @abstractmethod
    def _segment_single_image(self, index, phase, amplitude):
        pass
