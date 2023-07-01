import h5py
import numpy as np
import matplotlib.colors as clrs
import colorsys
import base64
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO

ColorNormalizer = plt.Normalize()

CellfaceStdNorm = clrs.Normalize(vmin=-4, vmax=14, clip=True)

CellfaceStdCMap = clrs.LinearSegmentedColormap.from_list(
    "CellFace Standard",
    [
        # Position               R     G     B     A
        (CellfaceStdNorm(-4.0), [0.65, 0.93, 1.00, 1.0]),  # Air bubbles
        (CellfaceStdNorm(0.0), [1.00, 0.97, 0.96, 1.0]),  # Background
    ]
    + [
        (
            CellfaceStdNorm(2 + p * (14 - 2)),
            [
                max(min(val, 1.0), 0.0)
                for val in list(
                    colorsys.hsv_to_rgb(
                        (280 - 90 * p) / 360,  # Hue: From Pink to Purple
                        0.5 + 1 * p,  # Saturation: Pastel to fully saturated
                        (1 - p) ** 2,  # Value: From Bright to Black
                    )
                )
            ]
            + [1.0],
        )
        for p in np.linspace(0.0, 1.0, 20)
    ],
)


def denormalize(img: np.array) -> np.array:
    """
    Denormalizes an image.

    Args:
        img: The image to be denormalized.
    Returns:
        The denormalized image.
    """
    return (img * 255).astype(np.uint8)


def encode_b64(img: np.array) -> str:
    """
    Encodes an image as a base64 string.

    Args:
        img: The image to be encoded.
    Returns:
        The base64 encoded image.
    """
    img = Image.fromarray(img)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def prepare_amplitude_img(img: np.array) -> str:
    """
    Encodes an amplitude image as a base64 string.

    Args:
        img: The amplitude image to be encoded.
    Returns:
        The base64 encoded amplitude image.
    """
    img = img.clip(0, 255)
    img = ColorNormalizer(img)
    img = denormalize(img)
    img = encode_b64(img)
    return img


def prepare_phase_img(img: np.array) -> str:
    """
    Applies the CellFace Standard colormap to a phase image, denormalizes it and encodes it as a base64 string.

    Args:
        img: The phase image to be prepared.
    Returns:
        The prepared phase image.
    """
    img = (img - img.min()) / (img.max() - img.min())
    img = CellfaceStdCMap(img)[..., :3]
    img = denormalize(img)
    img = encode_b64(img)
    return img


class ImageLoader:
    def __init__(self, path):
        self._file = h5py.File(path, "r")
        self._amplitude_images = self._file["amplitude/images"]
        self._phase_images = self._file["phase/images"]

    def get_amplitude_image(self, index):
        return self._amplitude_images[index]

    def get_phase_image(self, index):
        return self._phase_images[index]

    def get_images(self, index):
        return self.get_amplitude_image(index), self.get_phase_image(index)

    def __contains__(self, index):
        return index < len(self._amplitude_images)

    def __len__(self):
        return len(self._amplitude_images)

    @classmethod
    def from_file(cls, path):
        return cls(path)

# path = "/home/fidelinus/tum/applied_machine_intelligence/final_project/data/real_world_sample01.pre"
# loader = ImageLoader.from_file(path)
# _, phase = loader.get_images(0)
# cmapped = colormap_phase_img(phase)
# cmapped_denorm = denormalize(cmapped)

# plt.subplots(1, 3, figsize=(10, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(phase, cmap=CellfaceStdCMap)
# plt.subplot(1, 3, 2)
# plt.imshow(cmapped)
# plt.subplot(1, 3, 3)
# plt.imshow(cmapped_denorm)
