from PIL import Image

from src.backend.core.constants import TFLITE_LABELS_PATH, TFLITE_MODEL_PATH
from src.backend.core.handlers import TFLiteHandler


class SegmentService:
    def __init__(self) -> None:
        self.tflite_handler = TFLiteHandler(
            TFLITE_MODEL_PATH, TFLITE_LABELS_PATH
        )

    def segment(self, image_path: str) -> Image.Image | None:
        return self.tflite_handler(image_path)

    def plot_segment(self, image: Image.Image) -> None:
        self.tflite_handler.plot_results(image)
