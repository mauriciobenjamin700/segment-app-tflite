from PIL import Image

from src.backend.services import SegmentService


class SegmentController:
    def __init__(self) -> None:
        self.__segment_service = SegmentService()

    def segment(self, image_path: str, plot: bool = True) -> Image.Image | None:
        result = self.__segment_service.segment(image_path)
        if plot and result is not None:
            self.__segment_service.plot_segment(result)
        return result
