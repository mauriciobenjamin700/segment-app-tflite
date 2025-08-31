from src.backend.controllers import SegmentController

image_path = "image.png"

segment_controller = SegmentController()
segment = segment_controller.segment(image_path)

