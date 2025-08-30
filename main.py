from src.backend.core.handlers import TFLiteHandler
from src.backend.core.constants import TFLITE_MODEL_PATH, TFLITE_LABELS_PATH


model = TFLiteHandler(model_path=TFLITE_MODEL_PATH, labels_path=TFLITE_LABELS_PATH)

image_path = "image.png"

segment = model(image_path)

model.plot_results(segment)
