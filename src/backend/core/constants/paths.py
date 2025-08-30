from os.path import abspath, dirname, join

BASE_DIR = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))

SRC_DIR = join(BASE_DIR, "src")
TFLITE_MODEL_PATH = join(
    SRC_DIR, "backend", "assets", "yolo11n-seg_float16.tflite"
)
TFLITE_LABELS_PATH = join(SRC_DIR, "backend", "assets", "labels.txt")
