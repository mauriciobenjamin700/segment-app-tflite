import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tflite_runtime.interpreter import Interpreter

from src.backend.utils.pre_process import nms
from src.backend.utils.read_files import load_labels


class TFLiteHandler:
    def __init__(self, model_path: str, labels_path: str) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.interpreter = Interpreter(model_path=model_path)
        self.labels = load_labels(labels_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.image = None

    def get_model_input_shape(self) -> list[int]:
        return self.input_details[0]["shape"]

    def get_model_input_width(self) -> int:
        return self.input_details[0]["shape"][1]

    def get_model_input_height(self) -> int:
        return self.input_details[0]["shape"][2]

    def load_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize(
            (self.get_model_input_width(), self.get_model_input_height())
        )
        self.image = np.array(image)
        return self.image

    def map_image_to_input_tensor(self, image: np.ndarray) -> np.ndarray:
        input_data = np.expand_dims(image, axis=0).astype(np.float32)
        if self.input_details[0]["dtype"] == np.float32:
            input_data = (input_data - 127.5) / 127.5
        return input_data

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )
        return output_data

    def process_yolo_output(self, output_data: np.ndarray):
        num_classes = len(self.labels)
        scores_start = (
            output_data.shape[1] - num_classes
        )  # Supondo que scores de classe estão no final
        output = output_data[0]  # shape: (116, 8400)

        best_score = -float("inf")
        best_class = None
        for anchor in range(output.shape[1]):
            class_scores = output[
                scores_start:, anchor
            ]  # shape: (num_classes,)
            objectness = output[
                4, anchor
            ]  # ajuste conforme seu modelo, normalmente posição 4
            class_idx = np.argmax(class_scores)
            score = class_scores[class_idx] * objectness  # score final
            if score > best_score:
                best_score = score
                best_class = class_idx
        if best_class is not None:
            print("Classe mais detectada:", self.labels[best_class])
            print("Score:", best_score)

            # Coletar boxes e scores para NMS
            boxes = []
            scores = []
            for anchor in range(output.shape[1]):
                objectness = output[4, anchor]
                if objectness > 0.5:
                    x, y, w, h = (
                        output[0, anchor],
                        output[1, anchor],
                        output[2, anchor],
                        output[3, anchor],
                    )
                    if (
                        0 <= x <= 1
                        and 0 <= y <= 1
                        and 0 <= w <= 1
                        and 0 <= h <= 1
                    ):
                        x *= self.get_model_input_width()
                        y *= self.get_model_input_height()
                        w *= self.get_model_input_width()
                        h *= self.get_model_input_height()
                    left = int(x - w / 2)
                    top = int(y - h / 2)
                    right = int(x + w / 2)
                    bottom = int(y + h / 2)
                    boxes.append([left, top, right, bottom])
                    scores.append(objectness)

            keep = nms(boxes, scores, iou_threshold=0.5)
            draw = ImageDraw.Draw(Image.fromarray(self.image))
            mask = np.zeros(
                (self.get_model_input_height(), self.get_model_input_width()),
                dtype=np.uint8,
            )

            for i in keep:
                left, top, right, bottom = boxes[i]
                print(
                    f"""Box NMS: 
                    left={left}, 
                    top={top}, 
                    right={right}, 
                    bottom={bottom}, 
                    score={scores[i]}
                    """
                )
                draw.rectangle(
                    [left, top, right, bottom], outline="red", width=2
                )
                # Preencher a máscara com 1 dentro da bounding box
                mask[
                    max(top, 0) : min(bottom, self.get_model_input_height()),
                    max(left, 0) : min(right, self.get_model_input_width()),
                ] = 1

            # Aplicar máscara: pixels fora das caixas ficam pretos
            img_np = np.array(self.image)
            img_np[mask == 0] = 0
            img_segmented = Image.fromarray(img_np)

            # Recortar zona de interesse detectada
            ys, xs = np.where(mask == 1)
            if len(xs) > 0 and len(ys) > 0:
                left_crop, right_crop = xs.min(), xs.max()
                top_crop, bottom_crop = ys.min(), ys.max()
                img_cropped = img_segmented.crop(
                    (left_crop, top_crop, right_crop, bottom_crop)
                )
                img_final = img_cropped.resize((640, 640), Image.LANCZOS)
            else:
                img_final = img_segmented.resize((640, 640), Image.LANCZOS)

            print(f"Total de bounding boxes após NMS: {len(keep)}")
            return img_final
        else:
            print("Nenhuma classe detectada com score significativo.")

    def plot_results(self, image: Image.Image) -> None:
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title("Segmentação recortada e redimensionada 640x640")
        plt.axis("off")
        plt.show()

    def __call__(self, image_path: str) -> np.ndarray:
        image = self.load_image(image_path)
        tensor = self.map_image_to_input_tensor(image)
        result = self.predict(tensor)
        segment = self.process_yolo_output(result)
        return segment
