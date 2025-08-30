import numpy as np
from PIL import Image, ImageDraw
from tflite_runtime.interpreter import Interpreter
import matplotlib.pyplot as plt

# Caminhos dos arquivos
IMAGE_PATH = "image.png"
MODEL_PATH = "./src/backend/assets/yolo11n-seg_float16.tflite"
LABEL_PATH = "labels.txt"

def load_labels(filename):
    with open(filename, 'r') as f:
        result = [line.strip() for line in f.readlines()]
    print(result)
    return result

# Carregar imagem
img = Image.open(IMAGE_PATH).convert("RGB")

# Carregar modelo
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Redimensionar imagem para o shape esperado pelo modelo
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
img_resized = img.resize((width, height))
img_array = np.array(img_resized)
input_data = np.expand_dims(img_array, axis=0).astype(np.float32)

# Normalizar se necessário
if input_details[0]['dtype'] == np.float32:
    input_data = (input_data - 127.5) / 127.5

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])  # shape: (1, 116, 8400)
print("Shape do output_data:", output_data.shape)

# Processar detecções YOLOv11n
labels = load_labels(LABEL_PATH)
num_classes = len(labels)
scores_start = output_data.shape[1] - num_classes  # Supondo que scores de classe estão no final

output = output_data[0]  # shape: (116, 8400)

best_score = -float('inf')
best_class = None
best_anchor = None

for anchor in range(output.shape[1]):
    class_scores = output[scores_start:, anchor]  # shape: (num_classes,)
    objectness = output[4, anchor]  # ajuste conforme seu modelo, normalmente posição 4
    class_idx = np.argmax(class_scores)
    score = class_scores[class_idx] * objectness  # score final
    if score > best_score:
        best_score = score
        best_class = class_idx
        best_anchor = anchor

if best_class is not None:
    print("Classe mais detectada:", labels[best_class])
    print("Score:", best_score)

    # Coletar boxes e scores para NMS
    boxes = []
    scores = []
    for anchor in range(output.shape[1]):
        objectness = output[4, anchor]
        if objectness > 0.5:
            x, y, w, h = output[0, anchor], output[1, anchor], output[2, anchor], output[3, anchor]
            if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                x *= width
                y *= height
                w *= width
                h *= height
            left = int(x - w / 2)
            top = int(y - h / 2)
            right = int(x + w / 2)
            bottom = int(y + h / 2)
            boxes.append([left, top, right, bottom])
            scores.append(objectness)

    def nms(boxes, scores, iou_threshold=0.5):
        boxes = np.array(boxes)
        scores = np.array(scores)
        idxs = np.argsort(scores)[::-1]
        keep = []
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            if len(idxs) == 1:
                break
            ious = []
            for j in idxs[1:]:
                xx1 = max(boxes[i][0], boxes[j][0])
                yy1 = max(boxes[i][1], boxes[j][1])
                xx2 = min(boxes[i][2], boxes[j][2])
                yy2 = min(boxes[i][3], boxes[j][3])
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h
                area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                union = area_i + area_j - inter
                iou = inter / union if union > 0 else 0
                ious.append(iou)
            idxs = np.delete(idxs, np.where(np.array(ious) > iou_threshold)[0] + 1)
            idxs = np.delete(idxs, 0)
        return keep

    keep = nms(boxes, scores, iou_threshold=0.5)
    draw = ImageDraw.Draw(img_resized)
    for i in keep:
        left, top, right, bottom = boxes[i]
        print(f"Box NMS: left={left}, top={top}, right={right}, bottom={bottom}, score={scores[i]}")
        draw.rectangle([left, top, right, bottom], outline='red', width=2)
    print(f"Total de bounding boxes após NMS: {len(keep)}")
    plt.figure(figsize=(8,8))
    plt.imshow(img_resized)
    plt.title('Detecções com Bounding Boxes (NMS)')
    plt.axis('off')
    plt.show()
else:
    print("Nenhuma classe detectada com score válido.")