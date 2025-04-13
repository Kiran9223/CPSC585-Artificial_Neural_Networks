from ultralytics import YOLO
import cv2
import os
import torch
import time
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np

# Paths
image_folder = "images"
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# Images
image_street = os.path.join(image_folder, "Assignment3_street.jpg")
image_people = os.path.join(image_folder, "Assignment3_people.jpg")

# Load YOLOv8 Model
yolo_model = YOLO("yolov8n.pt")

def run_yolo(image_path):
    start = time.time()
    results = yolo_model(image_path)[0]
    end = time.time()
    
    boxes = results.boxes.xyxy.cpu().numpy()
    labels = results.names
    class_ids = results.boxes.cls.cpu().numpy().astype(int)

    predictions = [(labels[i], tuple(map(int, boxes[j]))) for j, i in enumerate(class_ids)]

    print("\n[YOLOv8 Results]")
    for label, box in predictions:
        print(f"Object: {label}, Box: {box}")

    people_count = sum(1 for i in class_ids if labels[i] == "person")
    print(f"Predicted People Count (YOLO): {people_count}")
    print(f"Detection Time (YOLO): {end - start:.2f}s")

    # Save image
    annotated = results.plot()
    output_path = os.path.join(output_folder, f"yolo_detected_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, annotated)

    return predictions, people_count, end - start


# Load Detectron2 Model
def setup_detectron2():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)

detectron_predictor = setup_detectron2()

def run_detectron2(image_path):
    img = cv2.imread(image_path)
    start = time.time()
    outputs = detectron_predictor(img)
    end = time.time()

    instances = outputs["instances"]
    pred_classes = instances.pred_classes.cpu().numpy()
    pred_boxes = instances.pred_boxes.tensor.cpu().numpy().astype(int)

    labels = MetadataCatalog.get("coco_2017_val").thing_classes
    predictions = [(labels[i], tuple(box)) for i, box in zip(pred_classes, pred_boxes)]

    print("\n[Detectron2 Results]")
    for label, box in predictions:
        print(f"Object: {label}, Box: {box}")

    people_count = sum(1 for i in pred_classes if labels[i] == "person")
    print(f"Predicted People Count (Detectron2): {people_count}")
    print(f"Detection Time (Detectron2): {end - start:.2f}s")

    # Save image
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("coco_2017_val"))
    out = v.draw_instance_predictions(instances.to("cpu"))
    output_path = os.path.join(output_folder, f"detectron2_detected_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

    return predictions, people_count, end - start

print("Actual People Count (Assignment3_people.jpg): [mocked: 28]")

print("\n--- Processing Assignment3_street.jpg ---")
run_yolo(image_street)
run_detectron2(image_street)

print("\n--- Processing Assignment3_people.jpg ---")
_, yolo_people_count, _ = run_yolo(image_people)
_, det_people_count, _ = run_detectron2(image_people)

# Accuracy Evaluation
actual_people_count = 28

def calculate_accuracy(predicted, actual):
    if actual == 0:
        return 0.0 if predicted > 0 else 100.0
    return round(100 * (1 - abs(predicted - actual) / actual), 2)

yolo_accuracy = calculate_accuracy(yolo_people_count, actual_people_count)
det_accuracy = calculate_accuracy(det_people_count, actual_people_count)

# Output Summary
print(f"\n[Summary: People Count]")
print(f"Actual: {actual_people_count}")
print(f"YOLOv8 Predicted: {yolo_people_count}")
print(f"Detectron2 Predicted: {det_people_count}")
print(f"\n[People Count Accuracy (approx)]")
print(f"YOLOv8 Accuracy: {yolo_accuracy}%")
print(f"Detectron2 Accuracy: {det_accuracy}%")

