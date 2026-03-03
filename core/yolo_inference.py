import os
import cv2
from ultralytics import YOLO

class ModelIterator:
    def __init__(self, model_dir):
        self.models = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.models):
            model_path = self.models[self.index]
            self.index += 1
            return model_path
        else:
            raise StopIteration

class YOLOLogic:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def predict(self, image_path):
        results = self.model(image_path)

        res_plotted = results[0].plot()

        rgb_res = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)

        status = "OK"

        return rgb_res, status