from PIL import Image

import torch
from torchvision import transforms

from ultralytics import YOLO

import time

import logging

class YOLOInference:
    def __init__(self, preTrainedModelPath, labelPath, classPath="./label/class.txt"):
        # Set logging level to ERROR for the Ultralytics module
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
        
        # Load a pretrained YOLO model
        self.model = self.load_model(preTrainedModelPath) 
        self.model.predict()
        
        self.label = self.load_labels(labelPath)
        self.classId = self.load_labels(classPath)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def load_labels(self, labelPath):
        # Read list label from .txt file
        with open(labelPath, 'r') as file:
            # Read all lines from the file into a list
            labels = file.readlines()
            
            # Strip whitespace characters from each label and create a list of stripped labels
            labels = [label.strip() for label in labels]

        # Return the list of stripped labels
        return labels

    def load_model(self, preTrainedModelPath):
        # Load a pretrained YOLOv8n model
        return YOLO(preTrainedModelPath, task ="obb")
        
    def preprocess_image(self, pathImage):
        # Load and preprocess the input image
        inputImage = Image.open(pathImage)
        
        preprocess = transforms.Compose([
            transforms.CenterCrop((240, 320)),  # Resize the image to match the input shape
            # transforms.ToTensor(),   # Chuyển đổi thành tensor
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Chuẩn hóa dữ liệu
        ])
        
        return preprocess(inputImage)
    

    def predict(self, pathImage):
        results = self.model.predict(pathImage, device=self.device) # stream_buffer=True
        # results[0].show() show bounding box
        cls = None
        angle = None
        try:
            angle = torch.round(180 - (57.29577951 * (results[0].obb.data[0, 4])))
            cls = results[0].obb.data[0, -1]  

            # Check if cls and angle have been assigned properly
            if cls is None or angle is None:
                return None

        except Exception as e:
            print(e)
            return None
        # Return class of color and angle
        # Note: label start from 0
        return int(cls), int(angle)
    
    def display(self, cls, angle):            
        # Display prediction on terminal console
        print(f'This is {self.classId[cls][3:]} object at {self.label[angle][4:]}\n')


if __name__ == '__main__':    
    
    # Path of pretrained model
    preTrainedModelPath = "./model/onnx/best.pt"
    # Load the exported TensorRT model
    #preTrainedModelPath = "./model/best.engine" 

    # Path of label
    labelPath = "./label/label.txt"

    # Initialize YOLOInference class
    yolo_inference = YOLOInference(preTrainedModelPath, labelPath)

    pathImage = "./image/rightway.jpg"
    
    t0 = time.time()

    # Perform prediction
    results = yolo_inference.predict(pathImage)

    if results is not None:
        cls, angle = results
        yolo_inference.display(cls, angle)  
    else:
        print("Try again!\n")
    
    print(f'Done. ({time.time() - t0:.3f}s)')
    print("Build successfully!")
