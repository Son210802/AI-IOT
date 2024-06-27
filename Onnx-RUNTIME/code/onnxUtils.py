from PIL import Image

from torchvision import models, transforms
import torch.nn as nn
import torch

import numpy as np
import onnxruntime
import onnx

# Check cuda
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path of onnx model
onnx_model_path = "/home/deepstream/Desktop/OnnxAPI/model/onnx/detect_angle_5.onnx" 

# Path of pretrained model
preTrainnedModelPath = "/home/deepstream/Desktop/OnnxAPI/model/normal/detect_angle_5.pt"

# Path of label
pathLabels = "/home/deepstream/Desktop/OnnxAPI/label/label.txt"

def loadModel():
    # Create device
    print(DEVICE)

    # Create model
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features

    # Change the last number of feauters
    model.fc = nn.Linear(num_ftrs, 36)
    
    # Load your pre trainned model from device
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(preTrainnedModelPath, map_location=torch.device(DEVICE)))

    # Set the model to evaluation mode
    model.eval()

    return model

def exportOnnx(model):
    batch_size = 1
    dummy_input = preProcessImage("./image/sample.jpg")    
    dummy_input.to(DEVICE)
    sampleX = dummy_input.unsqueeze(0) # add new dimesion at zero position

    torch.onnx.export(model,                 # model being run
                  sampleX,               # model input (or a tuple for multiple inputs)
                  onnx_model_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input hihi'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

    return onnx.load(onnx_model_path)

def loadOnnxModel(OnnxModel):
    # Load the ONNX model using ONNX Runtime
    return  onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

def preProcessImage(pathImage):
    # Load and preprocess the input image
    # input_image_path = "/home/deepstream/Desktop/OnnxAPI/image/cat_224x224.jpg"

    # These components represent luminance (Y), and the blue-difference (Cb) 
    # and red-difference (Cr) chroma components
    inputImage = Image.open(pathImage)
    print(np.array(Image.open(pathImage)).shape)

    preprocess = transforms.Compose([
        transforms.CenterCrop((247, 730)),  # Thay đổi kích thước ảnh
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),   # Chuyển đổi thành tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Chuẩn hóa dữ liệu
    ])
    
    return preprocess(inputImage)


def inferrence(ortSession, imageY):
    
    imageY = imageY.unsqueeze(0)

    # Perform inference using ONNX Runtime
    ortInput = {ortSession.get_inputs()[0].name: imageY.numpy()}
    ortOutputs = ortSession.run(None, ortInput)
    
    return ortOutputs

def loadLabels(labelPath):
    # Read list label from .txt file
    with open(labelPath, 'r') as file:
        # Read all lines from the file into a list
        labels = file.readlines()
        
        # Strip whitespace characters from each label and create a list of stripped labels
        labels = [label.strip() for label in labels]

    # Return the list of stripped labels
    return labels


if __name__ == '__main__':
    # Load pre trainned model in your local disk
    model = loadModel()

    # Export torch model to type of onnx model 
    onnxModel = exportOnnx(model)
    # Parse onnx model, if it errors, the note will appear
    onnx.checker.check_model(onnxModel)
    # Load the ONNX model using ONNX Runtime
    ortSession = loadOnnxModel(onnxModel)

    # Input your image want to inferrence
    # Then preprocess to right format image before
    # giving to onnx model
    pathImage = "./image/rightway.jpg"
    imageY = preProcessImage(pathImage)
    # print(preProcessImage(ortSession, pathImage))

    # Inferencing your image
    outputY = inferrence(ortSession, imageY)
    print(outputY)

    # Change list object to numpy object
    out = np.array(outputY)
    # Indicate position of values in array
    preds = np.argmax(out)
    print(preds)

    # Load label to display terminal monitor
    label = loadLabels(pathLabels)
    print(f'Predicted: {label[preds]}')
    
    # Annouccing success
    print("Build succesfully!")
