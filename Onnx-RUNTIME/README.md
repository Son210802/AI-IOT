# Deploy with Onnx-RUNTIME

Here are instructions on how to deploy the model using Onnx-RUNTIME. ONNX Runtime is a performance-focused tool for ONNX models, enabling efficient inference across a variety of platforms and hardware such as Windows, Linux, Mac, and on both CPU and GPU. ONNX Runtime provides special optimizations that speed up inference, reduce latency, and improve overall model performance.

## Guide

### Load model: 

```python
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
```

### Export ONNX: 

```python
def export_onnx(self):
    # Export the loaded model to ONNX format.

    # Example input
    dummy_input = self._preprocess_image("./image/sample.jpg").unsqueeze(0).to(self.device)

    torch.onnx.export(self.model, 
                        dummy_input, 
                        self.onnx_model_path, 
                        export_params=True, 
                        opset_version=10, 
                        do_constant_folding=True, 
                        input_names=['input'], 
                        output_names=['output'])
    print(f"Model has been successfully exported to {self.onnx_model_path}")
```

### Image preprocessing: 

```python
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
```

### Inferrence:

```python
def inferrence(ortSession, imageY):
    
    imageY = imageY.unsqueeze(0)

    # Perform inference using ONNX Runtime
    ortInput = {ortSession.get_inputs()[0].name: imageY.numpy()}
    ortOutputs = ortSession.run(None, ortInput)
    
    return ortOutputs
```

### Display

```python
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
```
