# Deploy with DeepStream

The following are instructions on how to deploy a model that determines the rotation angle and color of an object into DeepStream. DeepStream is a powerful and intelligent multi-stream video analysis AI engine, the GStreamer framework powers it. This guide only stops at putting the model into deepstream and printing the model results to the console screen, but in terms of processing speed, there are still many limitations.

## Guide

### Modify files config: 

Here we will put the model into the config file including 1 onnx file, 1 engine file, 1 label file along with related properties. [`file config`](https://github.com/Son210802/AI-IOT/blob/main/Image/fileconfig.jpg)
![`file config`](https://github.com/Son210802/AI-IOT/blob/main/Image/fileconfig.jpg)



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

### Load Model

```python
import onnxruntime as ort

def _loadOnnxModel(self):
    # Load the ONNX model using ONNX Runtime.
    return ort.InferenceSession(self.onnxModelPath)
```

### Define Input Pin

```python
def callbackFcn(channel):
    """
    Callback function that is triggered when the sensor detects an event.
    Captures an image and starts a prediction thread.
    """
    GPIO.output(ouputPin, GPIO.HIGH)
    if GPIO.input(sensorPin) == 0:
        print("Sensor detected!")
        cap = cv2.VideoCapture(4)  # Adjust camera index if needed
        access, img = cap.read()
        if access:
            cv2.imwrite(pathImage, img)
            Thread(target=threadPredict, args=(pathImage,)).start()
        else:
            print("Failed to capture image.")
        cap.release()
    print("finish!")
    GPIO.output(ouputPin, GPIO.LOW)

def _init_():
    """
    Function to initialize GPIO settings.
    sensorPin = 18
    ouput PIn = 16
    """
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(sensorPin, GPIO.IN)
    GPIO.setup(ouputPin, GPIO.OUT, initial=GPIO.LOW)
    GPIO.add_event_detect(sensorPin, GPIO.FALLING, callback=callbackFcn, bouncetime=20)

    print("Starting demo now! Press CTRL+C to exit\n")
```

### Infer

```python
def infernce(self, imagePath):
    # Perform inference on the preprocessed image using the ONNX model.

    ortSession = self._loadOnnxModel()

    image = self.preProcessImage(imagePath)
    self._loadLabels()

    imageY = image.unsqueeze(0)
    ortInput = {ortSession.get_inputs()[0].name: imageY.numpy()}
    ortOutputs = ortSession.run(None, ortInput)
    
    return ortOutputs
```

### Display

```python
def dispPreds(self, output):
    # Display the prediction results.
    out = np.array(output)
    preds = np.argmax(out)
    print(f'Predicted: {self.labels[preds]}')
```

> [!NOTE]  
> <sup>- This quote is sourced from [`onnx_utils.py`](https://github.com/leehoanzu/angle-detection/blob/main/onnx-runtime/onnx_utils.py) and [`main.py`](https://github.com/leehoanzu/angle-detection/blob/main/onnx-runtime/main.py).</sup><br>
> <sup>- For more detailed information, please refer to these files.</sup>
