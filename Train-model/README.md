# Train 

To be able to train a model is not simple, first we have to see which training methods the model is suitable for such as classfication, regression, detection,... With the problem of determining angles as well as colors, the team has Use 2 types of problems for training to compare its effectiveness

## Classfication

With the classification method, we use a dataset of 2400 images with each different angle along with a csv file containing images with the corresponding angle, taking the error as +-5 degrees, for example from 0 to 4. degrees are put into class 1, 5 to 9 degrees are put into class 2, ... Training process at [`Classfication`](https://github.com/Son210802/AI-IOT/blob/main/Train-model/classification.ipynb) 

### Result

![`confusionmatrix`](https://github.com/Son210802/AI-IOT/blob/main/Image/confusionmatrix.jpg)

## Object detection

Using Ultralytics YOLOv8 along with running on the Google colab platform to be able to train the model, YOLOv8 will draw boxes for objects to make training more effective in determining color rotation angles, with only a dataset of 180 The image has been drawn with a box to identify the object. Training process at
[`Object detection`](https://github.com/Son210802/AI-IOT/blob/main/Train-model/object_detection.ipynb) 

### Result

![``](https://github.com/Son210802/AI-IOT/blob/main/Image/predict.jpg)
