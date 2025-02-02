# Emotion Detection from Images

## Overview
This project focuses on detecting emotions from images using various pre-trained deep learning models. The models are trained on raw data and evaluated based on accuracy, precision, and recall.

---
## Step 1: Training on Pre-Trained Models
The following models were used for training, and their performance metrics are recorded below:

### Model Performance Comparison
| Model                 | Trainer  | Epochs | Val Accuracy | Val Precision | Val Recall | Test Accuracy | Test Precision | Test Recall |
|-----------------------|----------|--------|--------------|--------------|------------|--------------|--------------|------------|
| Xception             | Monya    | 10     | 44.98%       | -            | -          | -            | -            | -          |
|                      |     | 20     | 48.44%       | -            | -          | -            | -            | -          |
| MobileNet_V2        | Advika   | 10     | 64.04%       | 0.64         | 0.64       | -            | -            | -          |
|                      |          | 15     | 73.20%       | 0.74         | 0.73       | -            | -            | -          |
|                      |          | 20     | 68.84%       | 0.69         | 0.69       | 62.57%       | 0.63         | 0.63       |
| ResNet18            | Advika   | 10     | 62.07%       | 0.62         | 0.62       | -            | -            | -          |
|                      |          | 15     | 61.79%       | 0.62         | 0.62       | -            | -            | -          |
|                      |          | 20     | 61.67%       | 0.61         | 0.62       | 61.44%       | 0.60         | 0.61       |
| SqueezeNet1_1       | Advika        | 10     | 56.29%       | 0.57         | 0.56       | -            | -            | -          |
|                      |          | 15     | 58.46%       | 0.58         | 0.58       | -            | -            | -          |
| ShuffleNet           |Manasvi  | 10     |        |          |        |        |          |        |
| VGG16               | Ayushi   | 15     | 32.63%       | -            | -          | 33.63%       | -            | -          |
| VGG19               | Ayushi   | 15     | 29.79%       | -            | -          | 30.44%       | -            | -          |
| InceptionV3         | Manasvi         | -      | -            | -            | -          | -            | -            | -          |
| EfficientNetV2-B0   | Ayushi   | 15     | 25.13%       | -            | -          | 24.71%       | -            | -          |
| ConvNeXt           | -        | -      | -            | -            | -          | -            | -            | -          |

---
## Step 2: To Be Continued...

