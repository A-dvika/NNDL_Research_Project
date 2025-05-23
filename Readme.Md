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
|           |  | 20     | 58.76%       | 0.61         | 0.59       | 57.70%       | 0.60         | 0.58       |
| VGG16               | Ayushi   | 15     | 32.63%       | 0.17            | 0.23          | 33.63%       | 0.18            | 0.22          |
| VGG19               | Ayushi   | 15     | 29.79%       | 0.19            | 0.24          | 30.44%       | 0.18            | 0.23          |
| InceptionV3         | Manasvi         | 10     | 37.33%           | -            | -          | 40.00%           | -            | -          |
| EfficientNetV2-B0   | Ayushi   | 10     | 62.24%       | 0.63            | 0.62          | -       | -            | -          |
|    |    | 15     | 62.00%       | 0.62            | 0.62          | -       | -            | -          |
|    |    | 20     | 62.97%       | 0.63            | 0.63          |  62.54%      | 0.63            | 0.63          |
| ConvNeXt           | Ayushi        | 10      | 58.48%            | 0.58            | 0.58          | -            | -            | -          |
|            |         | 15      | 59.11%            | 0.60            | 0.59          | -            | -            | -          |
|            |         | 20      | 57.99%            | 0.59            | 0.58          | 58.69%            | 0.59            | 0.59          |

---
## Step 2: To Be Continued...

