# Digital-image-forgery-detection
semester Project for SPU - Digital image forgery detection using Deep Learning

# Methodology 1
Digital Image Forgery Detection: Methodology 1 (Dual-Input ResNet50V2)
This project is part of a research thesis focused on developing an intelligent system for detecting digital image forgery using deep learning techniques. This methodology represents one of three experiments conducted to evaluate the effectiveness of different models for this task.

# Methodology Summary
This approach utilizes a dual-input ResNet50V2 neural network model. The model is trained on two parallel streams:

**Stream 1:** Takes the original RGB images as input.

**Stream 2:** Takes the Error Level Analysis (ELA) images as input.

The model merges the features extracted from both streams to enhance its ability to detect subtle signs of image manipulation.

# Key Results
The model was trained on the CASIA 2.0 dataset. This methodology demonstrated the best overall performance compared to the other approaches, achieving:

Overall Accuracy: 87.88%.

Best AUC (Area Under Curve) Value: 0.9468.

Recall for Forged Class: 0.8937.

These results prove that the dual-input model strikes an excellent balance between accuracy and computational efficiency.

How to Use the Code
To access and run the code for this methodology, please open the following Google Colab link:
[(https://colab.research.google.com/drive/1Ih6SUtPCtkl-Ix89bDkqD1srwx5ZbiXg?usp=sharing)]
