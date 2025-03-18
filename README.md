# Gait-Recognition-Using-Template-Based-Methods
This repository presents an advanced approach to gait identification using three prominent template-based methods: Gait Energy Image (GEI), Gait Moment Image (GMI), and Motion Flow Energy Image (MFEI). The study leverages the CASIA-B dataset to extract gait features and enhance recognition performance.

## Methodology
Feature Extraction
The extracted gait templates were refined using the Histogram of Oriented Gradients (HOG), a powerful feature descriptor known for capturing detailed edge and gradient information.

Dimensionality Reduction
To enhance computational efficiency and ensure the most discriminative features are retained, Linear Discriminant Analysis (LDA) was employed as the dimensionality reduction technique. LDA optimally separates different gait patterns, improving classification accuracy.

Classification
Three distinct classifiers were evaluated to determine the best approach for gait recognition:

Support Vector Machine (SVM)
k-Nearest Neighbors (KNN)
Nearest Centroid (NC)
Results and Findings
The performance analysis of different gait templates and classifiers revealed that the GMI-NC combination achieved the best results. Specifically, the Nearest Centroid (NC) classifier with GMI features outperformed other methods, attaining an average accuracy of 82.82%. This combination demonstrated superior robustness across various covariate conditions, such as walking while carrying a bag, wearing a coat, and different viewing angles.

## Key Contributions
✅ Implementation of GEI, GMI, and MFEI for gait recognition \n
✅ Feature extraction using HOG for detailed edge representation
✅ Dimensionality reduction with LDA for optimal feature selection
✅ Comparative analysis of SVM, KNN, and NC classifiers
✅ Identification of the GMI-NC combination as the most effective

## Potential Applications
This study highlights the GMI-NC framework as a computationally efficient and accurate approach for gait recognition. Its robustness makes it suitable for biometric authentication, surveillance, and forensic analysis.

🔹 Code Implementation: This repository contains the full implementation of the methods used, including dataset processing, feature extraction, dimensionality reduction, classification, and evaluation.

## 📌 Dataset: The experiments were conducted on the CASIA-B dataset. Ensure the dataset is correctly placed before running the code.

Getting Started
To replicate the results, follow the instructions in the repository. Dependencies and setup guidelines are provided in the README.

🚀 Explore the repository and contribute to advancing gait recognition research!
