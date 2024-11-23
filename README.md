# Flower Classification and Clustering with Deep Learning

This project involves two primary tasks: a flower classification task using Convolutional Neural Networks (CNNs) and a clustering task to group flower images based on their visual features. The objective is to utilize deep learning techniques to extract meaningful features from the flower images and categorize them effectively.

## Task 1: Flower Classification with CNN (25 Points)

### **Objective**
The goal of this task is to implement a flower classification system using a Convolutional Neural Network (CNN) trained on the Flowers102 dataset. The model should classify flower images into one of 102 categories. This task involves data preprocessing, building a baseline model, improving the model, and using transfer learning for better performance.

### **Steps to Complete the Task**

1. **Data Loading & Preprocessing**
   - Load the Flowers102 dataset using `torchvision.datasets`.
   - Apply necessary transformations such as resizing and normalization to ensure the images are compatible with the model.

2. **Baseline Model Implementation**
   - Implement the baseline CNN model as described in the table below:
   
   | Layer          | Layer Type    | Kernel Size | Stride | Padding | Output Channels |
   |----------------|---------------|-------------|--------|---------|-----------------|
   | Input          | ---           | ---         | ---    | ---     | 3               |
   | Convolutional  | Conv2D        | 3x3         | 1      | 1x1     | 32              |
   | Max Pooling    | MaxPool2D     | 2x2         | 2      | ---     | ---             |
   | Convolutional  | Conv2D        | 3x3         | 1      | 1x1     | 64              |
   | Max Pooling    | MaxPool2D     | 2x2         | 2      | ---     | ---             |
   | Convolutional  | Conv2D        | 3x3         | 1      | 1x1     | 128             |
   | Max Pooling    | MaxPool2D     | 2x2         | 2      | ---     | ---             |
   | Flatten        | ---           | ---         | ---    | ---     | ---             |
   | Fully Connected| Linear        | ---         | ---    | ---     | 512             |
   | Output Layer   | Linear        | ---         | ---    | ---     | 102             |

3. **Training the Baseline Model**
   - Train the model for 10 epochs using the SGD optimizer with a learning rate of 0.001.
   - Use min-max scaling for preprocessing and ReLU as the activation function.
   - Log model performance (loss, accuracy, F1-score) to TensorBoard for visualization.

4. **Model Evaluation**
   - Evaluate the model’s performance based on training and validation loss, accuracy, and F1 score.

5. **Model Improvement**
   - Improve the model’s performance by applying a minimum of three techniques learned in the course (Dropout, Batch Normalization).

6. **Transfer Learning**
   - Use a pre-trained CNN model ResNet50 from `torchvision.models` to achieve better performance than the baseline model.

### **Expected Outcomes**
- Build a robust CNN model for classifying flower images.
- Improve model performance through various techniques and transfer learning.
- Log and visualize training progress and metrics using TensorBoard.

---

## Task 2: Flower Clustering Using Unsupervised Learning (25 Points)

### **Objective**
This task focuses on clustering flower images based on their visual features. Using a pre-trained CNN, feature vectors will be extracted from flower images, and then unsupervised clustering (e.g., K-means) will be applied to group similar flowers together.

### **Steps to Complete the Task**

1. **Data Loading & Preprocessing**
   - Load the Flowers102 dataset using `torchvision.datasets` with appropriate preprocessing (e.g., resizing, normalization).

2. **Feature Extraction**
   - Use a pre-trained CNN model (e.g., ResNet50 or the custom CNN model from Task 1) to extract feature vectors from the flower images.

3. **Clustering**
   - Apply K-means clustering to group the extracted feature vectors into distinct clusters.
   - Choose an appropriate number of clusters based on the data.

4. **Clusters Visualization**
   - Use dimensionality reduction (e.g., PCA) to reduce the feature vectors to 2D or 3D for visualization.
   - Plot the clustered data points in a 2D or 3D space, using different colors to represent the different clusters.

### **Expected Outcomes**
- Extract meaningful features from flower images using a pre-trained CNN.
- Apply unsupervised learning techniques to group similar flowers together.
- Visualize the clusters effectively using dimensionality reduction and plot them in 2D or 3D space.

---

## Tools and Technologies

- **Python**: Used for all implementations.
- **Libraries**: PyTorch, TensorBoard, sklearn, numpy, matplotlib.
- **Pretrained Models**: ResNet50 from `torchvision.models` or custom CNN model.
- **Dataset**: Flowers102 dataset available in `torchvision.datasets`.

---

## Results

- **Task 1 (Classification)**: The performance of the baseline and improved models will be evaluated using metrics like accuracy, F1 score, and loss curves. The transfer learning model is expected to outperform the baseline model.
- **Task 2 (Clustering)**: After feature extraction and clustering, the flowers will be grouped into meaningful clusters, and their visual representation will be provided.

---

This project demonstrates the use of deep learning for image classification and unsupervised clustering, utilizing both supervised and unsupervised learning techniques to solve real-world problems in the floristry industry.
