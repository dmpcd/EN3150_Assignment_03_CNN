
# **EN3150 Assignment 03**  

## **Project Overview**  
This repository contains the complete implementation of a **Convolutional Neural Network (CNN)** to perform image classification. The project focuses on developing a custom CNN model and comparing it with fine-tuned, state-of-the-art pre-trained models. The assignment explores the impact of architecture design, optimizer selection, and learning rates on model performance.

This repository adheres to the EN3150 assignment guidelines and incorporates all the required tasks, analysis, and discussion points outlined in the assignment document.

---

## **Key Features**  
1. **Custom CNN Architecture**:
   - Constructed using convolutional, pooling, and fully connected layers.  
   - Includes dropout to reduce overfitting and softmax for classification.  
   - Trained on a dataset from the **UCI Machine Learning Repository**.  

2. **State-of-the-Art Pre-Trained Models**:
   - Fine-tuning applied to pre-trained models (**ResNet** and **DenseNet**).  
   - Training, validation, and testing on the same dataset for comparison.  

3. **Evaluation and Analysis**:
   - Comparison of training/validation loss and test accuracy between the custom CNN and fine-tuned models.  
   - Visualizations for loss and accuracy at varying learning rates (0.0001, 0.001, 0.01, and 0.1).  
   - Metrics including confusion matrix, precision, and recall.  

4. **Justifications and Discussions**:
   - Explanation for activation function choices, optimizer selection, and loss function.  
   - Analysis of trade-offs between custom and pre-trained models.  

---

## **Dataset Details**  
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets).  
- **Split Ratios**:
  - Training: 60%  
  - Validation: 20%  
  - Testing: 20%  
- **Processing**: 
  - Data was preprocessed and normalized for training.  
  - Labels were encoded to be compatible with categorical crossentropy loss.

---

## **Implementation Steps**  

### **1. Setting Up the Environment**  
- Installed necessary packages for Python:  
  - TensorFlow/Keras for building the CNN.  
  - PyTorch for working with pre-trained models.  
- Ensured a compatible Python environment (version >= 3.8).  

### **2. Preparing the Dataset**  
- Selected an image classification dataset (excluding CIFAR-10) from the UCI repository.  
- Preprocessed the data for input to the CNN:
  - Normalized pixel values to the range `[0, 1]`.  
  - Split the dataset into training, validation, and testing subsets.  

### **3. Designing the Custom CNN Architecture**  
The following architecture was implemented for the custom CNN:
- **Layer 1**: Convolutional layer with `x1` filters and `m1×m1` kernel.  
- **Layer 2**: MaxPooling layer for spatial dimension reduction.  
- **Layer 3**: Convolutional layer with `x2` filters and `m2×m2` kernel.  
- **Layer 4**: MaxPooling layer.  
- **Layer 5**: Flattened output from previous layers.  
- **Layer 6**: Fully connected layer with `x3` units and a suitable activation function.  
- **Layer 7**: Dropout layer with rate `d` to reduce overfitting.  
- **Output Layer**: Fully connected layer with `K` units for classification and softmax activation.

The architecture was optimized based on:
- Activation functions: Chosen for non-linearity and efficient gradient flow.  
- Dropout rate: Tuned to balance regularization without underfitting.  

### **4. Training the Custom Model**  
- **Optimizer**: Adam (chosen over SGD for better convergence speed and stability).  
- **Loss Function**: Sparse categorical crossentropy (suitable for multi-class classification).  
- **Learning Rate**: Experimented with multiple rates (`0.0001`, `0.001`, `0.01`, `0.1`).  
- **Epochs**: 20 iterations to ensure sufficient convergence.  

Training results were analyzed using:
- Plots of training/validation loss across epochs.  
- Test accuracy and additional metrics (confusion matrix, precision, recall).

### **5. Fine-Tuning Pre-Trained Models**  
- **Models Used**: ResNet and DenseNet (pre-trained on ImageNet).  
- Fine-tuned using the same training/validation/testing datasets.  
- Evaluation metrics recorded and compared with the custom CNN model.

### **6. Evaluation and Comparison**  
- **Metrics**:
  - Training and testing accuracy.  
  - Confusion matrix, precision, and recall.  
- **Visualizations**:
  - Plots of loss curves for different learning rates.  
- **Discussion**:
  - Trade-offs between training a custom model versus using pre-trained models.  
  - Advantages and limitations of both approaches.

---

## **Results**  
- **Custom CNN**:
  - Training accuracy: XX%  
  - Test accuracy: XX%  
- **Fine-Tuned Pre-Trained Models**:
  - ResNet test accuracy: XX%  
  - DenseNet test accuracy: XX%  
- The pre-trained models outperformed the custom CNN in terms of accuracy, but at the cost of increased training time and computational resources.  

---

## **Repository Structure**  
```
├── data/                  # Dataset files  
├── src/                   # Source code  
│   ├── custom_cnn.py      # Custom CNN model implementation  
│   ├── pretrained_model.py  # Fine-tuning pre-trained models  
├── plots/                 # Training and validation plots  
├── results/               # Metrics and evaluation results  
├── README.md              # Documentation  
└── report.pdf             # Final report submission  
```

---

## **How to Run**  

### **Prerequisites**  
Ensure Python 3.8+ is installed along with the following libraries:  
- TensorFlow  
- PyTorch  
- Matplotlib  

### **Steps**  
1. Clone the repository:
   ```bash
   git clone https://github.com/username/simple-cnn-classifier.git
   cd simple-cnn-classifier
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset:
   - Download the dataset and place it in the `data/` directory.
   - Ensure the dataset is preprocessed and split.  

4. Run the custom CNN model:
   ```bash
   python src/custom_cnn.py
   ```

5. Run the pre-trained model:
   ```bash
   python src/pretrained_model.py
   ```

---

## **References**  
- Murphy, K. P. *Probabilistic Machine Learning: An Introduction*, MIT Press, 2022.  
- [TensorFlow](https://www.tensorflow.org/)  
- [PyTorch](https://pytorch.org/)  
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets)  


