# img-deeplearning-brain

🧠 Brain Tumor Classification from CT Scans using Deep Learning
This deep learning project focuses on classifying brain CT scan images to determine whether they show:

- A specific tumor type
- Or a non-tumor (normal) brain
  
The model is trained to distinguish between these categories using convolutional neural networks (CNNs).


📁 Dataset
The dataset used in this project is publicly available on Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset


🎯 Objective
To develop a deep learning model that can automatically classify brain CT scan images into different tumor types or non-tumor categories with high accuracy.


⚙️ Project Workflow
- Import libraries (TensorFlow, Keras, NumPy, Matplotlib, etc.)
- Define directory structure for training and testing data
- Perform data visualization to understand class distribution and inspect sample images
- Build CNN model architecture
- Compile the model with appropriate optimizer, loss function, and metrics
- Using Transfer Learning to compare with CNN model
- Train the model using training data and validate with test data
- Visualize training accuracy and loss to monitor model learning
- Evaluate model using: Confusion Matrix to observe misclassifications
- ROC Curve to measure classification performance across thresholds
  

📈 Model Performance & Evaluation
- The initial CNN model achieved 99% training accuracy and 97% validation accuracy, though slight signs of overfitting were observed as validation loss fluctuated in later epochs.
- After applying transfer learning using VGG16, the model reached around 94% training accuracy and 93% test accuracy, showing strong generalization with improved class balance.
- Training shows a clear positive trend: loss decreases and accuracy increases overall. The most noticeable accuracy jump occurs early (0.0–0.1 epochs), followed by smaller fluctuations. Despite ups and downs, performance improves over time.
- The Confusion Matrix revealed 91% overall accuracy with consistent performance across classes and slightly better results on non-tumor cases, indicating strong and reliable classification.
- The ROC Curve confirmed the model's ability to differentiate well between multiple tumor classes.


🧠 Insight & Takeaways
Deep learning models often act as black boxes, making it hard to interpret internal decision-making. However, from this project, we can observe:
- The model is capable of learning useful patterns from CT scan images
- While accuracy is high, further tuning and possibly data augmentation or regularization are needed to reduce overfitting
- Visual inspection of learning curves is essential to monitor model behavior
- This project demonstrates the potential of deep learning for medical image classification, though interpretability and robustness remain key challenges.
