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
- Import necessary libraries (TensorFlow, Keras, NumPy, Matplotlib, etc.)
- Define directory structure for training and testing data
- Perform data visualization to understand class distribution and inspect sample images
- Build CNN model architectureCompile the model with appropriate optimizer, loss function, and metrics
- Train the model using training data and validate with test data
- Visualize training accuracy and loss to monitor model learning
- Evaluate model using:Confusion Matrix to observe misclassifications
- ROC Curve to measure classification performance across thresholds
  

📈 Model Performance & Evaluation
- The training accuracy initially reached 97%, but showed potential signs of overfitting
- Final test accuracy settled around 93%, indicating a still-strong generalization
- The loss curve decreased, and accuracy increased over epochs — suggesting good learning
- The Confusion Matrix shows where the model struggles and biased toward a specific class also which tumor types are commonly misclassified (e.g., tumor type confusion)
- The ROC Curve helps visualize how well the model distinguishes between classes at different threshold levels
  

🧠 Insight & Takeaways
Deep learning models often act as black boxes, making it hard to interpret internal decision-making. However, from this project, we can observe:
- The model is capable of learning useful patterns from CT scan images
- While accuracy is high, further tuning and possibly data augmentation or regularization are needed to reduce overfitting
- Visual inspection of learning curves is essential to monitor model behavior
- This project demonstrates the potential of deep learning for medical image classification, though interpretability and robustness remain key challenges.
