# Brain Tumor Classification with Deep Learning

This repository contains a deep learning project focused on classifying brain tumors from MRI scans. Using a pre-trained convolutional neural network (CNN) model, we aim to build an efficient and accurate diagnostic tool for medical imaging analysis.

## Overview

### Key Features
- **Data Preparation**: Acquisition, exploration, and preprocessing of MRI brain tumor datasets.
- **Model Architecture**: Utilization of the Xception model as the base, fine-tuned with additional dense layers for classification.
- **Training and Evaluation**: Model training with metrics such as accuracy, precision, and recall for performance evaluation.
- **Predictions**: Capable of predicting tumor types from input MRI images.

### Dataset
The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/). It contains MRI images categorized into different tumor types, ensuring balanced representation for training and evaluation.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/brain-tumor-classification.git
    ```

2. Navigate to the project directory:
    ```bash
    cd brain-tumor-classification
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Preprocessing the Data
Run the preprocessing script to resize, normalize, and augment the dataset:
```bash
python preprocess_data.py
```

### Training the Model
To train the model, execute the following command:
```bash
python train_model.py
```
This will save the trained model and log the performance metrics.

### Making Predictions
Use the prediction script to classify an MRI image:
```bash
python predict.py --image path_to_image.jpg
```

### Visualizing Results
Evaluate the model performance using:
```bash
python evaluate_model.py
```
This will generate accuracy scores, confusion matrices, and other evaluation metrics.

## Project Structure

```
brain-tumor-classification/
├── data/                    # Dataset directory
├── models/                  # Saved model files
├── notebooks/               # Jupyter notebooks for exploration
├── scripts/                 # Python scripts for preprocessing, training, and evaluation
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## Results
Key results of the project include:
- **Accuracy**: Achieved high accuracy in classifying brain tumor types.
- **Metrics**: Precision, recall, and confusion matrix evaluations demonstrate robust model performance.

## Future Work
- Utilize larger datasets to improve generalization.
- Explore advanced architectures and transfer learning.
- Implement explainability techniques to increase model transparency.

## Credits
This project was developed using datasets from Kaggle and tools such as TensorFlow and Gamma for visualizations.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
