
SKIN CONDITION CLASSIFICATION USING CNN, RESNET50 & EFFICIENTNET

This project builds, trains, and evaluates three deep learning models to classify skin condition images into 6 common categories:

- Acne
- Cancer
- Eczema
- Keratosis
- Milia
- Rosacea

Models Implemented:
- Custom CNN (from scratch)
- ResNet50 (transfer learning)
- EfficientNetB0 (lightweight and high accuracy)

----------------------------------------------------------
DATASET

Dataset Source: ASCID - Augmented Skin Conditions Image Dataset
Link: https://www.kaggle.com/datasets/syedalinaqvi/augmented-skin-conditions-image-dataset

Details:
- 2,394 images
- 399 images per class
- Augmented to increase variability
- Each condition is stored in its own subdirectory

Directory Structure (after setup):

skin_dataset/
├── acne/
├── cancer/
├── eczema/
├── keratosis/
├── milia/
└── rosacea/

----------------------------------------------------------
PROJECT WORKFLOW

1. Data Loading and Preprocessing
- Used image_dataset_from_directory
- Resized all images to 224x224
- Dataset split: 70% training, 15% validation, 15% testing

2. Model Building
- CNN: 3 convolutional layers with batch normalization and dropout
- ResNet50: Pretrained base with frozen layers and custom top
- EfficientNetB0: Pretrained and fine-tuned, best trade-off of size and accuracy

3. Training and Evaluation
- Models trained with EarlyStopping and ModelCheckpoint
- Evaluated using accuracy, loss, and confusion matrix

4. Model Comparison
- Validation accuracy and loss plotted for all three models
- EfficientNetB0 showed the highest performance

5. Streamlit App
- A web-based tool to test the model on uploaded skin images
- Shows predicted condition and top 3 classes with confidence scores

6. Deployment
- Model exported in .keras and .tflite formats
- Ready for mobile health app integration (Android, iOS, Flutter)

----------------------------------------------------------
MODEL PERFORMANCE

Example Results:

Model        | Test Accuracy
-------------|----------------
CNN          | Approximately 55%
ResNet50     | Approximately 72%
EfficientNet | Approximately 84%

Note: Actual results may vary based on training environment and tuning.

----------------------------------------------------------
STREAMLIT APP

Features:
- Upload any skin image (JPEG/PNG)
- Returns the predicted skin condition with confidence
- Displays top 3 predictions with their probability

How to run locally:
1. Install Streamlit using: pip install streamlit
2. Run the app: streamlit run app.py

----------------------------------------------------------
DEPLOYMENT OPTIONS

1. Exported model as best_model.keras
2. Converted to TensorFlow Lite for mobile apps:
   - resnet_skin_classifier.tflite
3. Can be embedded in Android, iOS, or Flutter applications

----------------------------------------------------------
REQUIREMENTS

- Python 3.8+
- TensorFlow 2.9+
- Streamlit
- Pillow
- NumPy
- Matplotlib
- scikit-learn

Install all dependencies:
pip install -r requirements.txt

----------------------------------------------------------
LESSONS LEARNED

- Transfer learning significantly improves performance
- EfficientNetB0 offers high accuracy with a lightweight footprint
- Streamlit is powerful for quickly building ML demos
- TensorFlow Lite simplifies model deployment on mobile devices

----------------------------------------------------------
FUTURE IMPROVEMENTS

- Integrate Grad-CAM visual explanations for model transparency
- Collect more diverse, real-world data for fine-tuning
- Add REST API endpoint for backend integration
- Build a Flutter app with on-device inference using TFLite

----------------------------------------------------------
AUTHOR

Amiee Hyacinth
AI Engineer | Machine Learning Specialist
