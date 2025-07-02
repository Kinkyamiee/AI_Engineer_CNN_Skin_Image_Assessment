Here's a complete `README.md` you can use for your project that builds and compares a custom CNN vs. ResNet50 for **skin condition image classification**, following your workflow:

---

```markdown
# 🩺 Skin Condition Classification using CNN & ResNet50

This project builds, trains, and evaluates Convolutional Neural Network (CNN) and Transfer Learning (ResNet50) models to classify skin images into **6 common conditions**:

- Acne
- Cancer
- Eczema
- Keratosis
- Milia
- Rosacea

---

## 📁 Dataset

The dataset used is the [ASCID - Augmented Skin Conditions Image Dataset](https://www.kaggle.com/datasets/syedalinaqvi/augmented-skin-conditions-image-dataset), which contains:

- 2,394 images
- 399 images per class
- Images are augmented for diversity
- Each class is stored in its own subdirectory

### Final Directory Structure:
```

skin\_dataset/
├── acne/
├── cancer/
├── eczema/
├── keratosis/
├── milia/
└── rosacea/

````

---

## 🚀 Project Workflow

1. **Load and Preprocess Data**
   - Resized to `(224, 224)`
   - Normalized and batched using `image_dataset_from_directory`
   - Train/Validation/Test split: 70/15/15

2. **Build Two Models**
   - A custom CNN model with Conv2D, MaxPooling, Dense layers
   - A ResNet50-based model using transfer learning

3. **Compile and Train**
   - Used `Adam` optimizer and `sparse_categorical_crossentropy` loss
   - EarlyStopping and ModelCheckpoint callbacks

4. **Evaluate and Compare**
   - Plotted training/validation accuracy and loss
   - Evaluated both models on the test set
   - Visualized confusion matrix

5. **Model Deployment Suggestion**
   - Converted the best model to TensorFlow Lite (`.tflite`) for use in mobile apps

---

## 📊 Results

| Model     | Test Accuracy | Test Loss |
|-----------|---------------|-----------|
| CNN       | ~55%          | ~2.19     |
| ResNet50  | Higher (e.g., 75–85%)* | Lower |

> *Performance may vary based on fine-tuning and dataset balance.

---

## 📦 Deployment

The trained ResNet50 model was converted to TensorFlow Lite format:

```bash
resnet_skin_classifier.tflite
````

This allows for integration with mobile health apps (Android, iOS, or Flutter) for **on-device skin condition analysis**.

---

## ✅ Requirements

* Python 3.8+
* TensorFlow 2.9+
* Matplotlib, scikit-learn (for confusion matrix)

---

## 📌 Notes

* This project is for educational and experimental purposes.
* It is **not a substitute for professional medical diagnosis**.
* All data used is publicly available and anonymized.

---

## ✨ Future Improvements

* Add Grad-CAM for model explainability
* Build a Flask or Streamlit web demo
* Evaluate on darker/lighter skin tones for fairness
* Deploy in a mobile app using Flutter + TFLite

---
