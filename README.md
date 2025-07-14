Here’s your updated `README.md` file reflecting all the latest updates — including **EfficientNet**, **Streamlit app**, and **TensorFlow Lite deployment options**:

---

```markdown
# 🩺 Skin Condition Classification using CNN, ResNet50 & EfficientNet

This project builds, trains, and evaluates three deep learning models to classify skin condition images into **6 common categories**:

- Acne
- Cancer
- Eczema
- Keratosis
- Milia
- Rosacea

Models implemented:
- ✅ Custom CNN (from scratch)
- ✅ ResNet50 (transfer learning)
- ✅ EfficientNetB0 (lightweight + high accuracy)

---

## 📁 Dataset

The dataset used is from the [ASCID - Augmented Skin Conditions Image Dataset](https://www.kaggle.com/datasets/syedalinaqvi/augmented-skin-conditions-image-dataset), which includes:

- 2,394 images
- 399 per class
- Augmented to increase variability
- Organized in folders per condition

### 📂 Directory Structure (after setup)
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

1. **Data Loading & Preprocessing**
   - Used `image_dataset_from_directory`
   - Images resized to `(224x224)`
   - Train/Validation/Test split: 70/15/15

2. **Model Building**
   - ✅ CNN: 3 Conv layers + BatchNorm + Dropout
   - ✅ ResNet50: Frozen base + Dense layers
   - ✅ EfficientNetB0: Best performance with fewer params

3. **Training & Evaluation**
   - EarlyStopping & ModelCheckpoint used
   - Trained for 10–20 epochs
   - Evaluated using accuracy, loss, and confusion matrix

4. **Model Comparison**
   - Accuracy and loss plotted
   - EfficientNet outperformed CNN and ResNet50

5. **Streamlit Web App**
   - Users can upload a skin image
   - App predicts condition + confidence score
   - Deployed locally or to Streamlit Cloud


## 🧪 Model Results (Example)

| Model        | Test Accuracy | Notes                        |
|--------------|---------------|------------------------------|
| CNN          | ~39%          | Basic, prone to overfitting  |
| ResNet50     | ~89%          | Better, deeper features      |
| EfficientNet | ✅ **~93%**   | Best accuracy + efficiency   |

> Note: Accuracy may vary slightly based on training setup and environment.

---

## 💻 Streamlit App

### 📸 Features:
- Upload any skin image
- Classifies into one of 6 conditions
- Shows top 3 predictions with confidence
- Responsive and easy to use

### ▶️ Run the App Locally

```bash
streamlit run app.py
````

---

## 🔄 Model Export & Mobile Deployment

* Models saved in `.keras` format:

  ```bash
  model.save('best_model.keras')
  ```

* Exported to TensorFlow Lite:

  ```python
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  ```

* TFLite model ready for mobile apps (Android, iOS, Flutter)

---

## 🔧 Requirements

* Python 3.8+
* TensorFlow 2.9+
* Streamlit
* PIL, NumPy, Matplotlib, scikit-learn

You can install dependencies via:

```bash
pip install -r requirements.txt
```

---

## 🧠 Lessons Learned

* Transfer learning greatly improves performance
* EfficientNet provides high accuracy + low size
* Streamlit is ideal for demoing ML models to users
* TensorFlow Lite makes deployment to mobile feasible

---

## ✨ Future Improvements

* Add Grad-CAM explainability
* Collect real-world samples for fine-tuning
* Deploy full app with TFLite in a Flutter app
* Build REST API for backend inference

---

## 👩‍💻 Author

**Amiee Hyacinth**
AI Engineer | Machine Learning Specialist
[LinkedIn](#) • [GitHub](#) • [Portfolio](#)

