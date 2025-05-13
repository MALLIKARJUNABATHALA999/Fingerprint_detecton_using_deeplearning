# blood group detection using finger print images
# 🩸 Blood Group Classification Using Fingerprint Images

This project uses a Convolutional Neural Network (CNN) model to classify blood groups (e.g., A+, O−, AB−) from fingerprint images. The system is trained on the **Fingerprint-based Blood Group Dataset**, demonstrating a strong performance with up to **93% validation accuracy**.

## 📌 Project Highlights

- ✅ Achieved **93% validation accuracy**
- 🧠 Built using **TensorFlow** and **Keras**
- 📷 Input: Fingerprint image
- 🩺 Output: Predicted blood group (e.g., `O+`, `AB-`)
- 🌐 Flask-based Web Interface for live predictions
- 📊 Balanced dataset via oversampling to reduce class imbalance
- 📉 Included ROC Curve, Confusion Matrix, and Accuracy/Loss plots

---

## 🚀 Technologies Used

- **Python 3.10**
- **TensorFlow / Keras**
- **Flask** for the web API
- **Matplotlib**, **Seaborn** for visualization
- **Sklearn** for evaluation metrics
- **HTML/CSS** for frontend (`templates/index.html`)

---

## 📂 Dataset

- Source: [Fingerprint-Based Blood Group Dataset](https://www.kaggle.com/datasets/nafis928/finger-print-based-blood-group-dataset)
- Format: Folder structure with images categorized by blood group
- Preprocessing: Resized to 64x64, normalized pixel values

---

## 🧠 Model Training Overview

- **Architecture**: Custom CNN
- **Input size**: 64x64x3
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Regularization**: EarlyStopping + ReduceLROnPlateau
- **Epochs**: 50
- **Accuracy Achieved**: **93% on validation set**

### ✅ Evaluation Metrics

- **Classification Report**
- **Confusion Matrix**
- **ROC Curves for Multiclass**
- **Training & Validation Accuracy and Loss Curves**

---


## 🖥️ Web App (Flask)

You can upload a fingerprint image through the `/predict` endpoint, and it returns:

```json
{
  "predicted_class": 0,
  "predicted_label": "A+",
  "confidence": 0.92
}
```
RUN LOCALLY 


git clone https://github.com/your-username/blood-group-classifier.git
cd blood-group-classifier
pip install -r requirements.txt
python app.py
