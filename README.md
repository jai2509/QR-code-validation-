# **QR Code Authentication - README**

## **Project Overview**
This project implements a **QR code authentication system** to differentiate between **genuine (First Print)** and **counterfeit (Second Print)** QR codes. Using **Deep Learning (CNN)**, the model analyzes subtle differences in print quality to identify duplicates.

## **Features**
✅ **Pretrained CNN Model** for QR code classification.
✅ **Automated Data Preprocessing** (grayscale conversion, resizing, normalization).
✅ **Real-time Image Upload & Validation**.
✅ **Robust Error Handling** for missing folders & imbalanced datasets.

## **Installation & Setup**
### **1. Environment Setup**
Ensure you have the required dependencies installed:
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

### **2. Download Dataset**
Since Google Drive restrictions can prevent direct access, manually download the dataset from the provided link, then upload it to Google Colab.

### **3. Extract & Verify Data**
```python
import zipfile
zip_path = "/content/dataset.zip"
extract_path = "/content/dataset"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
!ls /content/dataset
```

## **Usage**
### **1. Train the Model**
Run the training script to preprocess data and train the CNN model:
```python
cnn_model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))
```

### **2. Test with a New Image**
You can upload a QR code image and check its authenticity:
```python
from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
predict_image(image_path)
```

## **Model Evaluation**
- **Accuracy, Precision, Recall, F1-score** used for evaluation.
- **Confusion matrix** helps identify misclassifications.
- Handled **single-class warnings** to ensure balanced training.

## **Challenges & Solutions**
🔹 **Google Drive access issues?** → Switched to manual dataset upload.
🔹 **Missing folders?** → Script auto-creates required directories.
🔹 **Imbalanced dataset?** → Checks ensure both classes exist.
🔹 **Live testing?** → Local image upload feature added.

## **Future Enhancements**
🚀 Improve accuracy with **data augmentation**.
🚀 Deploy as a **mobile app with TensorFlow Lite**.
🚀 Implement **real-time QR scanning** instead of static image uploads.

## **Contributing**
Feel free to contribute by improving model performance, adding features, or optimizing code. Fork the repository and submit a pull request!

## **License**
This project is open-source under the **MIT License**.

---

**Developed for robust QR code authentication & anti-counterfeiting solutions.**

