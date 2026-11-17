# CropGuard AI – Smart Pest Detection System

## About the Project
CropGuard AI is a machine learning-based web application designed to classify crop pests from uploaded images. It helps users identify common pest categories and provides basic pest management suggestions.

## Objectives
- Identify pest categories using machine learning.
- Provide a simple and user-friendly interface.
- Display prediction results and confidence scores.
- Provide basic pest management advice.

## Features
- Upload pest images.
- Predict the pest category.
- Display prediction confidence.
- Show basic pest management suggestions.
- View dataset category information.

## Technologies Used
- **Programming Language:** Python
- **Frontend:** Streamlit
- **Image Processing:** OpenCV
- **Machine Learning:** Scikit-learn
- **Numerical Processing:** NumPy
- **Image Handling:** Pillow

## Machine Learning Algorithm
The project uses the **Random Forest Classifier** with HSV color histogram features extracted from images.

## Dataset
The dataset contains approximately 5,494 images across 12 pest categories.

The dataset is stored in `crop_images.zip` and is used to train the classification model.

## Project Structure

```text
CropGuard-AI/
│
├── app.py
├── requirements.txt
├── README.md
└── crop_images.zip
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/CropGuard-AI.git
```

### 2. Open the Project Folder

```bash
cd CropGuard-AI
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

## How It Works
1. The user uploads a pest image.
2. The application extracts image features.
3. The trained Random Forest model predicts the pest category.
4. The predicted category and confidence score are displayed.
5. Basic pest management suggestions are provided.

## Limitations
- Prediction depends on the quality and variety of dataset images.
- The model classifies images; it does not locate individual pests.
- Prediction confidence is not a verified measure of real-world accuracy.
- Further testing is required before practical agricultural use.

## Future Enhancements
- Improve prediction accuracy using deep learning.
- Add more pest categories.
- Support real-time camera-based detection.
- Develop a mobile-friendly application.
- Provide multilingual support.

## Developed By
**Swagata Bijli**

B.Tech – Computer Science and Engineering (AIML)

## Project Name
**CropGuard AI – Smart Pest Detection System**
