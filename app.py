
import streamlit as st
import numpy as np
import cv2
import zipfile
from pathlib import Path
from PIL import Image
from io import BytesIO
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 CropGuard - AI Smart Pest Detection")
st.write("An intelligent machine learning system for crop pest identification.")

st.markdown("---")

# =====================================
# 2. DATASET PATH
# =====================================

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "crop_images.zip"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# =====================================
# 3. IMAGE FEATURE EXTRACTION
# =====================================

def extract_features(image):

    image = cv2.resize(image, (64, 64))

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    features = []

    for channel, upper in enumerate([180, 256, 256]):

        hist = cv2.calcHist(
            [hsv],
            [channel],
            None,
            [8],
            [0, upper]
        )

        hist = cv2.normalize(hist, hist).flatten()

        features.extend(hist)

    return np.array(features, dtype=np.float32)


# =====================================
# 4. LOAD DATASET FROM ZIP
# =====================================

def load_dataset():

    X = []
    y = []

    if not ZIP_PATH.exists():
        return X, y

    try:

        with zipfile.ZipFile(ZIP_PATH, "r") as archive:

            for file in archive.namelist():

                if file.lower().endswith(IMAGE_EXTENSIONS):

                    parts = Path(file).parts

                    if len(parts) < 2:
                        continue

                    # Folder name is the pest category
                    pest_name = parts[0]

                    try:

                        image_data = archive.read(file)

                        image = Image.open(
                            BytesIO(image_data)
                        ).convert("RGB")

                        image = np.array(image)

                        features = extract_features(image)

                        X.append(features)
                        y.append(pest_name)

                    except Exception:
                        continue

    except zipfile.BadZipFile:
        st.error("The dataset ZIP file is corrupted.")

    return X, y


# =====================================
# 5. TRAIN MACHINE LEARNING MODEL
# =====================================

@st.cache_resource
def train_model(zip_modified_time):

    X, y = load_dataset()

    if len(X) == 0:
        return None, {}, "No images found in the dataset."

    counts = Counter(y)

    if len(counts) < 2:
        return None, counts, "At least two pest categories are required."

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(np.array(X), np.array(y))

    return model, counts, None


# =====================================
# 6. PEST INFORMATION
# =====================================

pest_info = {

    "ants": "Ants may be associated with other crop pests. Monitor the plants and inspect affected areas.",

    "bees": "Bees are important pollinators. Avoid harming them and consult an agricultural expert if needed.",

    "snail": "Snails may damage young leaves. Monitor affected plants and follow integrated pest management.",

    "wasp": "Many wasps are beneficial insects. Identify the species before taking any control action.",

    "moth": "Some moth larvae damage crops. Inspect leaves for caterpillars and feeding signs.",

    "grasshopper": "Grasshoppers can damage leaves and young plants. Monitor crop areas regularly.",

    "weevil": "Some weevils damage leaves, seeds and stored crops. Inspect affected plant parts.",

    "earwig": "Earwigs may feed on plant material but can also be beneficial. Check crop damage first.",

    "catterpillar": "Caterpillars may eat crop leaves. Inspect plants and seek suitable pest management advice.",

    "beetle": "Some beetles damage crops, while others are beneficial. Confirm the species before treatment.",

    "slug": "Slugs may damage seedlings and leaves. Monitor damp areas and affected plants.",

    "earthworms": "Earthworms are generally beneficial for soil health and should not normally be treated as pests."
}


# =====================================
# 7. LOAD AND TRAIN MODEL
# =====================================

if ZIP_PATH.exists():

    model, class_counts, error = train_model(
        ZIP_PATH.stat().st_mtime
    )

else:

    model = None
    class_counts = {}
    error = "crop_images.zip not found. Keep it beside app.py."


# =====================================
# 8. SIDEBAR
# =====================================

st.sidebar.title("🌱 CropGuard AI")

st.sidebar.subheader("Dataset Information")

if class_counts:

    st.sidebar.success("Dataset Loaded Successfully")

    st.sidebar.metric(
        "Total Images",
        sum(class_counts.values())
    )

    st.sidebar.metric(
        "Pest Categories",
        len(class_counts)
    )

    st.sidebar.write("### Available Categories")

    for name, count in sorted(class_counts.items()):

        st.sidebar.write(
            f"• {name.title()} : {count} images"
        )

else:

    st.sidebar.error(error)


# =====================================
# 9. MAIN DASHBOARD
# =====================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📸 Upload Crop / Pest Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    uploaded_image = None

    if uploaded_file:

        try:

            uploaded_image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                uploaded_image,
                caption="Uploaded Image",
                use_container_width=True
            )

        except Exception:

            st.error("Invalid image file.")


with col2:

    st.subheader("🔍 Pest Detection Result")

    if uploaded_image is not None:

        if model is None:

            st.error(error)

        else:

            image_array = np.array(uploaded_image)

            features = extract_features(image_array)

            prediction = model.predict(
                [features]
            )[0]

            probabilities = model.predict_proba(
                [features]
            )[0]

            confidence = np.max(probabilities) * 100

            st.success("Classification Completed!")

            st.metric(
                "Predicted Pest",
                prediction.title()
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                float(confidence / 100)
            )

            st.subheader("🌿 Crop Protection Advice")

            advice = pest_info.get(
                prediction.lower(),
                "Monitor the crop and consult an agricultural expert."
            )

            st.info(advice)

            st.caption(
                "This is a prototype classification result. "
                "Confidence is not verified prediction accuracy."
            )

    else:

        st.info("Upload an image to start pest classification.")


# =====================================
# 10. ABOUT PROJECT
# =====================================

st.markdown("---")

st.subheader("About CropGuard")

st.write("""
CropGuard is an AI-based agricultural pest classification system.

It uses:

- Python for programming
- OpenCV for image processing
- Random Forest for machine learning
- Streamlit for the web interface

The system extracts image colour features,
classifies uploaded images into pest categories
and displays basic crop protection suggestions.
""")

st.markdown("---")

st.caption("CropGuard AI | Smart Agriculture Project")
