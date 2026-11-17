import time
from pathlib import Path

import numpy as np
import psutil
import streamlit as st
from PIL import Image
from ai_edge_litert.interpreter import Interpreter


# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = Path("models/mobilenet_v2_1.0_224_quant.tflite")
LABEL_PATH = Path("labels.txt")


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Edge AI Image Classification",
    page_icon="🧠",
    layout="centered"
)


# =========================================================
# PAGE TITLE
# =========================================================

st.title("🧠 Edge AI Image Classification")

st.write(
    "A lightweight image classification application using "
    "MobileNetV2 and TensorFlow Lite/LiteRT."
)

st.info(
    "Inference is performed locally on this computer. "
    "The image is not sent to a cloud prediction service."
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    interpreter = Interpreter(
        model_path=str(MODEL_PATH),
        num_threads=2
    )

    interpreter.allocate_tensors()

    return interpreter


# =========================================================
# LOAD LABELS
# =========================================================

@st.cache_data
def load_labels():

    with open(
        LABEL_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return [
            line.strip()
            for line in file.readlines()
        ]


# =========================================================
# INITIALIZE MODEL
# =========================================================

interpreter = load_model()
labels = load_labels()


# =========================================================
# MODEL INFORMATION
# =========================================================

input_details = interpreter.get_input_details()[0]

output_details = interpreter.get_output_details()[0]

input_shape = input_details["shape"]

input_height = int(input_shape[1])
input_width = int(input_shape[2])


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image to model input size
    image = image.resize(
        (input_width, input_height),
        Image.Resampling.BILINEAR
    )

    # Convert image to NumPy array
    image_array = np.asarray(
        image
    ).astype(np.float32)

    # MobileNetV2 preprocessing
    # Convert 0-255 pixels to approximately -1 to +1
    image_array = (
        image_array / 127.5
    ) - 1.0

    # Quantization parameters
    scale, zero_point = (
        input_details["quantization"]
    )

    # Convert normalized values to uint8
    image_array = np.round(
        image_array / scale + zero_point
    )

    image_array = np.clip(
        image_array,
        0,
        255
    ).astype(np.uint8)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# CLASSIFICATION
# =========================================================

def classify_image(image):

    # Preprocess image
    input_tensor = preprocess_image(image)

    # Get current process
    process = psutil.Process()

    # Memory before inference
    memory_before = (
        process.memory_info().rss
        / (1024 * 1024)
    )

    # Start timer
    start_time = time.perf_counter()

    # Send image to model
    interpreter.set_tensor(
        input_details["index"],
        input_tensor
    )

    # Run local inference
    interpreter.invoke()

    # Get model output
    output = interpreter.get_tensor(
        output_details["index"]
    )

    # Stop timer
    inference_time = (
        time.perf_counter()
        - start_time
    ) * 1000

    # Memory after inference
    memory_after = (
        process.memory_info().rss
        / (1024 * 1024)
    )

    # =====================================================
    # DEQUANTIZE OUTPUT
    # =====================================================

    output_scale, output_zero_point = (
        output_details["quantization"]
    )

    scores = (
        output.astype(np.float32)
        - output_zero_point
    ) * output_scale

    # Remove batch dimension
    scores = scores[0]

    # Ignore ImageNet background class
    scores[0] = -np.inf

    # Find highest scoring class
    predicted_index = int(
        np.argmax(scores)
    )

    predicted_label = labels[
        predicted_index
    ]
# =====================================================
# CONFIDENCE
# =====================================================

# Convert model scores to probabilities using softmax
    stable_scores = (
        scores - np.max(scores)
    )

    exp_scores = np.exp(
        stable_scores
    )

    probabilities = (
        exp_scores
        / np.sum(exp_scores)
    )

    confidence = (
        float(
            probabilities[
                predicted_index
            ]
        )
        * 100
    )

    return (
        predicted_label,
        confidence,
        inference_time,
        memory_before,
        memory_after
    )


# =========================================================
# USER INTERFACE
# =========================================================

st.header("📷 Upload an Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# =========================================================
# DISPLAY AND CLASSIFY IMAGE
# =========================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(
        uploaded_file
    )

    # Display original image
    st.image(
        image,
        caption="Input Image",
        width="stretch"
    )

    # Classification button
    if st.button(
        "🔍 Classify Image",
        type="primary"
    ):

        with st.spinner(
            "Running local AI inference..."
        ):

            (
                prediction,
                confidence,
                inference_time,
                memory_before,
                memory_after
            ) = classify_image(
                image
            )

        # =================================================
        # PREDICTION
        # =================================================

        st.success(
            f"Prediction: {prediction}"
        )



        # =================================================
        # PERFORMANCE
        # =================================================

        st.subheader(
            "⚡ Edge AI Performance"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Inference Time",
                f"{inference_time:.2f} ms"
            )

        with col2:

            st.metric(
                "Memory Before",
                f"{memory_before:.1f} MB"
            )

        with col3:

            st.metric(
                "Memory After",
                f"{memory_after:.1f} MB"
            )

        # =================================================
        # MODEL INFORMATION
        # =================================================

        st.subheader(
            "🤖 Model Information"
        )

        st.write(
            "**Model:** MobileNetV2 Quantized"
        )

        st.write(
            f"**Input:** "
            f"{input_width} × {input_height} RGB"
        )

        st.write(
            "**Runtime:** LiteRT / TensorFlow Lite"
        )

        st.write(
            "**Execution:** Local CPU"
        )

        model_size = (
            MODEL_PATH.stat().st_size
            / (1024 * 1024)
        )

        st.write(
            f"**Model Size:** "
            f"{model_size:.2f} MB"
        )