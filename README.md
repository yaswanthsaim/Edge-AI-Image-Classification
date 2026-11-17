# Edge-AI-Image-Classification
Image classification application using MobileNetV2, LiteRT, Python and Streamlit, with local inference and performance monitoring.
# Edge AI Image Classification Application

A practical Edge AI project that performs image classification locally using Python, a quantized MobileNetV2 model, LiteRT, and Streamlit.

## Project Overview

This application allows users to upload an image and receive a predicted image category through a simple web interface.

The model performs inference locally on the computer's CPU. The application also measures inference execution time and Python process memory usage.

## Technologies Used

- Python
- Streamlit
- MobileNetV2
- LiteRT
- NumPy
- Pillow
- psutil

## Project Structure

```text
Edge-AI-Image-Classification/
├── models/
│   └── mobilenet_v2_1.0_224_quant.tflite
├── app.py
├── labels.txt
├── requirements.txt
└── README.md
```

The `venv/` virtual environment is not included in this repository.

## Hardware Used for Testing

- Processor: Intel Core Ultra 5 125H
- RAM: 16 GB
- Operating System: Windows 11 Home Single Language
- Inference: LiteRT on CPU

## Experimental Results

Five different images were tested.

| Input image | Predicted class | Inference time |
|---|---|---:|
| Cat | tabby | 24.44 ms |
| Dog | smearkat | 28.54 ms |
| Car | sports car | 29.74 ms |
| Flower | daisy | 23.85 ms |
| Apple | Granny Smith | 21.99 ms |

Average inference time across these five tests: **25.71 ms**.

These are observations from five test images, not a formal model accuracy evaluation.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Edge-AI-Image-Classification.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Open the project directory

```bash
cd Edge-AI-Image-Classification
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment on Windows

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the model

Download the quantized MobileNetV2 model used by this project and place it inside the `models/` directory with this exact filename:

`mobilenet_v2_1.0_224_quant.tflite`

Use the original model source and check its terms before redistributing it. The model file is not included in this repository.

Also ensure that `labels.txt` is present in the project root.

### 7. Run the application

```bash
streamlit run app.py
```

The application should open in your browser at:

`http://localhost:8501`

## Features

- Upload an image through a Streamlit interface.
- Preprocess images for MobileNetV2.
- Perform local image classification using LiteRT.
- Display the predicted image category.
- Measure inference time.
- Monitor Python process memory usage.

## Limitations

- Predictions may be incorrect for some images.
- The current implementation uses CPU inference.
- Only five different images were used in the reported experiment.
- Process memory measurements represent the Python process, not the isolated model.

## Future Improvements

- Evaluate more images and image categories.
- Compare other lightweight classification models.
- Explore GPU and NPU acceleration.
- Add top-five predictions.
- Add real-time webcam classification.
- Improve memory and performance profiling.

## Project Article

Medium article: **https://medium.com/@yaswanthsaim2k8/building-an-edge-ai-image-classification-application-using-mobilenetv2-and-litert-602de7f86079**

## Author
M Yaswanth Sai

Student project: Edge AI Image Classification Application.
