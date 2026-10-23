# Traffic Sign Recognition Using CNN

## Project Overview
This project uses a Convolutional Neural Network (CNN) to recognize German traffic signs from images using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## Features
- Interactive Streamlit dashboard
- Dataset exploration
- Traffic sign image prediction
- CNN model training
- Model evaluation and performance metrics
- Prediction history

## Technologies Used
- Python
- TensorFlow / Keras
- Streamlit
- NumPy and Pandas
- Plotly
- Scikit-learn

## Project Structure
```text
traffic_sign_recognition/
├── app.py
├── config.py
├── requirements.txt
├── dataset/
├── models/
├── outputs/
├── pages/
├── src/
└── tests/
```

## Dataset
The project uses the GTSRB dataset, which contains 43 traffic sign classes. Download the dataset separately and configure the dataset path according to `config.py`.

Dataset source: https://benchmark.ini.rub.de/

## Installation
Create and activate a virtual environment, then install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## Run the Application
From the project directory, run:

```powershell
py -m streamlit run app.py
```

Then open the local URL displayed in the terminal, usually http://localhost:8501.

## Model Status
A trained model must be available before image predictions can work. Train the model using the project's training script after configuring the dataset.

## Disclaimer
This project is intended for educational and research purposes. Predictions should not be used as the sole basis for real-world driving decisions.