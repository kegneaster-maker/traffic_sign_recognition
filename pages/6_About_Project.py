import streamlit as st

from src import ui

ui.setup("About the Project", "Concepts, methodology and limitations", "ℹ️")
st.graphviz_chart("""digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EAF1FF", color="#2E6BE6"];
 Dataset -> Inspect -> Preprocess -> Train -> Evaluate -> Dashboard; Train -> Model; Model -> Predict; Predict -> Dashboard; }""")
st.markdown("""
### Computer vision and image classification
Computer vision lets computers interpret images. *Image classification* assigns one label from a fixed set to an image;
here, one of 43 traffic-sign classes.

### Convolutional Neural Networks
A CNN learns filters that detect edges, shapes and textures. Convolution + ReLU layers extract features, pooling reduces
resolution, dropout reduces overfitting, dense layers combine features, and a softmax layer outputs class probabilities.

### GTSRB dataset
The German Traffic Sign Recognition Benchmark has 43 classes and over 50,000 images of German road signs.

### Preprocessing
Images are converted to RGB, resized (default 32 x 32), and scaled to [0, 1]. The same function is used for training and
prediction. Augmentation (small rotation, zoom, shift) is applied only while training; horizontal flips are avoided because
they could change a sign's meaning.

### Training and validation
Training images are split with a fixed seed into a training part and a stratified validation part. Early stopping,
checkpointing of the best model and learning-rate reduction guard against overfitting. Test images never influence training.

### Evaluation metrics
Accuracy, macro precision, macro recall, macro F1 (averaged over classes present), per-class scores and a confusion matrix.
Results are always labelled as validation or official test.

### Limitations
- Trained on German signs; may be unreliable for Ethiopian or other signs.
- Sensitive to blur, lighting, occlusion and images that contain more than the sign.
- Confidence is not a guarantee of correctness.
- Experimental and for academic demonstration only, not for driving decisions.

### Future improvements
Add a local-sign dataset, use deeper or pretrained models, add sign detection before classification,
calibrate confidence, and test robustness under weather and lighting changes.
""")
