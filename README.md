# 🗺️ Landmark Detection with CNN — Classification + Bounding Box Regression

A multi-task deep learning model that identifies **landmark classes** and predicts **bounding box locations** from grayscale images, built with TensorFlow/Keras on the Google Landmarks dataset.

---

## 📌 Overview

The model solves two tasks from a single forward pass:

1. **Classification** — Predicts which landmark is in the image (multi-class, softmax output).
2. **Bounding Box Regression** — Predicts normalized `[xmin, ymin, xmax, ymax]` coordinates locating the landmark within the image.

A shared CNN backbone feeds both output heads, keeping the architecture compact and the inference fast.

> **Note**: The current script uses randomly generated bounding box targets during training (`np.random.rand`). To get meaningful localization results, replace those with real annotated coordinates from your dataset.

---

## 🗂️ Project Structure

```
Landmark_detection.py    # Data loading, model, training, evaluation, IoU, plots
train.csv                # Dataset index with image IDs and landmark labels
images/                  # Image directory (3-level hash structure: f1/f2/f3/id.jpg)
```

---

## 🧠 Model Architecture

```
Input (75×75×1 grayscale)
    │
    ▼
Conv2D(32, 3×3, ReLU) → MaxPooling2D
    │
Conv2D(64, 3×3, ReLU) → MaxPooling2D
    │
Flatten → Dense(128, ReLU)
    │
    ├──▶ Dense(num_classes, Softmax)  →  Classification output
    └──▶ Dense(4, Linear)            →  Bounding box output [xmin, ymin, xmax, ymax]
```

The number of output classes is inferred from the dataset at runtime via `LabelEncoder`.

---

## 📦 Requirements

### Python Version
- Python 3.8+

### Dependencies

```bash
pip install tensorflow numpy pandas opencv-python scikit-learn matplotlib Pillow
```

| Package          | Purpose                                        |
|------------------|------------------------------------------------|
| `tensorflow`     | Model building, training, evaluation           |
| `numpy`          | Array ops and IoU computation                  |
| `pandas`         | Loading and filtering `train.csv`              |
| `opencv-python`  | Image reading and resizing                     |
| `scikit-learn`   | Label encoding for landmark IDs                |
| `matplotlib`     | Training metric plots                          |
| `Pillow`         | Image utilities (PIL)                          |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/landmark-detection-cnn.git
cd Landmark-Detection-using-CNN
```

### 2. Prepare the Dataset

Download the [Google Landmarks Dataset](https://github.com/cvdfoundation/google-landmark) and place files as follows:

Images follow a 3-level directory structure based on the first three characters of the image ID.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> `requirements.txt`:
> ```
> tensorflow
> numpy
> pandas
> opencv-python
> scikit-learn
> matplotlib
> Pillow
> ```

### 4. Run

```bash
python Landmark_detection.py
```

The script will:
- Load and filter `train.csv` (IDs starting with `'00'`)
- Encode landmark labels with `LabelEncoder`
- Preprocess the first 2,000 images (grayscale, resize to 75×75, normalize)
- Train the dual-output CNN for 5 epochs
- Print evaluation metrics
- Plot classification accuracy and loss curves

---

## 📊 Dataset

- **Source**: [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark)
- **Format**: `train.csv` with columns `id` and `landmark_id`
- **Filtering**: Only rows where `id` starts with `'00'` are used
- **Loaded subset**: First 2,000 valid images
- **Preprocessing**:
  - Grayscale conversion via OpenCV
  - Resized to 75×75 pixels
  - Pixel values normalized to `[0, 1]`
  - Labels one-hot encoded for `num_classes` categories

---

## ⚙️ Configuration

| Parameter       | Value   | Description                                      |
|-----------------|---------|--------------------------------------------------|
| `IMG_SIZE`      | `75`    | Height and width of input images                 |
| `num_classes`   | Dynamic | Inferred from unique landmark IDs in the subset  |
| Max rows loaded | `2000`  | `df.head(2000)` — increase for more training data|
| `epochs`        | `5`     | Training epochs                                  |
| `batch_size`    | `32`    | Samples per gradient update                      |
| `val_split`     | `0.2`   | 20% of data held out for validation              |
| Optimizer       | `Adam`  | Default learning rate                            |
| Class loss      | `categorical_crossentropy` | Multi-class classification     |
| BBox loss       | `MSE`   | Bounding box regression                          |

---

## 📐 Intersection over Union (IoU)

The script includes an IoU function for post-training bounding box evaluation:

```
IoU = (Intersection Area + ε) / (Union Area + ε)
```

A smoothing factor `ε = 1e-10` prevents division by zero. Call it after generating real bounding box predictions to measure localization quality.

---

## 📉 Training Plots

Two plots are generated after training:

1. **Classification Accuracy** — Train and validation accuracy per epoch.
2. **Classification Loss** — Categorical cross-entropy per epoch.

Bounding box MSE plots can be added by calling:
```python
plot_metrics(history, "bounding_box_mse", "Bounding Box MSE")
```

---

## ⚠️ Known Issues

**Placeholder bounding box targets**: The training and evaluation loops use `np.random.rand(len(X), 4)` as bounding box ground truth. The model trains on random targets, so bounding box predictions will be meaningless until you replace these with real annotated coordinates.

**Class imbalance**: Filtering to IDs starting with `'00'` and capping at 2,000 rows produces an uneven class distribution. Check `df["landmark_id"].value_counts()` before training.

**OpenCV grayscale on missing files**: The script silently skips images that don't exist on disk. Print skipped paths during the load loop to catch dataset path mismatches early.

---

## 🔧 Extending the Project

- **Real bounding box labels** — Source or annotate actual landmark coordinates and replace the `np.random.rand` targets.
- **Color images** — Change `cv2.IMREAD_GRAYSCALE` to `cv2.IMREAD_COLOR`, update `input_shape` to `(75, 75, 3)`, and add `BatchNormalization` layers.
- **Larger input size** — Increase `IMG_SIZE` to 224 to use ImageNet-pretrained backbones (ResNet, EfficientNet) via transfer learning.
- **Full dataset** — Remove `df.head(2000)` and add a proper data generator (`tf.data.Dataset`) to handle large-scale loading without running out of RAM.
- **Data augmentation** — Add random flips, rotations, and brightness jitter to improve generalization.

---

## ☁️ Running on Google Colab

Mount Google Drive with your dataset, then run:

```python
!python Landmark_detection.py
```

Or copy the code into notebook cells. Use a GPU runtime — the Conv2D layers on 2,000 images are fast on CPU, but scaling to the full dataset will need acceleration.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push and open a Pull Request

---

## 🙏 Acknowledgements

- [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark) — CVDFoundation / Google
- [TensorFlow / Keras](https://www.tensorflow.org/) — Model framework
- [OpenCV](https://opencv.org/) — Image preprocessing
