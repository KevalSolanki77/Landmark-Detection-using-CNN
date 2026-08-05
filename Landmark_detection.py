import numpy as np
import pandas as pd
import tensorflow as tf
import cv2, os, random
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from PIL import Image

# 1. Load Dataset
df = pd.read_csv("train.csv")
base_path = "./images/"

# Filter and encode labels
df = df.loc[df["id"].str.startswith('00', na=False), :]
lencoder = LabelEncoder()
df["landmark_id"] = lencoder.fit_transform(df["landmark_id"])

num_classes = len(lencoder.classes_)
print("Number of classes:", num_classes)


# 2. Data Pipeline
IMG_SIZE = 75
def preprocess(img_path, label):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.reshape(IMG_SIZE, IMG_SIZE, 1), label

images, labels, bboxes = [], [], []
for i, row in df.head(2000).iterrows():  
    fname = row["id"] + ".jpg"
    f1, f2, f3 = fname[0], fname[1], fname[2]
    path = os.path.join(base_path, f1, f2, f3, fname)
    
    if os.path.exists(path):
        img, lbl = preprocess(path, row["landmark_id"])
        
        # Extract true normalized box coordinates from your CSV row
        # (Assuming CSV columns: xmin, ymin, xmax, ymax in pixel values)
        orig_w, orig_h = row["width"], row["height"] # Original image dimensions
        box = [
            row["xmin"] / orig_w,
            row["ymin"] / orig_h,
            row["xmax"] / orig_w,
            row["ymax"] / orig_h
        ]
        
        images.append(img)
        labels.append(lbl)
        bboxes.append(box)

X = np.array(images)
y_cls = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
y_box = np.array(bboxes, dtype=np.float32)  # Real targets shape: (N, 4)

# 3. Model Definition
inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# Feature extractor
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)

classification_output = tf.keras.layers.Dense(num_classes, activation='softmax', name="classification")(x)
bounding_box_output = tf.keras.layers.Dense(4, name="bounding_box")(x)

model = tf.keras.Model(inputs=inputs, outputs=[classification_output, bounding_box_output])

model.compile(optimizer='adam',
              loss={'classification': 'categorical_crossentropy',
                    'bounding_box': 'mse'},
              metrics={'classification': 'accuracy',
                       'bounding_box': 'mse'})

model.summary()

# 4. Train
history = model.fit(
    X, 
    {"classification": y_cls, "bounding_box": y_box}, 
    validation_split=0.2,
    epochs=5, 
    batch_size=32
)

# 5. Evaluate with real target arrays
losses = model.evaluate(X, {"classification": y_cls, "bounding_box": y_box})
print("Evaluation Losses & Metrics:", losses)
# 5. Evaluation
losses = model.evaluate(X, {"classification": y, "bounding_box": np.random.rand(len(X), 4)})
print("Evaluation:", losses)

# 6. IoU Metric
def intersection_over_union(pred_box, true_box):
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = np.split(pred_box, 4, axis=1)
    xmin_true, ymin_true, xmax_true, ymax_true = np.split(true_box, 4, axis=1)

    smoothing_factor = 1e-10

    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymin_overlap = np.maximum(ymin_pred, ymin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    inter_area = np.maximum(0, xmax_overlap - xmin_overlap) * np.maximum(0, ymax_overlap - ymin_overlap)
    pred_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)
    union_area = pred_area + true_area - inter_area + smoothing_factor

    return (inter_area + smoothing_factor) / union_area

# 7. Visualization
def plot_metrics(history, metric, title):
    plt.plot(history.history[metric], label=metric)
    plt.plot(history.history["val_" + metric], label="val_" + metric)
    plt.title(title)
    plt.legend()
    plt.show()

plot_metrics(history, "classification_accuracy", "Classification Accuracy")
plot_metrics(history, "classification_loss", "Classification Loss")
