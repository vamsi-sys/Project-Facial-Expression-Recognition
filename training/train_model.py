"""
train_model.py  —  Run this on Google Colab (Runtime > GPU) to train the model.

DATASET SETUP (run these lines in a Colab cell before this script):

    !pip install -q kaggle
    from google.colab import files
    files.upload()   # upload your kaggle.json
    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    !kaggle datasets download -d msambare/fer2013
    !unzip -q fer2013.zip -d fer2013

Get kaggle.json from: kaggle.com → Account → Create New API Token

After training finishes:
  - Download emotion_model.keras  → place in backend/models/
  - Download labels.json          → place in backend/models/
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

DATA_DIR       = "fer2013"
IMG_SIZE       = 96
BATCH_SIZE     = 64
EPOCHS_FROZEN  = 15
EPOCHS_FINETUNE = 10

# ── Data ──────────────────────────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1.0/255, rotation_range=15, width_shift_range=0.1,
    height_shift_range=0.1, zoom_range=0.15, horizontal_flip=True,
    validation_split=0.1,
)
test_gen = ImageDataGenerator(rescale=1.0/255)

train_data = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, "train"), target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb", batch_size=BATCH_SIZE, class_mode="categorical", subset="training",
)
val_data = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, "train"), target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb", batch_size=BATCH_SIZE, class_mode="categorical", subset="validation",
)
test_data = test_gen.flow_from_directory(
    os.path.join(DATA_DIR, "test"), target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb", batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False,
)

NUM_CLASSES = train_data.num_classes
index_to_label = {v: k for k, v in train_data.class_indices.items()}
with open("labels.json", "w") as f:
    json.dump(index_to_label, f, indent=2)
print("Class mapping:", index_to_label)

# ── Model ─────────────────────────────────────────────────────────────────────
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                          include_top=False, weights="imagenet")
base_model.trainable = False

inputs  = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = GlobalAveragePooling2D()(x)
x       = Dense(256, activation="relu")(x)
x       = Dropout(0.4)(x)
x       = Dense(128, activation="relu")(x)
x       = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model   = Model(inputs, outputs)

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
]

# ── Phase 1: frozen backbone ───────────────────────────────────────────────────
print("\n=== Phase 1: training head only ===")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_data, validation_data=val_data, epochs=EPOCHS_FROZEN, callbacks=callbacks)

# ── Phase 2: fine-tune top layers ─────────────────────────────────────────────
print("\n=== Phase 2: fine-tuning top 30 layers ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_data, validation_data=val_data, epochs=EPOCHS_FINETUNE, callbacks=callbacks)

# ── Evaluate & save ────────────────────────────────────────────────────────────
loss, acc = model.evaluate(test_data)
print(f"\nTest accuracy: {acc*100:.2f}%")

model.save("emotion_model.keras")
print("\nDone! Download emotion_model.keras and labels.json from the Colab file browser.")
print("Place both in backend/models/")
