# import tensorflow as tf

# # Check your TensorFlow version
# print("TensorFlow version:", tf.__version__)

# # Try loading the model
# model_path = r"models\mobilenetv3_after_finetuning.keras"  # update path if needed
# model = tf.keras.models.load_model(model_path, compile=False)

# print("✅ Model loaded successfully!")
# print(model.summary())


import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model

# Model parameters
WIDTH, HEIGHT, CHANNELS = 224, 224, 3
NUM_CLASSES = 15  # change to match your dataset

# Rebuild same architecture
base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(WIDTH, HEIGHT, CHANNELS))
base_model.trainable = True  # you fine-tuned last 30 layers before, so this is okay

inputs = Input(shape=(WIDTH, HEIGHT, CHANNELS))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs, outputs)

# Load weights from your saved model
try:
    model.load_weights("models/mobilenetv3_after_finetuning.keras")
    print("✅ Weights loaded successfully!")
except Exception as e:
    print("❌ Error loading weights:", e)

model.summary()

 
