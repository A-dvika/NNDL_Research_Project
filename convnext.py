import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# Image dimensions
IMG_SIZE = (48, 48)  # FER dataset images are 48x48
BATCH_SIZE = 32

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load train and test datasets
train_generator = train_datagen.flow_from_directory(
    "dataset/fer/train", 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    color_mode="rgb",  # Change to RGB (3 channels)
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    "dataset/fer/test", 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    color_mode="rgb",  # Change to RGB (3 channels)
    class_mode="categorical"
)

num_classes = len(train_generator.class_indices)
print("Number of classes:", num_classes)

# Load ConvNeXt model (Tiny variant)
base_model = ConvNeXtTiny(weights="imagenet", include_top=False, input_shape=(48, 48, 3))

# Freeze base model layers (optional)
base_model.trainable = False

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")  # Output layer
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# Model summary
model.summary()

# Train the model
EPOCHS = 15
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS
)

# Evaluate the model
train_loss, train_acc = model.evaluate(train_generator)
test_loss, test_acc = model.evaluate(test_generator)

print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Assuming you have the true labels and predicted labels for both train and test datasets
train_predictions = model.predict(train_generator)
test_predictions = model.predict(test_generator)

# Convert predictions to class labels (if necessary, depends on your model output)
train_predictions_labels = np.argmax(train_predictions, axis=1)
test_predictions_labels = np.argmax(test_predictions, axis=1)

# Assuming your labels are categorical and represented as integers
print("Train Classification Report:")
print(classification_report(train_generator.classes, train_predictions_labels))

print("Test Classification Report:")
print(classification_report(test_generator.classes, test_predictions_labels))

# You can also extract individual metrics:
# For train
train_report = classification_report(train_generator.classes, train_predictions_labels, output_dict=True)
print(f"Train Precision: {train_report['weighted avg']['precision']:.2f}")
print(f"Train Recall: {train_report['weighted avg']['recall']:.2f}")
print(f"Train F1-Score: {train_report['weighted avg']['f1-score']:.2f}")

# For test
test_report = classification_report(test_generator.classes, test_predictions_labels, output_dict=True)
print(f"Test Precision: {test_report['weighted avg']['precision']:.2f}")
print(f"Test Recall: {test_report['weighted avg']['recall']:.2f}")
print(f"Test F1-Score: {test_report['weighted avg']['f1-score']:.2f}")


# Save the model
model.save("trained_models/convnext_fer_15_epochs.keras")
