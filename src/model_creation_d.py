import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Function to load images from a directory
def load_images_from_directory(directory):
    images = []
    image_labels = []
    shape_names = ['circle', 'kite', 'parallelogram', 'rectangle', 'rhombus', 'square', 'trapezoid', 'triangle']
    
    
    for shape_name in shape_names:
        shape_directory = os.path.join(directory, shape_name)
        for file_name in os.listdir(shape_directory):
            image_path = os.path.join(shape_directory, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (128, 128))
                images.append(image)
                image_labels.append(shape_name)
    
    return np.array(images), np.array(image_labels)

# Load training dataset
training_data_directory = '../dataset/train'  # Path to the training dataset
train_images, train_labels = load_images_from_directory(training_data_directory)

# Preprocess training data
train_images = train_images.reshape((train_images.shape[0], 128, 128, 1)) / 255.0
label_encoder = LabelEncoder()
train_labels = to_categorical(label_encoder.fit_transform(train_labels))

# Load testing dataset
testing_data_directory = '../dataset/test'  # Path to the testing dataset
test_images, test_labels = load_images_from_directory(testing_data_directory)

# Preprocess testing data
test_images = test_images.reshape((test_images.shape[0], 128, 128, 1)) / 255.0
test_labels = to_categorical(label_encoder.transform(test_labels))

# Build the model
shape_classifier_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
shape_classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
shape_classifier_model.fit(train_images, train_labels, epochs=20, validation_split=0.2, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = shape_classifier_model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
shape_classifier_model.save('../CURVES_MODELS/curves_model.h5')
