import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the model
loaded_model = load_model('../CURVES_MODELS/curves_made_model.h5')


# Load label encoder
# Assuming you saved and reloaded the label encoder separately
# If you didn’t save it, you need to recreate it with the same classes
shape_names = ['line', 'circle', 'ellipse', 'rectangle', 'rounded_rectangle', 'regular_polygon', 'star']
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(shape_names)

# Function to preprocess images
def preprocess_image(image_file_path, target_size=(128, 128)):
    grayscale_image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
    if grayscale_image is not None:
        resized_image = cv2.resize(grayscale_image, target_size)
        normalized_image = resized_image.reshape((1, 128, 128, 1)) / 255.0
    return normalized_image

# Function to predict image class
def predict_image_class(model, image_file_path, encoder):
    processed_image = preprocess_image(image_file_path)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = encoder.inverse_transform([predicted_class_index])
    return predicted_class[0]

# Function to display image with prediction
def show_image_with_prediction(image_file_path, predicted_class):
    image = cv2.imread(image_file_path)
    cv2.imshow(f'Predicted Class: {predicted_class}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test with user-provided image
test_image_path = '../problems/occlusion2_sol_rec.png'  # Replace with the path to the user image
predicted_class = predict_image_class(loaded_model, test_image_path, label_encoder)
print(f'Predicted Class: {predicted_class}')
show_image_with_prediction(test_image_path, predicted_class)
