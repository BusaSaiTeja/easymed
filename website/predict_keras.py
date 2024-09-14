from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved Keras model
model = load_model('C:\\Users\\Saite\\OneDrive\\Coding Folder\\Hackathon\\SIH_2024\\keras_model.h5')

# Preprocess the input image to match the model's expected format
def preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)  # Load and resize the image
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model's input shape
    img_array /= 255.0  # Normalize the pixel values
    return img_array

# Define the class labels (update this according to your model's output)
class_labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']  # Replace with actual labels

# Path to the image you want to predict
image_path = 'C:\\Users\\Saite\\OneDrive\\Coding Folder\\Hackathon\\SIH_2024\\website\\static\\images\\eye1.jpeg'

# Preprocess the image
preprocessed_image = preprocess_image(image_path, target_size=(224, 224))  # Update size based on model input

# Make a prediction
predictions = model.predict(preprocessed_image)

# Get the predicted class index
predicted_class = np.argmax(predictions, axis=-1)

# Get the human-readable label for the predicted class
predicted_label = class_labels[predicted_class[0]]

# Output the prediction
print(f"Predicted class: {predicted_label}")
