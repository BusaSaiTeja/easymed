import torch
import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
import config
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Hardcoded paths
IMAGE_PATH = "website\\static\\images\\eye1.jpeg"
CHECKPOINT_PATH = "C:\\Users\\Saite\\OneDrive\\Coding Folder\\Hackathon\\SIH_2024\\website\\b3_2.pth.tar"

def load_checkpoint(checkpoint, model, optimizer=None, lr=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
    if optimizer is not None and lr is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

def load_model(checkpoint_path, device):
    # Define the model architecture
    model = EfficientNet.from_name("efficientnet-b3")
    model._fc = torch.nn.Linear(1536, 5)  # Adjust output layer for 5 classes
    model = model.to(device)

    # Load the model checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_checkpoint(checkpoint, model)  # No need for optimizer or lr here
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image_path):
    # Define the transformations
    transform = A.Compose([
        A.Resize(height=120, width=120),
        A.Normalize(
            mean=[0.3199, 0.2240, 0.1609],
            std=[0.3020, 0.2183, 0.1741],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image=np.array(image))  # Convert PIL image to numpy array for albumentations
    image = image['image'].unsqueeze(0)  # Add batch dimension
    return image

def make_prediction(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        # Get the class with the highest score
        prediction = outputs.argmax(dim=1)
    return prediction.item()

def main():
    device = 'cpu' # e.g., 'cuda' or 'cpu'

    # Load the model
    model = load_model(CHECKPOINT_PATH, device)

    # Preprocess the image
    image_tensor = preprocess_image(IMAGE_PATH)

    # Make a prediction
    prediction = make_prediction(model, image_tensor, device)

    # Print the prediction
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
