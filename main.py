# Import Libraries
import os
import random
from PIL import Image
from IPython.display import display
import torch

# Finding File Extensions
def get_file_extensions(folder_path):
    # List all files in the specified folder
    files = os.listdir(folder_path)

    # Extract file extensions using set comprehension
    file_extensions = {os.path.splitext(file)[1] for file in files}
    return file_extensions

# Specify the directory containing images
base_path = r'images\images\images'

# Retrieve file extensions in the specified directory
#kaggle_file_extensions = get_file_extensions(base_path)

# Print the file extensions
#print("File extensions in folder:")
#print(kaggle_file_extensions)

# List all images in the directory
all_images = os.listdir(base_path)

# Choose a random image from the list
random_image = random.choice(all_images)

# Open and display the chosen image
img = Image.open(os.path.join(base_path, random_image))
display(img)# Get all images in the directory
all_images = os.listdir(base_path)

# Choose a random image
random_image = random.choice(all_images)

# Display the image
img = Image.open(os.path.join(base_path, random_image))
display(img)

def detect_objects(model, image_paths):
    # Perform object detection on the specified images
    results = model(image_paths)
    
    # Print and visualize the detection results
    results.print()
    results.show()
    
    return results

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Use the chosen image as input
imgs = img  # batch of images

# Perform inference using the YOLOv5 model
results = model(imgs)

# Perform object detection on the image passed as argument using the YOLOv5 model

detection_results = detect_objects(model, imgs)

# List of image paths for further object detection
image_paths = [
    f"{base_path}/1478020515199458307.jpg",
    f"{base_path}/1478020231691535596.jpg",
    f"{base_path}/1478020351195471769.jpg",
    f"{base_path}/1478898499983147215.jpg",
    f"{base_path}/1478898651375864863.jpg",
    f"{base_path}/1479506165491761103.jpg",
    f"{base_path}/1478898957016224931.jpg",
    f"{base_path}/th1.jpg",
    f"{base_path}/GRdCC.jpg"
]
# Perform object detection on a list of specified images using the YOLOv5 model
results = detect_objects(model, image_paths)

# Extract object detection results as a Pandas DataFrame
data_frame = results.pandas().xyxy[0]

# Print the object detection results
print("Object Detection Results:")
print(data_frame)

