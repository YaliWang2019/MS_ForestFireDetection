import os
import cv2
import csv
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 4))
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3))
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=2, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.fc_output_shape = 2 * 8 * 8
        
        self.fc1 = nn.Linear(self.fc_output_shape, 25)  # Adjusted to 25 to match the input of fc2
        self.fc2 = nn.Linear(25, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Function to process the image
def process_image(img_path, image_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img_tensor

# Function to predict the class of the image
def predict(model, img_tensor, device):
    img_tensor = img_tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        prediction = torch.round(output).item()
    return 'Fire' if prediction == 1 else 'No Fire'

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel() # Create an instance of the model
model.load_state_dict(torch.load('pytorch_cnn_sig1.pth', map_location=device))
model.to(device)

# Define the directory path and image size
dir_path = "../train420"
image_size = (254, 254)

def foo():
    # Open the csv file
    with open('pytorch_cnn_Sig1_420_pred1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["filename", "prediction", "time"])
        # Iterate over all files in the directory
        for filename in os.listdir(dir_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Full path
                full_path = os.path.join(dir_path, filename)
                img_tensor = process_image(full_path, image_size)
                
                # Make a prediction
                class_label_prediction = predict(model, img_tensor, device)
                
                print(f"{filename} has {class_label_prediction}.")
                # Write the data to the csv file
                writer.writerow([filename, class_label_prediction])
# Start timing
start_time = time.time()
foo()
# End timing
end_time = time.time()
# Calculate the time difference
time_diff = end_time - start_time
print(f"Time consumed: {time_diff} seconds.")
