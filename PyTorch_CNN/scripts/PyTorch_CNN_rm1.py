import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

image_width_in_pixels = 254
image_height_in_pixels = 254

# Define transforms
transform = transforms.Compose([
    transforms.Resize((image_width_in_pixels, image_height_in_pixels)),
    transforms.ToTensor()
])

# Load the dataset using ImageFolder
full_dataset = ImageFolder('C:/Users/nicol/Downloads/training', transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = int(0.10 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, validation_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class TrainingDataset(Dataset):
    def __init__(self, tf_dataset):
        images_list, labels_list = [], []
        for image, label in tf_dataset:
            images_list.append(image.numpy())
            labels_list.append(label.numpy())
        
        # Concatenate along the first dimension to flatten out batches
        self.images = torch.tensor(np.concatenate(images_list, axis=0), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        self.labels = torch.tensor(np.concatenate(labels_list, axis=0), dtype=torch.int64).squeeze()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 4))
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3))
        
        # Third convolutional layer
        #self.conv3 = nn.Conv2d(in_channels=5, out_channels=2, kernel_size=3)
        #self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        #self.fc1 = nn.Linear(128, 100)
        self.fc1 = nn.Linear(1805, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        #x = self.pool3(self.relu(self.conv3(x)))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x.squeeze()  # To match the shape expected by binary cross entropy loss

model = SimpleCNN()
if torch.cuda.is_available():
    model = model.cuda()

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Changed loss function to match the binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Changed optimizer to Adam
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
# Train the model

color_channel = 3
image_size = (image_width_in_pixels, image_height_in_pixels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

best_loss = float('inf')
early_stopping_epochs = 3  # stop if no improvement after 3 epochs
no_improvement_epochs = 0
for epoch in range(10):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
            
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.float()) 
        loss.backward()
        optimizer.step()
            
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
# Check early stopping criteria
    epoch_loss = running_loss / len(train_loader)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1

    if no_improvement_epochs >= early_stopping_epochs:
        print("Early stopping triggered!")
        break

    # Adjust learning rate based on epoch_loss
    scheduler.step(epoch_loss)
print('Finished Training')

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        outputs = model(images)
        predicted = (outputs.data > 0.5).float()

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy = {correct / total}')

# Save the trained model
torch.save(model.state_dict(), 'C:/Users/nicol/Downloads/pytorch_cnn_rm1.pth')
