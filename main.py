import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
from collections import OrderedDict

# ===================================================================
#                       1. Configuration
# ===================================================================

# --- Training Hyperparameters ---
NUM_EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- CNN Architecture Parameters ---
# Input image parameters (MNIST is 1 channel, 28x28 pixels)
IN_CHANNELS = 1
IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CLASSES = 10  # Number of output classes (0-9)

# Convolutional Layer Parameters
CONV_OUT_CHANNELS = 1  # Number of output channels (number of filters)
CONV_KERNEL_SIZE = 3  # 5  # Size of convolution kernel (filter)
CONV_STRIDE = 1  # Stride of convolution kernel
CONV_PADDING = (
    1  # 2  # Number of padding around the image (2 = 'same' padding for 5x5 kernel)
)


# Pooling Layer Parameters
POOL_KERNEL_SIZE = 4  # Size of pooling kernel
POOL_STRIDE = 4  # Stride of pooling kernel

# ===================================================================
#                      2. Prepare MNIST Dataset
# ===================================================================

# Image transformation pipeline
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST global mean and std
    ]
)

# Download and load dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform
)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===================================================================
#                  3. Define CNN Neural Network Model
# ===================================================================


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Define convolution and pooling layers
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=IN_CHANNELS,
                out_channels=CONV_OUT_CHANNELS,
                kernel_size=CONV_KERNEL_SIZE,
                stride=CONV_STRIDE,
                padding=CONV_PADDING,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE),
        )

        # Automatically calculate the number of features after flattening
        fc_input_features = self._get_conv_output_size()

        # Define fully connected layer (classifier)
        self.fc_layer = nn.Linear(fc_input_features, NUM_CLASSES)

    def _get_conv_output_size(self):
        """
        Helper function: Automatically calculate the flattened size of the output from the convolutional layer.
        Create a dummy input tensor, pass it through the convolutional layer, and calculate the total number of elements.
        """
        # Create a dummy tensor matching the input format
        dummy_input = torch.randn(1, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        # Pass the dummy tensor through the convolutional layer
        output = self.conv_layer(dummy_input)
        # Return the total number of elements after flattening
        return int(torch.flatten(output, 1).shape[1])

    def forward(self, x):
        # Original dimension of x: [batch_size, 1, 28, 28]

        # 1. Pass through convolution and pooling layers
        out = self.conv_layer(x)

        # 2. Flatten to 1D vector for fully connected layer
        out = torch.flatten(
            out, 1
        )  # Flatten from the 1st dimension (keep batch dimension)

        # 3. Pass through fully connected layer for classification, get logits
        out = self.fc_layer(out)

        return out


# ===================================================================
#                 4. Create Model, Loss Function, and Optimizer
# ===================================================================

print(f'Using device: {DEVICE}')
model = CNN().to(DEVICE)
print('\nModel architecture:')
print(model)

# Calculate and print total number of model parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nTotal number of model parameters: {total_params:,}')

# Define loss function (CrossEntropyLoss includes Softmax)
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===================================================================
#                          5. Train Model
# ===================================================================

total_steps = len(train_loader)
print(
    f'\nStarting training... {NUM_EPOCHS} epochs in total, {total_steps} steps per epoch.'
)

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            avg_loss = running_loss / 100
            print(
                f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{total_steps}], Average Loss: {avg_loss:.4f}'
            )
            running_loss = 0.0

print('Training complete!')

# ===================================================================
#                 6. Evaluate Model and Save Weights
# ===================================================================

model.eval()  # Switch to evaluation mode

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\nAccuracy of the model on 10000 test images: {accuracy:.2f} %')

# --- Save weights to JSON file ---
weights_dict = model.state_dict()
json_dict = OrderedDict()

# Convert Tensor to list for JSON writing
for key, tensor in weights_dict.items():
    json_dict[key] = tensor.cpu().numpy().tolist()

# Write to JSON file
json_filename = 'mnist_weights.json'
with open(json_filename, 'w') as f:
    json.dump(json_dict, f, indent=2)

print(f'Model weights successfully written to {json_filename}')
