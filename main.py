import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
from collections import OrderedDict

# 1. Define hyperparameters
# ---------------------------------
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)  # Check if GPU is available
input_size = 784  # Size after flattening 28x28 image
hidden_size = 2  # Hidden layer size
num_classes = 10  # Number of output classes (0-9)
num_epochs = 5  # Total number of training epochs
batch_size = 100  # Batch size for training
learning_rate = 0.001  # Learning rate

# 2. Prepare MNIST dataset
# ---------------------------------
# Use torchvision.transforms to transform images
# ToTensor() converts PIL Image or numpy.ndarray to FloatTensor and scales pixel values from [0, 255] to [0.0, 1.0]
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,)
        ),  # Normalize tensor (using global mean and std of MNIST)
    ]
)

# Download and load training dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)

# Download and load test dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform
)

# Create data loaders
# DataLoader helps with batching, shuffling, and parallel data loading
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)  # shuffle=True means shuffle data at the start of each epoch

test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)  # Shuffling is usually not needed for testing


# 3. Define neural network model
# ---------------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # First layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function
        self.relu = nn.ReLU()
        # Second layer
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Original shape of x: [batch_size, 1, 28, 28]
        # Need to flatten to [batch_size, 784]
        x = x.reshape(-1, input_size)

        # Pass through first layer and activation
        out = self.fc1(x)
        out = self.relu(out)
        # Pass through second layer to get final output (logits)
        out = self.fc2(out)
        return out


# 4. Create model, loss function, and optimizer
# ---------------------------------
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Define loss function
# CrossEntropyLoss includes Softmax, so no need to add Softmax in model output
criterion = nn.CrossEntropyLoss()

# Define optimizer
# Adam is a commonly used and effective optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 5. Train the model
# ---------------------------------
total_steps = len(train_loader)
print(
    f'Starting training... {num_epochs} epochs in total, {total_steps} steps per epoch.'
)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Move data to device (CPU or GPU)
        # images shape: [100, 1, 28, 28] -> [100, 784]
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            avg_loss = running_loss / 100
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], Average Loss: {avg_loss:.4f}'
            )
            running_loss = 0.0


print('Training complete!')

# # Save model
# model_path = 'mnist_model.pth'
# torch.save(
#     {
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'epoch': num_epochs,
#     },
#     model_path,
# )
# print(f'Model saved to {model_path}')

# Get weights dictionary
weights_dict = model.state_dict()
json_dict = OrderedDict()

# Convert tensors to lists for JSON serialization
for key, tensor in weights_dict.items():
    json_dict[key] = tensor.cpu().numpy().tolist()

# Write to JSON file
with open('mnist_model_weights.json', 'w') as f:
    json.dump(json_dict, f, indent=4)

print('Model weights successfully written to mnist_model_weights.json')

# 6. Evaluate model accuracy
# ---------------------------------
# Set model to evaluation mode
# This disables layers like Dropout and BatchNorm used only during training
model.eval()

# No need to compute gradients during evaluation to save resources
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # torch.max returns (max value, index of max value)
        # We only need the index, which is the predicted class
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on 10000 test images: {accuracy:.2f} %')

# Switch model back to training mode
model.train()
