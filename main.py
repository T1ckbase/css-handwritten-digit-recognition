import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
from collections import OrderedDict


# 1. Configuration

# --- Training hyperparameters ---
NUM_EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Neural Network parameters ---
INPUT_SIZE = 28 * 28  # Each MNIST image is 28x28 pixels
NUM_CLASSES = 10  # Number of output classes (digits 0-9)


# 2. Prepare MNIST Dataset

# Image transformation pipeline (flatten images)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),  # Global mean and std for MNIST
        transforms.Lambda(lambda x: (x > 0.5).float()),  # Binarize pixels to 0 or 1
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten the image to 784-dim vector
    ]
)

# Download and load datasets
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform
)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 3. Define Neural Network


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(INPUT_SIZE, NUM_CLASSES)

    def forward(self, x):
        # x shape: [batch_size, 784]
        out = self.fc(x)
        return out


# 4. Create Model, Loss, Optimizer

print(f'Using device: {DEVICE}')
model = SimpleNN().to(DEVICE)
print('\nModel architecture:')
print(model)

# Print total number of trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nTotal number of parameters: {total_params:,}')

# Define loss function (CrossEntropyLoss includes Softmax)
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 5. Train Model

total_steps = len(train_loader)
print(f'\nStart training... {NUM_EPOCHS} epochs, {total_steps} steps per epoch.')

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
                f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{total_steps}], Avg Loss: {avg_loss:.4f}'
            )
            running_loss = 0.0

print('Training finished!')

# Print model output for all-zero input
model.eval()
with torch.no_grad():
    zero_input = torch.zeros(1, INPUT_SIZE).to(DEVICE)
    zero_output = model(zero_input)
    print('\nModel output (logits) for all-zero input:')
    print(zero_output.cpu().numpy())

# Print model output for all-one input
model.eval()
with torch.no_grad():
    one_input = torch.ones(1, INPUT_SIZE).to(DEVICE)
    one_output = model(one_input)
    print('\nModel output (logits) for all-one input:')
    print(one_output.cpu().numpy())

# Print model output for half-one input
model.eval()
with torch.no_grad():
    half_one_input = torch.zeros(1, INPUT_SIZE).to(DEVICE)
    half_one_input[:, :INPUT_SIZE // 2] = 1.0
    half_one_output = model(half_one_input)
    print('\nModel output (logits) for half-one input:')
    print(half_one_output.cpu().numpy())

# 6. Evaluate and Save Model

model.eval()  # Set to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    sample_logits_printed = False
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print sample raw logits (before softmax)
        if not sample_logits_printed:
            print('\nSample test data raw logits:')
            print(outputs[0].cpu().numpy())
            print(f'True label: {labels[0].item()}')
            sample_logits_printed = True

    accuracy = 100 * correct / total
    print(f'\nTest accuracy on 10000 images: {accuracy:.2f} %')


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
