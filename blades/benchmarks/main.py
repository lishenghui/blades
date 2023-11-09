import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from fllib.models.catalog import ModelCatalog

model = ModelCatalog.get_model("resnet")
data_root = "./data"
train_set = torchvision.datasets.CIFAR10(train=True, download=True, root=data_root)
test_set = torchvision.datasets.CIFAR10(train=False, download=True, root=data_root)


# Assuming `ModelCatalog.get_model("resnet")` returns a ResNet18 model.
# If not, use torchvision.models.resnet18 to get it.
# model = torchvision.models.resnet18(pretrained=False)

# Adjust the final layer to match the number of classes in CIFAR10 (10 classes).
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 10)

# Check for GPU availability and move the model to GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 1. Define transformations for the dataset
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# Apply transformations to datasets.
train_set.transform = transform_train
test_set.transform = transform_test

# 2. Create data loaders.
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=128, shuffle=True, num_workers=4
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=128, shuffle=False, num_workers=4
)

# 3. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Adjust learning rate during training (optional)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# 4. Train the model
num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader)}")
    scheduler.step()

# Optionally, 5. Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f"Accuracy on test set: {100. * correct / total}%")
