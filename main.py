import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


 # 1. Settings
 
batch_size = 64
learning_rate = 0.001
epochs = 5


 # 2. Prepare Fashion-MNIST dataset
 
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


 # 3. Create DataLoaders
 
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


 # 4. Class names
 
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


 # 5. Define neural network
 
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(

            nn.Linear(28 * 28, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):

        x = self.flatten(x)

        x = self.network(x)

        return x


 # 6. Create model
 
model = NeuralNetwork()

print(model)


 # 7. Loss function and optimizer
 
loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)


 # 8. Train model
 
train_losses = []

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = (
        running_loss /
        len(train_loader)
    )

    train_losses.append(
        average_loss
    )

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {average_loss:.4f}"
    )


 # 9. Test model
 
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


accuracy = 100 * correct / total

print(
    f"\nTest Accuracy: {accuracy:.2f}%"
)


 # 10. Plot training loss
 
plt.figure()

plt.plot(
    range(1, epochs + 1),
    train_losses,
    marker="o"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training Loss"
)

plt.tight_layout()

plt.savefig(
    "training_loss.png"
)

plt.show()


 # 11. Show prediction examples
 
images, labels = next(
    iter(test_loader)
)

outputs = model(images)

_, predictions = torch.max(
    outputs,
    1
)


plt.figure(figsize=(10, 5))

for i in range(6):

    plt.subplot(
        2,
        3,
        i + 1
    )

    plt.imshow(
        images[i].squeeze(),
        cmap="gray"
    )

    plt.title(
        f"True: {class_names[labels[i]]}\n"
        f"Pred: {class_names[predictions[i]]}"
    )

    plt.axis("off")


plt.tight_layout()

plt.savefig(
    "prediction_examples.png"
)

plt.show()
