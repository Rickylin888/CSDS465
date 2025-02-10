import os
from tkinter import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)
        return outputs

class InceptionCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.inception1 = InceptionBlock(64, 32)
        self.inception2 = InceptionBlock(128, 64)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"loop {epoch + 1}, loss: {running_loss / len(train_loader)}")
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"accuracy: {100 * correct / total:.2f}%")

    return model

def calculate_inception_score(model, folder_path, device='cuda', splits=10):
   
    model.eval()
    images = []

    # Preprocessing for the input images
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load images from the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('png', 'jpg', 'jpeg')):
            image = Image.open(os.path.join(folder_path, file_name)).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            images.append(image)
    
    if not images:
        raise ValueError("error")
    images = torch.cat(images, dim=0)
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch = images[i:i+32]
            preds.append(F.softmax(model(batch), dim=1).cpu().numpy())
    preds = np.vstack(preds)

    split_scores = []
    for k in range(splits):
        part = preds[k * preds.shape[0] // splits: (k + 1) * preds.shape[0] // splits]
        py = np.mean(part, axis=0) 
        scores = [np.sum(p * np.log(p / py)) for p in part]  
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)



if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InceptionCIFAR(num_classes=10)
    model = train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device=device)
    torch.save(model.state_dict(), 'inception_cifar10.pth')
    print("Model training complete and saved.")
    folder_path = './batch256-100samples'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InceptionCIFAR(num_classes=10) 
    model.load_state_dict(torch.load('inception_cifar10.pth', map_location=device))
    model.to(device)
    mean, std = calculate_inception_score(model, folder_path, device=device)
    print(f"Inception Score: {mean:.4f} ± {std:.4f}")
