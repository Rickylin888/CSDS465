import os
import argparse
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
os.makedirs('model', exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#Load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
#download CIFAR10 if CIfar10 is not exist
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
#Using four Fully Connected Neural Network
class fullyConnected(nn.Module):
    def __init__(self):
        super(fullyConnected, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1024)  
        self.fc2 = nn.Linear(1024, 512)          
        self.fc3 = nn.Linear(512, 256)           
        self.fc4 = nn.Linear(256, 10) 
        #Drop out optimization          
        self.dropout = nn.Dropout(0.5)           
    
    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

#Initialize network
net = fullyConnected().to(device)
#Loss Function
criterion = nn.CrossEntropyLoss()
#L2 Function
def l2Regularization(net, lambda_):
    penalty = 0.0
    for param in net.parameters():
        penalty += torch.sum(param ** 2)
    return lambda_* penalty
#Gradient Descent Function
def gradientDescent(net, lr, lambda_):
    with torch.no_grad():
        for param in net.parameters():
            param -= lr * (param.grad + lambda_ * param)
            param.grad = None

#Train the Model 
def train(net, train_loader, test_loader, criterion, lr=0.01, lambda_=0.0005, loops=10):
    print("Starting Training...\n")
    print(f"{'loops':<10}{'train Loss':<15}{'train Accuracy%':<20}{'test Loss':<15}{'test Accuracy%':<20}")
    print(' ' * 100)
    for loop in range(loops):
        net.train()  
        runningLoss = 0.0
        correctTrain = 0
        totalTrain = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            penalty = l2Regularization(net, lambda_)
            totalLoss = loss + penalty
            totalLoss.backward()
            gradientDescent(net, lr, lambda_)
            runningLoss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            totalTrain += labels.size(0)
            correctTrain += (predicted == labels).sum().item()
            
        #Calculate train loss and accuracy
        trainLoss = runningLoss / len(train_loader)
        trainAccuracy = 100 * correctTrain / totalTrain
        testLoss, testAccuracy = evaluate(net, test_loader)
        print(f"{loop+1:<10}{trainLoss:<15.4f}{trainAccuracy:<20.2f}{testLoss:<15.4f}{testAccuracy:<20.2f}")
    #Save the trained model
    torch.save(net.state_dict(), 'model/model.ckpt')
    print('Model saved in file: ./model/model.ckpt')

#Evaluate the Model
def evaluate(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    runningLoss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            runningLoss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    testLoss = runningLoss / len(test_loader)
    testAccuracy = 100 * (correct / total)
    return testLoss, testAccuracy

#Test the Model
def test_image(image_path):
    #Load the model
    net = fullyConnected().to(device)
    net.load_state_dict(torch.load('model/model.ckpt'))
    net.eval()
    transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    #predict the image
    with torch.no_grad():
        outputs = net(image)
        _, predicted = torch.max(outputs, 1)

    print(f"Predicted class for {image_path}: {classes[predicted.item()]}")

#First cd to the python file directory, then type python classify.py train to train the model the model will be saved under a directory
#Called model. Then run the python classify.py test "...\csds465HW1\a.png" to test the image
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test the CIFAR-10 classifier.')
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('image_path', nargs='?')
    args = parser.parse_args()
    if args.mode == 'train':
        train(net, train_loader, test_loader, criterion)
    elif args.mode == 'test':
        if args.image_path:
            test_image(args.image_path)
        else:
            print("Error")
    else:
        print("Invalid")
