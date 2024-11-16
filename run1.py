import os
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
      
        self.root_dir = root_dir
        self.transform = transform
        self.classes = self._get_classes()  # Get all classes (0-9, A-Z)
        self.image_paths = self._load_image_paths()

    def _get_classes(self):
        # Include both digits 0-9 and letters A-Z
        digit_classes = [str(i) for i in range(10)]
        letter_classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        return digit_classes + letter_classes

    def _load_image_paths(self):
        image_paths = []
        labels = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append((os.path.join(class_path, img_name), self.classes.index(class_name)))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ASLNet(nn.Module):
    def __init__(self, num_classes=36):  # 10 digits + 26 letters = 36 classes
        super(ASLNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train(model, train_loader, criterion, optimizer, device, epoch, epsilon):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
          # Epsilon-greedy strategy
        
        if np.random.rand() < epsilon:
            # Explore: take a random action
            predicted = torch.randint(0, outputs.size(1), target.size()).to(device)
        else:
            # Exploit: take the best action
            _, predicted = outputs.max(1)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                  f'Loss: {running_loss/(batch_idx+1):.3f}, '
                  f'Acc: {100.*correct/total:.2f}%, Epsilon: {epsilon:.3f}')
       

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    acc = 100.*correct/total
    print(f'\nTest set: Average loss: {test_loss/len(test_loader):.4f}, '
          f'Accuracy: {correct}/{total} ({acc:.2f}%)\n')
    return acc

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='ASL Recognition')
    parser.add_argument('--data-dir', type=str, 
                      default=r'C:\Users\sriau\Downloads\PreddRNN\PredRNN\data',
                      help='data directory')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)

    parser.add_argument('--no-cuda', action='store_true', default=False)
    args = parser.parse_args()



    # Setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    full_dataset = ASLDataset(root_dir=args.data_dir, transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=2)

    # Initialize model, criterion, and optimizer
    epsilon = args.epsilon_start
    model = ASLNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_acc = 0
    for epoch in range(1, 10):
        print(f'\nEpoch: {epoch}')
        train(model, train_loader, criterion, optimizer, device, epoch, epsilon)
        acc = test(model, test_loader, criterion, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'asl_recognition_model.pth')
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)


if __name__ == '__main__':
    main()