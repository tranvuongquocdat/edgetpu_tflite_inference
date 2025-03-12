#import lib and models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

#import models
from base_model.mobilenetv2 import MobileNetV2
from base_model.mobilenetv3 import MobileNetV3


# Chuẩn bị dữ liệu
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Khởi tạo mô hình và các tham số
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV3(num_classes=10, config="small").to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Hàm đánh giá mô hình
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Thu thập dự đoán và nhãn để tính mAP
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = test_loss / len(test_loader)
    
    # Tính mAP@50 (giả lập cho classification)
    all_preds = np.array(all_preds)
    all_labels_one_hot = np.eye(10)[np.array(all_labels)]  # One-hot encoding
    map_score = np.mean([average_precision_score(all_labels_one_hot[:, i], all_preds[:, i]) for i in range(10)])
    
    return accuracy, avg_loss, map_score

# Hàm hiển thị ảnh sample
def show_samples(model, test_loader, num_samples=5):
    model.eval()
    classes = test_dataset.classes
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    images = images.cpu().numpy()
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        img = np.transpose(images[i], (1, 2, 0)) * 0.5 + 0.5  # Denormalize
        axes[i].imshow(img)
        axes[i].set_title(f"Pred: {classes[preds[i]]}\nTrue: {classes[labels[i]]}")
        axes[i].axis('off')
    plt.show()

# Huấn luyện với progress bar và thông số
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Progress bar
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Thu thập dự đoán để tính mAP
        probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        
        # Cập nhật progress bar
        train_bar.set_postfix({'loss': running_loss / (train_bar.n + 1), 'acc': 100 * correct / total})
    
    # Tính mAP@50 cho epoch
    all_preds = np.array(all_preds)
    all_labels_one_hot = np.eye(10)[np.array(all_labels)]
    map_score = np.mean([average_precision_score(all_labels_one_hot[:, i], all_preds[:, i]) for i in range(10)])
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}, "
          f"Train Accuracy: {100 * correct / total:.2f}%, mAP@50: {map_score:.4f}")

# Đánh giá mô hình sau khi train
test_acc, test_loss, test_map = evaluate_model(model, test_loader)
print(f"\nĐánh giá trên tập test:")
print(f"Test Accuracy: {test_acc:.2f}%, Test Loss: {test_loss:.4f}, Test mAP@50: {test_map:.4f}")

# Hiển thị ảnh sample
show_samples(model, test_loader)

# Lưu mô hình
torch.save(model.state_dict(), "mobilenetv3.pth")