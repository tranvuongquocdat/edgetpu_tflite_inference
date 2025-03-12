#import lib and models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import yaml

#import models
from base_model.mobilenetv2 import MobileNetV2
from base_model.mobilenetv3 import MobileNetV3

# Định nghĩa kích thước đầu vào ở một nơi duy nhất
INPUT_SIZE = 384
model_type = "large"

# Custom dataset cho dữ liệu YOLO
class YOLODataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.image_files = []
        self.labels = []
        self.class_names = []
        
        # Đọc file data.yaml để lấy tên các lớp
        yaml_path = os.path.join(root_dir, 'data.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                self.class_names = data.get('names', [])
        else:
            raise FileNotFoundError(f"Không tìm thấy file data.yaml tại {yaml_path}")
        
        # Đường dẫn đến thư mục images của split tương ứng (train/test/valid)
        images_dir = os.path.join(root_dir, split, 'images')
        
        # Kiểm tra thư mục images có tồn tại không
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục {images_dir}")
            
        # Lấy danh sách tất cả các file ảnh trong thư mục
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            # Đường dẫn đầy đủ đến file ảnh
            img_path = os.path.join(images_dir, img_file)
            self.image_files.append(img_path)
            
            # Đường dẫn đến file label tương ứng
            label_file = img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            label_path = os.path.join(root_dir, split, 'labels', label_file)
            
            # Đọc labels từ file txt
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    boxes = []
                    for line in f.readlines():
                        values = line.strip().split()
                        if len(values) >= 5:  # Đảm bảo đủ thông tin (class_id, x, y, w, h)
                            class_id = int(values[0])
                            boxes.append(class_id)
                    # Lấy class_id đầu tiên làm nhãn chính (giả sử mỗi ảnh chỉ có một đối tượng chính)
                    if boxes:
                        self.labels.append(boxes[0])
                    else:
                        self.labels.append(0)  # Gán nhãn mặc định nếu không có box
            else:
                self.labels.append(0)  # Gán nhãn mặc định nếu không có file label
                print(f"Không tìm thấy file label: {label_path}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
            
        # Kiểm tra xem file có tồn tại không
        if not os.path.exists(img_path):
            print(f"Không tìm thấy file ảnh: {img_path}")
            # Tạo một ảnh trống nếu không tìm thấy file
            image = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), color='black')  # Sử dụng biến INPUT_SIZE
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Lỗi khi mở file {img_path}: {e}")
                image = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), color='black')  # Sử dụng biến INPUT_SIZE
                
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Chuẩn bị dữ liệu
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),  # Sử dụng biến INPUT_SIZE
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Chuẩn hóa ImageNet
])

# Đường dẫn đến dataset GUD
dataset_path = 'data/gud_dataset_filtered'  # Thay đổi đường dẫn này tùy theo vị trí dataset

# Tạo dataset
train_dataset = YOLODataset(root_dir=dataset_path, split='train', transform=transform)
test_dataset = YOLODataset(root_dir=dataset_path, split='test', transform=transform)
val_dataset = YOLODataset(root_dir=dataset_path, split='valid', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Khởi tạo mô hình và các tham số
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.class_names)
model = MobileNetV3(num_classes=num_classes, config=model_type).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

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
    all_labels_one_hot = np.eye(num_classes)[np.array(all_labels)]  # One-hot encoding
    map_score = np.mean([average_precision_score(all_labels_one_hot[:, i], all_preds[:, i]) for i in range(num_classes)])
    
    return accuracy, avg_loss, map_score

# Hàm hiển thị ảnh sample
def show_samples(model, test_loader, num_samples=30):
    model.eval()
    classes = test_dataset.class_names
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
    all_labels_one_hot = np.eye(num_classes)[np.array(all_labels)]
    map_score = np.mean([average_precision_score(all_labels_one_hot[:, i], all_preds[:, i]) for i in range(num_classes)])
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}, "
          f"Train Accuracy: {100 * correct / total:.2f}%, mAP@50: {map_score:.4f}")

# Đánh giá mô hình sau khi train
test_acc, test_loss, test_map = evaluate_model(model, test_loader)
print(f"\nĐánh giá trên tập test:")
print(f"Test Accuracy: {test_acc:.2f}%, Test Loss: {test_loss:.4f}, Test mAP@50: {test_map:.4f}")

# Hiển thị ảnh sample
show_samples(model, test_loader)

# Lưu mô hình
torch.save(model.state_dict(), "mobilenetv3_gud.pth")

# Lưu file labels tương tự như coco_labels.txt
with open("gud_labels.txt", "w") as f:
    for class_name in train_dataset.class_names:
        f.write(f"{class_name}\n")

print(f"Đã lưu model tại mobilenetv3_gud.pth")
print(f"Đã lưu labels tại gud_labels.txt")