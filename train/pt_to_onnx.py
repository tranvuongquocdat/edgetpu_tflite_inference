import torch
from base_model.mobilenetv3 import MobileNetV3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mô hình đã train
model = MobileNetV3(num_classes=10, config="small").to(device)
model.load_state_dict(torch.load("mobilenetv3.pth"))
model.eval()

# Tạo input giả để export
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Export sang ONNX
torch.onnx.export(model, dummy_input, "mobilenetv3.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})