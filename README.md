# PyCoral Installation and Setup Guide for Edge TPU Inference

This guide provides step-by-step instructions to set up PyCoral for working with Edge TPU devices. It covers installation of required dependencies, model training, and inference on Edge TPU devices.

## Prerequisites

Ensure that your system is running a Debian-based Linux distribution (e.g., Ubuntu or Raspberry Pi OS). You must have sudo privileges to perform these operations.

---

## 1. System Update and Essential Tools Installation

Update your system and install essential tools like `git`, `curl`, and `gnupg`.

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git curl gnupg
```

- **`apt-get update`**: Updates the package index.
- **`apt-get upgrade`**: Installs available upgrades for packages.
- **`git`**: Tool for cloning repositories.
- **`curl`**: Tool for transferring data from URLs.
- **`gnupg`**: Essential for managing GPG keys.

---

## 2. Install PyCoral

### Add Coral Package Repository
Add the Coral package repository to your system's sources list and import the GPG key.

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
```

### Update Package Index and Install PyCoral

```bash
sudo apt-get update
sudo apt-get install libedgetpu1-std
sudo apt-get install python3-pycoral
```

- **`libedgetpu1-std`**: Standard Edge TPU runtime library.
- **`python3-pycoral`**: Python library for Coral.

#### Optional (For Better Performance)
Install the maximum performance runtime library. This may risk device overheating.

```bash
sudo apt-get install libedgetpu1-max
```

---

## 3. Install Additional Requirements

Install additional tools and Python packages required for your project.

```bash
sudo apt-get install python3-pip ffmpeg libsm6 libxext6
pip install picamera2
sudo pip install --upgrade pip setuptools wheel
sudo pip install numpy opencv-python-headless tflite-runtime tqdm pyyaml
```

- **`python3-pip`**: Python package manager.
- **`ffmpeg libsm6 libxext6`**: Multimedia handling libraries.
- **`picamera2`**: Library for handling Raspberry Pi Camera.
- **`numpy`**: Numerical computations.
- **`opencv-python-headless`**: OpenCV without GUI support.
- **`tflite-runtime`**: TensorFlow Lite runtime for inference.
- **`tqdm`**: Progress bar library.
- **`pyyaml`**: YAML parsing library.

---

## 4. Manage Swap Memory

Adjusting swap memory helps avoid crashes on low-memory devices like Raspberry Pi.

### Check Current Swap Status

```bash
sudo swapon --show
free -h
```

### Disable Current Swap File

```bash
sudo dphys-swapfile swapoff
```

### Modify Swap File Configuration

Edit the swap configuration file to allocate more memory:

```bash
sudo nano /etc/dphys-swapfile
```

- Modify the `CONF_SWAPSIZE` value to your desired size (e.g., 2048 for 2GB).

### Reapply and Enable Swap File

```bash
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

---

## 5. Project Structure

This project is organized into two main components:

### Training Module
Located in the `/train` directory, it contains:
- `train_ssd.py`: Script for training SSD models
- `config.py`: Configuration settings for training
- `test_dataset.ipynb`: Notebook for testing dataset functionality
- `test.ipynb`: Notebook for testing trained models
- `trained_models/`: Directory for saving trained models

### Inference Module
Located in the `/inference` directory, it contains:
- `edgetpumodel.py`: Core Edge TPU model implementation
- `detect.py`: Basic detection script
- `detect_api.py`: API for detection functionality
- `nms.py`: Non-maximum suppression implementation
- `utils.py`: Utility functions
- `aicam.py` & `apicam.py`: Camera integration scripts
- `client.py` & `client2.py`: Client applications
- `install.sh`: Automated installation script

---

## 6. Training Models

To train an SSD model for object detection:

```bash
cd train
python train_ssd.py --config config.py
```

Training configuration parameters can be adjusted in `config.py` including:
- Dataset paths
- Model architecture
- Training hyperparameters
- Augmentation settings

---

## 7. Running Inference

To perform object detection using a trained model:

```bash
cd inference
python detect.py --model path/to/model.tflite --image path/to/image.jpg
```

For camera-based detection:

```bash
python aicam.py --model path/to/model.tflite
```

You can also use the API functionality:

```bash
python client.py --server_ip YOUR_SERVER_IP --server_port YOUR_SERVER_PORT
```

---

## 8. Automated Installation

For quick setup, you can use the provided installation script:

```bash
cd inference
chmod +x install.sh
./install.sh
```

---

## Troubleshooting

- If you encounter "Edge TPU is unavailable" error, check USB connections.
- For performance issues, consider using `libedgetpu1-max` but monitor device temperature.
- If training crashes on devices with limited RAM, increase swap memory.
- Ensure model compatibility with Edge TPU compiler using the official documentation.

---

## Notes

- Use `libedgetpu1-max` only if your device can handle the extra load without overheating.
- For Raspberry Pi, ensure adequate cooling if you modify swap memory or use high-performance libraries.
- Trained models need to be compiled specifically for Edge TPU using Google's Edge TPU Compiler.

Happy coding!
