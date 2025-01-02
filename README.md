# PyCoral Installation and Setup Guide

This guide provides step-by-step instructions to set up PyCoral for working with Edge TPU devices. It also covers the installation of required dependencies and managing swap memory for efficient usage on resource-limited devices.

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
Add the Coral package repository to your system’s sources list and import the GPG key.

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
```

- **Repository**: Ensures access to Coral Edge TPU packages.
- **GPG Key**: Ensures the authenticity of the repository.

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
sudo pip install numpy opencv-python-headless tqdm pyyaml
```

- **`python3-pip`**: Python package manager.
- **`ffmpeg libsm6 libxext6`**: Multimedia handling libraries.
- **`picamera2`**: Library for handling Raspberry Pi Camera.
- **`pip upgrade`**: Ensures latest version of Python tools.
- **`numpy`**: Numerical computations.
- **`opencv-python-headless`**: OpenCV without GUI support.
- **`tqdm`**: Progress bar library.
- **`pyyaml`**: YAML parsing library.

---

## 4. Manage Swap Memory

Adjusting swap memory helps avoid crashes on low-memory devices.

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

## 5. Reboot System

Reboot your device to apply all changes.

```bash
sudo reboot
```

Reconnect to your device once it restarts.

---

## 6. Clone and Set Up the Project Repository

Clone the project repository and navigate to its directory:

```bash
git clone https://github.com/tranvuongquocdat/edgetpu_tflite_inference.git
cd edgetpu_tflite_inference
```

- **`git clone`**: Downloads the project repository.
- **`cd`**: Changes the directory to the project folder.

---

## Troubleshooting

- If you face installation issues, ensure you have a stable internet connection.
- Check for typos in commands and ensure dependencies are correctly installed.
- Use logs from failed installations to debug issues.

---

## Notes

- Use `libedgetpu1-max` only if your device can handle the extra load without overheating.
- For Raspberry Pi, ensure adequate cooling if you modify swap memory or use high-performance libraries.

Happy coding!
