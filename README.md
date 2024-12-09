# Image Segmentation with DeepLabV3

## Segmenting People in Images

This project demonstrates how to use PyTorch's DeepLabV3 model to segment images and isolate the areas containing people. The segmented results can be visualized or processed further, making this tool suitable for applications in image analysis, object detection, and more.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Example Results](#example-results)
6. [License](#license)

---

## Introduction

In this project, we leverage the power of DeepLabV3 with a ResNet-101 backbone to detect and segment people in images. This pipeline:

- **Loads a local image file** for processing.
- **Applies segmentation** using a pre-trained model.
- **Masks the segmented regions**, highlighting the detected person(s).
- **Visualizes the results** in an easy-to-understand format.

---

## Features

- **State-of-the-art Model**: Uses DeepLabV3 for high-quality image segmentation.
- **Customizable**: Adaptable to different classes beyond just people.
- **GPU Support**: Leverages CUDA for faster inference (if available).
- **Simple Integration**: Lightweight and easy to incorporate into existing projects.

---

## Installation

To set up the environment and install the required dependencies, run the following command:

bash
pip install torch torchvision opencv-python matplotlib

---

## Usage
1. Clone the repository and navigate to the project directory:
git clone https://github.com/zip-sa/Image-segmentation_DeepLabV3.git
cd Image-segmentation_DeepLabV3

2. Replace the image_path in the script with the path to your desired image:
image_path = "path/to/your/image.jpg"  # Specify the path to your local image.

3. Run the script:
python segment.py

4. View the results:
The original image and the segmented output will be displayed side by side.

---

## Example Results

Input Image

Segmented Image

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

