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

```bash
pip install torch torchvision opencv-python matplotlib
