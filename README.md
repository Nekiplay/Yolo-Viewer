# 🚀 YOLO v11-12 Viewer (Rust)

![License](https://img.shields.io/github/license/Nekiplay/Yolo-Viewer)
![Rust](https://img.shields.io/badge/language-Rust-orange.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11%20%7C%20v12-green)
![Performance](https://img.shields.io/badge/performance-high-brightgreen)

A high-performance, lightweight graphical interface for visualizing and testing **YOLO v11** and **v12** models. Built with **Rust** for maximum efficiency, safety, and near-zero overhead during inference.

<img width="100%" alt="YOLO Viewer Interface" src="https://github.com/user-attachments/assets/5ffeb5cf-166e-455d-a09b-7cdbef130938" />

## ✨ Key Features

*   **Blazing Fast Inference:** Leverages Rust's performance to handle high-resolution video streams with minimal latency.
*   **Latest YOLO Support:** Native compatibility with YOLOv11 and the cutting-edge YOLOv12 architectures.
*   **Real-time Interaction:** Adjust Confidence and IoU thresholds on the fly using interactive sliders without restarting the stream.
*   **Hardware Acceleration:** Support for CUDA/TensorRT (NVIDIA) and CoreML (Apple) via specialized backends.
*   **Versatile Inputs:** Seamlessly switch between static images, local video files, and live RTSP/Webcam feeds.
*   **Memory Efficient:** Significantly lower RAM and CPU usage compared to traditional Python-based viewers.

## 🛠 Tech Stack

*   **Core:** [Rust](https://www.rust-lang.org/)
*   **Inference Engine:** [ONNX Runtime (ort)](https://github.com/pykeio/ort)
*   **GUI Framework:** [egui](https://github.com/emilk/egui)

## 🚀 Quick Start

### Prerequisites
Make sure you have the [Rust toolchain](https://rustup.rs/) installed.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/yolo-v11-12-viewer.git
   cd yolo-v11-12-viewer
