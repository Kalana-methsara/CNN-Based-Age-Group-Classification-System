# 👁️ FairVision AI: Convolutional Neural Network-Based Age Group Classification System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://cnn-based-age-group-classification-system-kalanamethsara.streamlit.app/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?logo=pytorch)](https://pytorch.org/)

FairVision AI is a production-ready, Deep Learning-powered computer vision framework tailored for age demographics estimation. This project addresses the critical challenge of biased facial classification by implementing an optimized Convolutional Neural Network architecture built on PyTorch, complete with a modern, glassmorphic Streamlit analytics dashboard.

### 🌐 Live Production Deployment
🚀 **Experience the live dashboard here:** [https://cnn-based-age-group-classification-system-kalanamethsara.streamlit.app/](https://cnn-based-age-group-classification-system-kalanamethsara.streamlit.app/)

---

## 🚀 Key Features
* **Deep Neural Architecture:** Built on top of a fine-tuned ResNet-50 backbone specifically optimized for facial structural mapping.
* **Anti-Light Mode Cyber-Dark UI:** Fully customized Streamlit dashboard utilizing strict custom CSS injections to ensure a sleek dark theme regardless of user system-level preferences.
* **Font Awesome & Vector Core:** Enhanced UI telemetry headings using Font Awesome v6 crisp vector icons.
* **Automated Weight Provisioning:** Built-in telemetry that automatically handles model downloading (`FairVision.pt`) securely from Google Drive if local weights are missing.
* **Granular Statistical Spectrum:** Extracts and displays top-3 probability indices alongside full probability spectrum bar charts.

---

## 📊 Technical Methodology & Backbone
The underlying system adapts a **ResNet-50 (Residual Network)** architecture to bypass vanishing gradient bottlenecks during training. The final fully connected layer is structured with a Dropout value of `0.6` to robustly mitigate overfitting, channeling outputs directly into 9 targeted age segments.

### Target Cohorts Mapping:
| Class Index | Age Group Range (Years) |
| :---: | :---: |
| 0 | 0 – 2 |
| 1 | 3 – 9 |
| 2 | 10 – 19 |
| 3 | 20 – 29 |
| 4 | 30 – 39 |
| 5 | 40 – 49 |
| 6 | 50 – 59 |
| 7 | 60 – 69 |
| 8 | 70+ |

---

## 🛠️ Local Installation & Execution Guide

Follow these steps to deploy and run the app on your local workstation:

### 1. Clone the Space
```bash
git clone [https://github.com/Kalana-methsara/CNN-Based-Age-Group-Classification-System.git](https://github.com/Kalana-methsara/CNN-Based-Age-Group-Classification-System.git)
cd CNN-Based-Age-Group-Classification-System