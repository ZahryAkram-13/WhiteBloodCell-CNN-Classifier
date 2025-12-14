# Deep Convolutional Neural Network for Big Data Medical Image Classification

## 📌 Overview

This project focuses on applying Deep Convolutional Neural Networks (CNNs) to classify medical images at scale, with a special emphasis on white blood cell (WBC) recognition. Using GoogleNet with a Transfer Learning approach, the objective is to build an accurate and efficient model capable of distinguishing between five types of WBCs.

Modern hospitals produce vast quantities of medical images (MRI, CT, ultrasound, X-rays). Manual interpretation is slow, dependent on specialists, and subject to variability. Deep learning offers a reliable and scalable solution for automated classification.

---

## Installation and Setup

This project leverages AMD hardware acceleration via ROCm for GPU compute. It was developed and run on a system featuring an AMD Ryzen 7 6000 series CPU and an AMD Radeon RX 6650M XT GPU. The environment is configured with PyTorch and ROCm-compatible libraries to fully utilize this AMD hardware stack for model training and inference.

This project requires Python packages listed in `requirements.txt` and, optionally, a full conda environment via `environment.yml` for GPU/ROCm support.

### Using conda - Recommended:

```bash

pip update
pip install --upgrade pip

conda env create -f environment.yml
conda activate rocm

pip install -r requirements.txt

```

---

## 📚 Table of Contents

- [Deep Convolutional Neural Network for Big Data Medical Image Classification](#deep-convolutional-neural-network-for-big-data-medical-image-classification)
  - [📌 Overview](#-overview)
  - [Installation and Setup](#installation-and-setup)
    - [Using conda - Recommended:](#using-conda---recommended)
  - [📚 Table of Contents](#-table-of-contents)
  - [🩺 Context](#-context)
  - [🧠 State of the Art](#-state-of-the-art)
    - [Classical Approaches](#classical-approaches)
    - [Deep Learning Approaches](#deep-learning-approaches)
  - [🎯 Project Objective](#-project-objective)
  - [🗂 Dataset: White Blood Cells Dataset from Kaggle](#-dataset-white-blood-cells-dataset-from-kaggle)
    - [Data Acquisition](#data-acquisition)
    - [Classes Included](#classes-included)
  - [🛠 Methodology](#-methodology)
  - [🏗 Model Architecture](#-model-architecture)
  - [🔁 Transfer Learning Strategy](#-transfer-learning-strategy)
  - [📖 References](#-references)

---

## 🩺 Context

Medical institutions generate massive amounts of imaging data daily. Manual analysis presents several limitations:

- significant processing time  
- heavy reliance on expert knowledge  
- strong inter-operator variability  

Deep learning, particularly CNNs, enables the automatic extraction of relevant features and greatly improves classification performance in medical imaging.

---

## 🧠 State of the Art

### Classical Approaches
- Manual feature extraction (shape, texture, color)
- Low robustness and limited accuracy
- Sensitive to noise and variability

### Deep Learning Approaches
- Convolutional Neural Networks (CNNs) learn hierarchical features automatically
- Strong performance across medical imaging tasks
- Pre-trained models (GoogleNet, ResNet, etc.) yield high accuracy even with smaller datasets

---

## 🎯 Project Objective

The goal of this project is to automatically classify five types of white blood cells using the KRD-WBC dataset:

- Neutrophils  
- Lymphocytes  
- Monocytes  
- Eosinophils  
- Basophils  

To achieve this, the project uses GoogleNet as a pre-trained backbone and fine-tunes it for 5-class classification.

---

## 🗂 Dataset: White Blood Cells Dataset from Kaggle

For this project, we utilized a large, publicly available dataset of white blood cells (WBCs) sourced from Kaggle.

Dataset Source & URL: The dataset, titled "White blood cells dataset," was created by Masoud Nickparvar and is hosted on Kaggle. You can access it directly via this link: https://www.kaggle.com/datasets/masoudnickparvar/white-blood-cells-dataset?select=Train.

Class Distribution: The dataset reflects the natural distribution found in blood, which is an important characteristic for model training. The breakdown is as follows:

- Neutrophil: 6,231 images (~60%)
- Lymphocyte: 2,427 images (~24%)
- Eosinophil: 744 images (~7%)
- Monocyte: 561 images (~5%)
- Basophil: 212 images (~2%)


### Data Acquisition
- Olympus BX51 microscope  
- Basler high-resolution camera  
- ×1000 magnification  
- 42 px/µm resolution  

Annotations were manually created and validated by medical experts.

### Classes Included
- Neutrophils  
- Lymphocytes  
- Monocytes  
- Eosinophils  
- Basophils  

---

## 🛠 Methodology

The overall workflow includes:

1. Loading the WBC dataset  
2. Preprocessing and normalization  
3. Importing GoogleNet pre-trained on ImageNet  
4. Freezing early layers to preserve general visual features  
5. Replacing the final layer with a 5-output classifier  
6. Fine-tuning on the KRD-WBC dataset  
7. Evaluating model performance  

---

## 🏗 Model Architecture

Multiple CNN architectures were implemented and compared:

- MobileNet v2/v3: Lightweight models for mobile/edge deployment
- EfficientNet B0: Balanced accuracy and computational efficiency
- SqueezeNet: Extremely compact architecture with AlexNet-level accuracy
- ShuffleNet v2: Highly efficient model with channel shuffling operations
- ResNet18: Standard baseline with residual connections

---

## 🔁 Transfer Learning Strategy

The model training strategy follows these steps:

1. Load GoogleNet with ImageNet weights 
2. Freeze the initial convolutional layers  
3. Unfreeze the upper layers for specialization  
4. Replace the final fully connected layer with a 5-class head  
5. Train the model using the WBC dataset  
6. Validate and evaluate results  

This approach significantly reduces training time while improving accuracy on small/medium medical datasets.

---

## 📖 References

This project is supported by research contributions in areas such as:

- Deep learning for medical image classification  
- Transfer learning and fine-tuning techniques  
- High-resolution WBC datasets  
- Multimodal and transformer-based imaging models  

For full citations, refer to the detailed reference list in the project presentation.

---

