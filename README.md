# Deep Convolutional Neural Network for Big Data Medical Image Classification

## 📌 Overview

This project focuses on applying Deep Convolutional Neural Networks (CNNs) to classify medical images at scale, with a special emphasis on white blood cell (WBC) recognition. Using GoogleNet with a Transfer Learning approach, the objective is to build an accurate and efficient model capable of distinguishing between five types of WBCs.

Modern hospitals produce vast quantities of medical images (MRI, CT, ultrasound, X-rays). Manual interpretation is slow, dependent on specialists, and subject to variability. Deep learning offers a reliable and scalable solution for automated classification.

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Context](#-context)
- [State of the Art](#-state-of-the-art)
- [Project Objective](#-project-objective)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Architecture](#-model-architecture)
- [Transfer Learning Strategy](#-transfer-learning-strategy)
- [References](#-references)

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

## 🗂 Dataset: KRD-WBC

The KRD-WBC dataset contains:

- **600 RGB images** (512 × 512 pixels)
- Each image includes a corresponding **ground truth mask**
- Collected from **Nanakali Hospital** and **Bio Lab** (Erbil, Kurdistan, Iraq)

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

GoogleNet (Inception v1) was chosen for:

- high performance on image classification tasks  
- efficient architecture suitable for big data  
- strong feature extraction capabilities  
- compatibility with transfer learning  

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

