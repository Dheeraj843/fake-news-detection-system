# 📰 Explainable Fake News Detection System

A robust, hybrid deep learning framework built to classify news articles as **Real** or **Fake**. This system leverages the deep semantic understanding of **DistilBERT** paired with a **1D-CNN** head for high-accuracy local feature extraction. To ensure transparency, the framework integrates Explainable AI (XAI) tools (**LIME** & **SHAP**) to interpret and visually explain the model's classification decisions.

## 🚀 Tech Stack & Frameworks
* **Language:** Python
* **Core Deep Learning:** TensorFlow / Keras
* **NLP Backbone:** HuggingFace Transformers (DistilBERT)
* **Explainability (XAI):** LIME, SHAP
* **Data Processing & ML Pipelines:** Scikit-learn, Pandas, NumPy

## 📊 Core Architecture & Features
* **Hybrid Classifier:** Passes contextual token embeddings through an optimized 1D Convolutional Neural Network layer with Global Max Pooling to capture local textual patterns, achieving over 85% accuracy.
* **Dual-Layer Interpretability:**
  * **LIME Integration:** Generates localized model-agnostic text explanations, pointing out the exact keywords swaying a classification toward "Fake" or "Real".
  * **SHAP Kernel Explainer:** Approximates feature contribution weights using Shapley values to provide statistical integrity behind the model's inner workings.

## 🛠️ How To Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/Dheeraj843/fake-news-detection-system.git](https://github.com/Dheeraj843/fake-news-detection-system.git)
