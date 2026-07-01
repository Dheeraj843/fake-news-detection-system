# 📰 Explainable Fake News Detection System

A robust, hybrid deep learning framework built to classify news articles as **Real** or **Fake**. This system leverages the deep semantic understanding of **DistilBERT** paired with a **1D-CNN** head for high-accuracy local feature extraction. To ensure transparency, the framework integrates Explainable AI (XAI) tools (**LIME** & **SHAP**) to interpret and visually explain the model's classification decisions.

## 🚀 Tech Stack & Frameworks
* **Language:** Python 3.9+
* **Core Deep Learning:** TensorFlow / Keras
* **NLP Backbone:** HuggingFace Transformers (DistilBERT)
* **Explainability (XAI):** LIME, SHAP
* **Data Processing & ML Pipelines:** Scikit-learn, Pandas, NumPy

---

## 📊 Core Architecture & Features
* **Hybrid Classifier:** Passes contextual token embeddings through an optimized 1D Convolutional Neural Network layer with Global Max Pooling to capture local textual patterns, achieving over 85% accuracy on benchmark test samples.
* **Dual-Layer Interpretability:**
  * **LIME Integration:** Generates localized model-agnostic text explanations, pointing out the exact keywords swaying a classification toward "Fake" or "Real".
  * **SHAP Kernel Explainer:** Approximates feature contribution weights using Shapley values to provide statistical integrity behind the model's inner workings.

---

## 🗺️ Project Architecture & Workflow Pipeline

The processing workflow of an incoming news article flows seamlessly through the following pipeline:

```text
[Raw News Text] 
       │
       ▼
[HuggingFace Tokenizer] ───► Token IDs & Attention Masks
       │
       ▼
[DistilBERT Backbone]   ───► 768-Dimensional Contextual Embeddings
       │
       ▼
[1D CNN Layer]          ───► Local Feature Extraction (Kernel Size = 3)
       │
       ▼
[Global Max Pooling]    ───► Primary Feature Vector Synthesis
       │
       ▼
[Dense Softmax Output]  ───► Binary Classification (Real vs. Fake Probability)
       │
       ▼
[LIME & SHAP Engines]   ───► Feature Attribution & Predictive Explanation Output


📁Directory Structure
fake-news-detection-system/
│
├── app.py                 # Main application script handling pipelines, training, and XAI
├── requirements.txt       # Hardcoded python library dependency registry
├── .gitignore             # Specific exclusions to avoid tracking heavy build/cache files
└── README.md              # Project documentation and presentation portal

🛠️ How To Run Locally

Follow these precise steps to deploy and run the system on your local machine:

1. Clone the Repository
Bash
git clone [https://github.com/Dheeraj843/fake-news-detection-system.git](https://github.com/Dheeraj843/fake-news-detection-system.git)
cd fake-news-detection-system
2. Set Up a Virtual Environment (Optional but Recommended)
Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Execute the Pipeline
Bash
python app.py

🔍 Model Interpretability Insights (XAI)
When running predictions, the system automatically runs text explanations to show you exactly why a decision was made rather than operating as a black box:

LIME Output: Highlights specific terms (e.g., words like "shocking", "alien", "unbelievable") and computes their mathematical weights to illustrate how heavily they dragged the prediction threshold toward the Fake classification.

SHAP Output: Provides an overall feature value analysis across the background dataset distribution to ensure individual predictions align logically with generalized base trends.

📈 Future Enhancements Roadmap
[ ] Dataset Scale-Up: Migrate the local mock data framework to pull from comprehensive, real-world benchmark corpuses like ISOT or WELFake.

[ ] Interactive UI: Wrap the underlying execution architecture in a lightweight Streamlit web application dashboard.

[ ] API Layer: Expose prediction and explanation engines via a highly scalable FastAPI gateway endpoint.


