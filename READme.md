# Deep Learning for Brain Tumor Detection: Addressing Domain Shift in Low-Resource Settings

## 🏥 Project Overview
This project investigates the critical challenge of **domain generalization** in medical AI. It compares two deep learning approaches—a custom Convolutional Neural Network (CNN) and a Transfer Learning strategy (MobileNetV2)—to detect brain tumors in MRI scans.

Beyond standard validation, this study rigorously tests both models on an **external, real-world clinical dataset (BraTS-Africa)** to evaluate performance under domain shift. The goal was to determine which architecture offers the safety and robustness required for clinical deployment in diverse environments.

## ⚡ Key Highlights
* **Dual-Path Experimentation:** Benchmarked a Custom CNN (trained from scratch) against a frozen MobileNetV2 (Transfer Learning).
* **Real-World Validation:** Utilized the **BraTS-Africa** dataset (30 patients, ~150 slices) as a completely unseen external test set to measure robustness.
* **Critical Discovery:** Uncovered a severe case of **"Shortcut Learning"** in the custom model (0% external sensitivity) vs. robust generalization in the transfer learning model (79.3% external sensitivity).
* **Clinical Metrics:** Prioritized **Sensitivity (Recall)** and **Specificity** over raw accuracy to reflect clinical safety requirements.

## 📊 Methodology

### The Two Approaches
1.  **Path 1: Custom CNN**
    * A lightweight, 4-layer Convolutional Neural Network built from scratch.
    * Designed to test the efficacy of learning features purely from a small medical dataset (~200 images).

2.  **Path 2: Transfer Learning (MobileNetV2)**
    * A state-of-the-art architecture pre-trained on ImageNet.
    * **Strategy:** "Frozen" feature extraction. We froze the base layers to leverage robust, generalized visual filters and only trained the classification head.

### The Datasets
* **Internal Training/Validation:** Public Kaggle Brain Tumor MRI Dataset (Standardized, clean data).
* **External Stress Test:** **BraTS-Africa Dataset**. A challenging, multi-center dataset containing pre-operative MRIs from Nigerian patients. This served as a proxy for "real-world deployment."

## 📉 Results & Analysis

The models were evaluated on two fronts: **Internal Validation** (same domain) and **External Clinical Test** (new domain).

| Model Architecture | Internal Accuracy (Kaggle) | External Sensitivity (BraTS-Africa) | Status |
| :--- | :--- | :--- | :--- |
| **Custom CNN** | 78.0% | **0.0%** ⚠️ | **Model Collapse** |
| **MobileNetV2 (Frozen)** | **95.0%** | **79.3%** ✅ | **Robust** |

### 🧠 Critical Finding: The Danger of Shortcut Learning
The most significant outcome of this study was the failure of the Custom CNN on external data.

* **Observation:** While the Custom CNN achieved 78% accuracy on the training distribution, it failed to identify a *single* tumor in the BraTS-Africa dataset (0% Sensitivity).
* **Diagnosis:** Analysis of confidence scores (clustering ~0.2) suggests the model engaged in **Shortcut Learning**. Instead of learning tumor features, it overfitted to specific noise patterns and artifacts present only in the Kaggle dataset.
* **Solution:** The Transfer Learning model, leveraging robust features learned from ImageNet, successfully bridged the domain gap, maintaining high sensitivity (79.3%) even on MRI scans from different machines and populations.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/brain-tumor-generalization.git](https://github.com/your-username/brain-tumor-generalization.git)
    cd brain-tumor-generalization
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Setup:**
    * Download the [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/arwabasal/brain-tumor-mri-detection) and place it in `data/internal`.
    * (Optional for validation) Download the BraTS-Africa subset and place it in `data/external`.

4.  **Execute Training & Evaluation:**
    ```bash
    # Train and evaluate the Custom CNN
    python src/train_custom_cnn.py

    # Train and evaluate the Transfer Learning Model
    python src/train_transfer_learning.py
    ```

## 🛠️ Tech Stack
* **Frameworks:** TensorFlow/Keras, OpenCV, NumPy, Scikit-Learn.
* **Medical Imaging:** Nibabel (for NIfTI processing).
* **Visualization:** Matplotlib, Seaborn (Confusion Matrices, ROC Curves).

---

### 👨‍💻 Author's Note
This project was designed to simulate a real-world R&D pipeline—moving from initial prototyping to rigorous stress-testing against clinical realities. It highlights my focus on building AI systems that are not just accurate on paper, but safe and effective in the real world.
