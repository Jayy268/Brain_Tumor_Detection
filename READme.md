# Brain Tumor MRI Classification

## 📖 Project Overview

This project builds and compares two deep learning models for classifying brain MRI scans as either "tumour" or "non-tumour."

The goal is to analyze the performance difference between:

1.  **Path 1 (Custom CNN):** A Convolutional Neural Network built and trained from scratch.
2.  **Path 2 (Transfer Learning):** A pre-trained `MobileNetV2` model adapted for this classification task.

The repository includes the training scripts for both paths and a final analysis of their results.

## 🚀 How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/brain-tumor-classification.git](https://github.com/your-username/brain-tumor-classification.git)
    cd brain-tumor-classification
    ```

2.  **Get the Data:**
    This project uses the Brain Tumor MRI Dataset. You can download it from Kaggle:
    [Link to Kaggle Dataset](https://www.kaggle.com/datasets/arwabasal/brain-tumor-mri-detection)
    
    You will need to place the dataset folder in the main project directory.

3.  **Install Dependencies:**
    This project's requirements are listed in `requirements.txt`. You can install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Training:**
    You can run either training script:
    ```bash
    # To run the custom CNN (Path 1)
    python train_path1_custom.py
    # To run the transfer learning model (Path 2)
    python train_path2_transfer.py
    ```

## 📊 Findings & Conclusion

The performance of the two paths was compared based on their highest achieved validation accuracy.

| Model Path | Architecture | Peak Validation Accuracy |
| :--- | :--- | :--- |
| **Path 1** | Custom CNN (from scratch) | **78%** |
| **Path 2** | Transfer Learning (`MobileNetV2`) | **92%** |

### Key Takeaways:

* **Performance:** The Transfer Learning model (Path 2) significantly outperformed the custom-built CNN (Path 1) by **14 percentage points**.
* **Relevance:** This project clearly demonstrates the power of transfer learning for medical image analysis. By leveraging the pre-trained features of `MobileNetV2`, the model could focus on learning the specific features of brain tumors, leading to much higher accuracy than a model trying to learn *all* features from scratch.
* **Fine-Tuning:** Further experiments (as documented in the notebooks) showed that fine-tuning the `MobileNetV2` model did *not* improve performance, indicating the "frozen" base model provided the optimal feature set for this task.

