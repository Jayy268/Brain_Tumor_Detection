import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

# --- CONFIGURATION ---
# Use relative paths so this runs on any computer
MODEL_PATH = 'saved_models/mobilenetv2_transfer.keras' # Ensure you save your model here
INTERNAL_DATA_PATH = 'data/brain_tumor_dataset'        # Your Kaggle dataset
EXTERNAL_DATA_PATH = 'data/brats_africa/glioma'        # Your BraTS subset
IMG_SIZE = (224, 224)

def plot_clinical_metrics(y_true, y_pred_probs, dataset_name="Dataset", threshold=0.5):
    """
    Generates and saves a clinically focused report.
    """
    y_pred = (y_pred_probs > threshold).astype(int)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # 2. Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n=== RESULTS: {dataset_name.upper()} ===")
    print(f"Sensitivity (Recall): {sensitivity:.2%}")
    print(f"Specificity:          {specificity:.2%}")
    print(f"F1-Score:             {f1:.4f}")
    
    # 3. Visualization
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred: No Tumor', 'Pred: Tumor'],
                yticklabels=['Actual: No Tumor', 'Actual: Tumor'])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plt.tight_layout()
    
    # Save plot instead of just showing it (better for headless servers/GitHub)
    save_path = f"results_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def evaluate_external(model):
    """Run inference on the external BraTS folder"""
    print(f"\nLoading external data from {EXTERNAL_DATA_PATH}...")
    
    if not os.path.exists(EXTERNAL_DATA_PATH):
        print("External dataset not found. Skipping.")
        return

    image_files = [f for f in os.listdir(EXTERNAL_DATA_PATH) if f.endswith('.png')]
    
    if not image_files:
        print("No images found in external folder.")
        return

    # Prepare data
    # BraTS contains only Tumors (Class 1)
    y_true = np.ones(len(image_files)) 
    y_probs = []

    print(f"Running inference on {len(image_files)} external images...")
    
    for img_file in image_files:
        img_path = os.path.join(EXTERNAL_DATA_PATH, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        
        prob = model.predict(img, verbose=0)[0][0]
        y_probs.append(prob)

    plot_clinical_metrics(y_true, np.array(y_probs), dataset_name="External_BraTS_Africa")

if __name__ == "__main__":
    # 1. Load Model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # 2. Run Evaluation
        evaluate_external(model)
        
    else:
        print("Model file not found. Please run training script first.")
