import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
import zipfile
import matplotlib.pyplot as plt

# ======================== 1. DATA LOADING & AUGMENTATION ========================
def load_images_from_csv(csv_path, image_folder, target_size=(96, 96), augment=True):
    df = pd.read_csv(csv_path)
    class_names = sorted(df['label'].unique())
    label_dict = {name: idx for idx, name in enumerate(class_names)}
    
    images, labels = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(image_folder, row['filename'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size).astype(np.float32) / 255.0
        images.append(img)
        labels.append(label_dict[row['label']])
        
        # Augmentasi
        if augment:
            # Flip horizontal
            flipped = cv2.flip(img, 1)
            images.append(flipped)
            labels.append(label_dict[row['label']])
            
            # Rotasi 15 derajat
            M = cv2.getRotationMatrix2D((48,48), 15, 1.0)
            rotated = cv2.warpAffine(img, M, (96,96))
            images.append(rotated)
            labels.append(label_dict[row['label']])
            
            # Brightness adjustment
            bright = np.clip(img * 1.2, 0, 1)
            images.append(bright)
            labels.append(label_dict[row['label']])
    
    return np.array(images), np.array(labels), class_names

# ======================== 2. FEATURE EXTRACTION (HYBRID) ========================
def extract_hybrid_features(images):
    hog_features, color_features, glcm_features = [], [], []
    
    for img in images:
        # --- HOG Grayscale ---
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_uint8 = (gray * 255).astype(np.uint8)
        hog = cv2.HOGDescriptor((96,96), (16,16), (8,8), (8,8), 9)
        hog_feat = hog.compute(gray_uint8).flatten()
        
        # --- Color Histogram (HSV) ---
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hist = []
        for channel in range(3):
            hist_ch = np.histogram(hsv[:,:,channel], bins=16, range=(0,256))[0]
            hist.extend(hist_ch / (np.sum(hist_ch) + 1e-6))
        
        # --- GLCM ---
        glcm = calculate_glcm(gray)
        
        hog_features.append(hog_feat)
        color_features.append(hist)
        glcm_features.append(glcm)
    
    return np.hstack([np.array(hog_features), np.array(color_features), np.array(glcm_features)])

def calculate_glcm(image, levels=8):
    glcm = np.zeros((levels, levels), dtype=np.float32)
    # Implementasi GLCM sederhana
    for i in range(image.shape[0]-1):
        for j in range(image.shape[1]-1):
            val1 = int(image[i,j] * (levels-1))
            val2 = int(image[i+1,j] * (levels-1))
            glcm[val1, val2] += 1
    glcm /= np.sum(glcm)
    contrast = np.sum((np.arange(levels)[:,None] - np.arange(levels))**2 * glcm)
    energy = np.sum(glcm**2)
    homogeneity = np.sum(glcm / (1 + (np.arange(levels)[:,None] - np.arange(levels))**2))
    return np.array([contrast, energy, homogeneity])

# ======================== 3. NEURAL NETWORK WITH OPTIMIZATION ========================
class ButterflyNN:
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5, lambd=0.01):
        self.weights = [
            np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2./input_size),
            np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2./hidden_sizes[0]),
            np.random.randn(hidden_sizes[1], output_size) * np.sqrt(2./hidden_sizes[1])
        ]
        self.biases = [np.zeros((1, size)) for size in hidden_sizes + [output_size]]
        self.dropout_rate = dropout_rate
        self.lambd = lambd
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, X, training=True):
        self.z1 = X @ self.weights[0] + self.biases[0]
        self.a1 = self.relu(self.z1)
        if training:
            self.mask1 = (np.random.rand(*self.a1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.a1 *= self.mask1
        
        self.z2 = self.a1 @ self.weights[1] + self.biases[1]
        self.a2 = self.relu(self.z2)
        if training:
            self.mask2 = (np.random.rand(*self.a2.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.a2 *= self.mask2
        
        return self.softmax(self.a2 @ self.weights[2] + self.biases[2])
    
    def compute_loss(self, X, y_true):
        y_pred = self.forward(X, training=False)
        m = X.shape[0]
        log_probs = -np.log(np.maximum(y_pred[np.arange(m), y_true], 1e-10))
        data_loss = np.mean(log_probs)
        reg_loss = 0.5 * self.lambd * (np.sum(self.weights[0]**2) + np.sum(self.weights[1]**2) + np.sum(self.weights[2]**2)) / m
        return data_loss + reg_loss
    
    def backward(self, X, y_true, lr):
        m = X.shape[0]
        y_pred = self.forward(X, training=True)
        
        # Output layer gradient
        dz3 = y_pred
        dz3[np.arange(m), y_true] -= 1
        dz3 /= m
        dw3 = self.a2.T @ dz3 + self.lambd * self.weights[2] / m
        db3 = np.sum(dz3, axis=0, keepdims=True)
        
        # Hidden layer 2
        da2 = dz3 @ self.weights[2].T
        dz2 = da2 * (self.z2 > 0) * self.mask2
        dw2 = self.a1.T @ dz2 + self.lambd * self.weights[1] / m
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer 1
        da1 = dz2 @ self.weights[1].T
        dz1 = da1 * (self.z1 > 0) * self.mask1
        dw1 = X.T @ dz1 + self.lambd * self.weights[0] / m
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.weights[0] -= lr * dw1
        self.weights[1] -= lr * dw2
        self.weights[2] -= lr * dw3
        self.biases[0] -= lr * db1
        self.biases[1] -= lr * db2
        self.biases[2] -= lr * db3

# ======================== 4. TRAINING PIPELINE ========================
def main():
    # Load data
    csv_path = "./dataset/Training_set.csv"
    image_folder = "./dataset/train"
    images, labels, class_names = load_images_from_csv(csv_path, image_folder, augment=True)
    
    # Feature extraction
    print("Extracting hybrid features...")
    X = extract_hybrid_features(images)
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Normalize
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / (std + 1e-6)
    X_test = (X_test - mean) / (std + 1e-6)
    
    # Initialize model
    input_size = X_train.shape[1]
    output_size = len(class_names)
    model = ButterflyNN(
        input_size=input_size,
        hidden_sizes=[256, 128],
        output_size=output_size,
        dropout_rate=0.5,
        lambd=0.01
    )
    
    # Training with learning rate decay
    batch_size = 64
    epochs = 50
    initial_lr = 0.01
    
    print("Training model...")
    for epoch in range(epochs):
        lr = initial_lr * (0.9 ** (epoch // 10))  # LR decay every 10 epochs
        
        # Mini-batch training
        indices = np.random.permutation(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            model.backward(X_batch, y_batch, lr)
        
        # Evaluation
        train_loss = model.compute_loss(X_train, y_train)
        train_acc = np.mean(np.argmax(model.forward(X_train, training=False), axis=1) == y_train)
        test_acc = np.mean(np.argmax(model.forward(X_test, training=False), axis=1) == y_test)
        
        print(f"Epoch {epoch+1}/{epochs}, LR: {lr:.4f}, Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Final evaluation
    y_pred = np.argmax(model.forward(X_test, training=False), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.tight_layout()
    plt.show()

    return model, mean, std, class_names, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    model, mean, std, class_names, X_train, X_test, y_train, y_test = main()
    
    # Sekarang simpan model
    with open('butterfly_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'mean': mean,
            'std': std,
            'class_names': class_names
        }, f)
    print("Model saved!")
