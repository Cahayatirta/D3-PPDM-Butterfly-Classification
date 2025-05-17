import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
import matplotlib.pyplot as plt
import pickle
import time

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
        
        # Augmentation
        if augment:
            # Horizontal flip
            flipped = cv2.flip(img, 1)
            images.append(flipped)
            labels.append(label_dict[row['label']])
            
            # Rotate 15 degrees
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
    # Simple GLCM implementation
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
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.lambd = lambd
        self.initialize_weights()
    
    def initialize_weights(self):
        self.weights = [
            np.random.randn(self.input_size, self.hidden_sizes[0]) * np.sqrt(2./self.input_size),
            np.random.randn(self.hidden_sizes[0], self.hidden_sizes[1]) * np.sqrt(2./self.hidden_sizes[0]),
            np.random.randn(self.hidden_sizes[1], self.output_size) * np.sqrt(2./self.hidden_sizes[1])
        ]
        self.biases = [np.zeros((1, size)) for size in self.hidden_sizes + [self.output_size]]
    
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
    
    def predict(self, X):
        return np.argmax(self.forward(X, training=False), axis=1)

# ======================== 4. HYPERPARAMETER TUNING ========================
def hyperparameter_tuning(X_train, y_train, input_size, output_size):
    print("Starting hyperparameter tuning...")
    
    # Define hyperparameter grid
    hidden_sizes_options = [
        [128, 64],
        [256, 128]
    ]
    dropout_rate_options = [0.3, 0.5]
    lambd_options = [0.01, 0.1]
    learning_rate_options = [0.01, 0.1]
    batch_size_options = [32, 64, 128]
    
    # Calculate total number of combinations
    total_configs = (len(hidden_sizes_options) * 
                     len(dropout_rate_options) * 
                     len(lambd_options) * 
                     len(learning_rate_options) * 
                     len(batch_size_options))
    print(f"Total configurations to test: {total_configs}")
    
    # Split data for validation
    train_indices = np.random.permutation(len(X_train))
    split_idx = int(len(X_train) * 0.8)
    
    X_train_subset = X_train[train_indices[:split_idx]]
    y_train_subset = y_train[train_indices[:split_idx]]
    X_val = X_train[train_indices[split_idx:]]
    y_val = y_train[train_indices[split_idx:]]
    
    # Prepare for grid search
    best_val_acc = 0
    best_params = None
    results = []
    
    # Start grid search with progress tracking
    start_time = time.time()
    config_count = 0
    
    for hidden_sizes in hidden_sizes_options:
        for dropout_rate in dropout_rate_options:
            for lambd in lambd_options:
                for lr in learning_rate_options:
                    for batch_size in batch_size_options:
                        config_count += 1
                        print(f"Testing configuration {config_count}/{total_configs}")
                        
                        # Store configuration
                        config = {
                            'hidden_sizes': hidden_sizes,
                            'dropout_rate': dropout_rate,
                            'lambd': lambd,
                            'learning_rate': lr,
                            'batch_size': batch_size
                        }
                        
                        # Create and train model
                        model = ButterflyNN(
                            input_size=input_size,
                            hidden_sizes=hidden_sizes,
                            output_size=output_size,
                            dropout_rate=dropout_rate,
                            lambd=lambd
                        )
                        
                        # Quick training (reduced epochs for tuning)
                        epochs = 10  # Reduced for faster tuning
                        for epoch in range(epochs):
                            indices = np.random.permutation(X_train_subset.shape[0])
                            for j in range(0, X_train_subset.shape[0], batch_size):
                                batch_idx = indices[j:j+batch_size]
                                X_batch = X_train_subset[batch_idx]
                                y_batch = y_train_subset[batch_idx]
                                model.backward(X_batch, y_batch, lr)
                        
                        # Evaluate on validation set
                        y_val_pred = model.predict(X_val)
                        val_acc = accuracy_score(y_val, y_val_pred)
                        
                        # Store result
                        result = {**config, 'val_accuracy': val_acc}
                        results.append(result)
                        
                        # Update best params if needed
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_params = config
                            print(f"New best: {best_params} with accuracy: {best_val_acc:.4f}")
    
    # Sort results by validation accuracy
    results.sort(key=lambda x: x['val_accuracy'], reverse=True)
    
    # Report best configuration and timing
    elapsed_time = time.time() - start_time
    print(f"\nHyperparameter tuning completed in {elapsed_time:.2f} seconds")
    print(f"Best configuration: {best_params}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Show top 5 configurations
    print("\nTop 5 configurations:")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. Accuracy: {result['val_accuracy']:.4f}, Params: {result}")
    
    return best_params, results

# ======================== 5. TRAINING PIPELINE ========================
def main():
    # Load data
    csv_path = "../dataset/Training_set.csv"
    image_folder = "../dataset/train"
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
    
    # Hyperparameter tuning
    input_size = X_train.shape[1]  
    output_size = len(class_names)
    best_params, tuning_results = hyperparameter_tuning(X_train, y_train, input_size, output_size)
    
    # Initialize model with best parameters
    model = ButterflyNN(
        input_size=input_size,
        hidden_sizes=best_params['hidden_sizes'],
        output_size=output_size,
        dropout_rate=best_params['dropout_rate'],
        lambd=best_params['lambd']
    )
    
    # Training with learning rate decay
    batch_size = best_params['batch_size']
    epochs = 50
    initial_lr = best_params['learning_rate']
    
    print("\nTraining final model with best parameters...")
    train_losses = []
    train_accs = []
    test_accs = []
    
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
        train_acc = np.mean(model.predict(X_train) == y_train)
        test_acc = np.mean(model.predict(X_test) == y_test)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs}, LR: {lr:.4f}, Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Final evaluation
    y_pred = model.predict(X_test)
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
    
    # Learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accs, label='Train')
    plt.plot(range(1, epochs+1), test_accs, label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Save tuning results
    with open('hyperparameter_tuning_results.pkl', 'wb') as f:
        pickle.dump(tuning_results, f)
    
    return model, mean, std, class_names, best_params

if __name__ == "__main__":
    model, mean, std, class_names, best_params = main()
    
    # Save model
    with open('butterfly_model_hyperparameter.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'mean': mean,
            'std': std,
            'class_names': class_names,
            'best_params': best_params
        }, f)
    print("Model saved!")
