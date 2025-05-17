import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pickle
import os
from butterfly_nn_hyperparametertuning import ButterflyNN

class ButterflyClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Butterfly Species Classifier")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        # Load model
        self.model = None
        self.mean = None
        self.std = None
        self.class_names = None
        self.load_model()
        
        # GUI Elements
        self.create_widgets()
        
    def load_model(self):
        """Load model dari file pickle"""
        try:
            with open("butterfly_model_hyperparameter.pkl", 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.mean = data['mean']
                self.std = data['std']
                self.class_names = data['class_names']
                
                # Verifikasi dimensi
                expected_dim = self.model.weights[0].shape[0]
                print(f"Model loaded. Expected feature dimension: {expected_dim}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat model: {str(e)}")
            self.root.destroy()
    
    def create_widgets(self):
        """Membuat antarmuka pengguna"""
        # Header
        header_frame = tk.Frame(self.root, bg="#4a6baf", height=80)
        header_frame.pack(fill="x")
        
        tk.Label(
            header_frame,
            text="Butterfly Species Classifier",
            font=("Helvetica", 20, "bold"),
            bg="#4a6baf",
            fg="white"
        ).pack(pady=20)
        
        # Main Content
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Image Display
        img_frame = tk.Frame(main_frame, bg="white", bd=2, relief="groove")
        img_frame.pack(pady=10, fill="both", expand=True)
        
        self.image_label = tk.Label(img_frame, bg="white")
        self.image_label.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Button Frame
        btn_frame = tk.Frame(main_frame, bg="#f0f0f0")
        btn_frame.pack(pady=10)
        
        tk.Button(
            btn_frame,
            text="Pilih Gambar Kupu-Kupu",
            command=self.load_image,
            font=("Helvetica", 12),
            bg="#4a6baf",
            fg="white",
            padx=20,
            pady=10
        ).pack(side="left", padx=10)
        
        # Result Frame
        result_frame = tk.Frame(main_frame, bg="#f0f0f0")
        result_frame.pack(pady=20, fill="x")
        
        tk.Label(
            result_frame,
            text="Hasil Prediksi:",
            font=("Helvetica", 14, "bold"),
            bg="#f0f0f0"
        ).pack(anchor="w")
        
        self.prediction_label = tk.Label(
            result_frame,
            text="Spesies: -",
            font=("Helvetica", 12),
            bg="#f0f0f0"
        )
        self.prediction_label.pack(anchor="w")
        
        self.confidence_label = tk.Label(
            result_frame,
            text="Tingkat Kepercayaan: -",
            font=("Helvetica", 12),
            bg="#f0f0f0"
        )
        self.confidence_label.pack(anchor="w")
        
        # Footer
        footer_frame = tk.Frame(self.root, bg="#4a6baf", height=40)
        footer_frame.pack(fill="x", side="bottom")
        
        tk.Label(
            footer_frame,
            text="© 2025 Butterfly Classifier App by D3",
            font=("Helvetica", 10),
            bg="#4a6baf",
            fg="white"
        ).pack(pady=10)
    
    def load_image(self):
        """Memuat gambar dari dialog file"""
        file_types = [
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Kupu-Kupu",
            filetypes=file_types
        )
        
        if file_path:
            try:
                # Tampilkan gambar
                self.display_image(file_path)
                
                # Lakukan prediksi
                class_name, confidence = self.predict(file_path)
                
                # Tampilkan hasil
                self.prediction_label.config(text=f"Spesies: {class_name}")
                self.confidence_label.config(
                    text=f"Tingkat Kepercayaan: {confidence:.2%}",
                    fg="green" if confidence > 0.7 else "orange"
                )
                
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memproses gambar: {str(e)}")
    
    def display_image(self, file_path):
        """Menampilkan gambar di GUI"""
        MAX_SIZE = (400, 300)  # (width, height)
        
        img = Image.open(file_path)
        img.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
    
    def extract_hybrid_features(self, img_array):
        """Ekstraksi fitur dari gambar"""
        hog_features = []
        color_features = []
        glcm_features = []
        
        for img in img_array:
            # 1. HOG Features
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_uint8 = (gray * 255).astype(np.uint8)
            hog = cv2.HOGDescriptor((96,96), (16,16), (8,8), (8,8), 9)
            hog_feat = hog.compute(gray_uint8).flatten()
            
            # 2. Color Histogram (HSV)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hist = []
            for channel in range(3):
                hist_ch = np.histogram(hsv[:,:,channel], bins=16, range=(0,256))[0]
                hist.extend(hist_ch / (np.sum(hist_ch) + 1e-6))
            
            # 3. GLCM Features
            glcm = self.calculate_glcm(gray)
            
            hog_features.append(hog_feat)
            color_features.append(hist)
            glcm_features.append(glcm)
        
        return np.hstack([np.array(hog_features), 
                         np.array(color_features), 
                         np.array(glcm_features)])
    
    def calculate_glcm(self, image, levels=8):
        """Menghitung fitur GLCM"""
        glcm = np.zeros((levels, levels), dtype=np.float32)
        image = (image * (levels-1)).astype(np.uint8)
        
        for i in range(image.shape[0]-1):
            for j in range(image.shape[1]-1):
                val1 = image[i,j]
                val2 = image[i+1,j]
                glcm[val1, val2] += 1
                
        glcm /= np.sum(glcm)
        
        # Hitung fitur GLCM
        contrast = np.sum((np.arange(levels)[:,None] - np.arange(levels))**2 * glcm)
        energy = np.sum(glcm**2)
        homogeneity = np.sum(glcm / (1 + (np.arange(levels)[:,None] - np.arange(levels))**2))
        
        return np.array([contrast, energy, homogeneity])
    
    def preprocess(self, img_array):
        """Normalisasi fitur"""
        features = self.extract_hybrid_features(img_array)
        return (features - self.mean) / (self.std + 1e-6)
    
    def predict(self, img_path):
        """Prediksi spesies kupu-kupu"""
        # Baca dan preprocess gambar
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (96, 96)).astype(np.float32) / 255.0
        
        # Ekstrak fitur
        features = self.preprocess(np.array([img]))
        
        # Prediksi
        probs = self.model.forward(features, training=False)
        pred_class = np.argmax(probs)
        
        return self.class_names[pred_class], np.max(probs)

if __name__ == "__main__":
    root = tk.Tk()
    app = ButterflyClassifierApp(root)
    root.mainloop()
