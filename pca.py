'''
# Principal Component Analysis (PCA) for Image Detection

This project demonstrates how to use Principal Component Analysis (PCA) for image detection.
The implementation includes:
1. Basic PCA implementation from scratch
2. Face recognition using PCA (Eigenfaces)
3. Image compression using PCA
4. Anomaly detection in images using PCA
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import cv2
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

##############################################
# Part 1: PCA Implementation from Scratch
##############################################

class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        
    def fit(self, X):
        # Mean centering
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and corresponding eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store the first n_components eigenvectors
        self.components_ = eigenvectors[:, :self.n_components]
        
        # Calculate explained variance
        self.explained_variance_ = eigenvalues[:self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        return self
        
    def transform(self, X):
        # Project data onto principal components
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    
    def inverse_transform(self, X_transformed):
        # Reconstruct original data from transformed data
        return np.dot(X_transformed, self.components_.T) + self.mean_

##############################################
# Part 2: Face Recognition using PCA (Eigenfaces)
##############################################

def load_face_dataset():
    """Load the Olivetti faces dataset"""
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data
    y = faces.target
    return X, y, faces.images

def visualize_eigenfaces(pca, faces_images):
    """Visualize the top eigenfaces"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                            subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        if i < pca.components_.shape[0]:
            eigenface = pca.components_[i].reshape(faces_images[0].shape)
            # Normalize for visualization
            eigenface = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
            ax.imshow(eigenface, cmap='gray')
            ax.set_title(f'Eigenface {i+1}')
    plt.tight_layout()
    plt.show()

def face_recognition_demo():
    """Demonstrate face recognition using PCA (Eigenfaces)"""
    # Load dataset
    X, y, images = load_face_dataset()
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    
    # Apply PCA for dimensionality reduction
    n_components = 150  # Number of eigenfaces to keep
    pca = PCA(n_components=n_components, whiten=True)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Visualize eigenfaces
    visualize_eigenfaces(pca, images)
    
    # Train a classifier (KNN) on the reduced data
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train_pca, y_train)
    
    # Predict on test set
    y_pred = classifier.predict(X_test_pca)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Face recognition accuracy: {accuracy:.4f}")
    
    # Visualize some predictions
    fig, axes = plt.subplots(4, 5, figsize=(15, 12),
                            subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        if i < len(X_test):
            # Original image
            face_image = X_test[i].reshape(images[0].shape)
            ax.imshow(face_image, cmap='gray')
            
            # Get prediction
            pred_idx = y_pred[i]
            true_idx = y_test[i]
            
            # Add colored border based on correct/incorrect prediction
            if pred_idx == true_idx:
                ax.spines['bottom'].set_color('green')
                ax.spines['top'].set_color('green')
                ax.spines['right'].set_color('green')
                ax.spines['left'].set_color('green')
                ax.set_title(f"Correct: {pred_idx}", color='green')
            else:
                ax.spines['bottom'].set_color('red')
                ax.spines['top'].set_color('red')
                ax.spines['right'].set_color('red')
                ax.spines['left'].set_color('red')
                ax.set_title(f"Pred: {pred_idx}, True: {true_idx}", color='red')
    
    plt.tight_layout()
    plt.show()

##############################################
# Part 3: Image Compression using PCA
##############################################

def load_sample_image():
    """Load a sample image for compression demo"""
    # Try to load a sample image or generate one if that fails
    try:
        img = cv2.imread('sample_image.jpg')
        if img is None:
            # Create a sample image
            img = np.zeros((300, 400, 3), dtype=np.uint8)
            # Draw some shapes
            cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)
            cv2.circle(img, (300, 150), 100, (0, 255, 0), -1)
            cv2.line(img, (0, 0), (400, 300), (0, 0, 255), 5)
        return img
    except:
        # Create a sample image
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        # Draw some shapes
        cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)
        cv2.circle(img, (300, 150), 100, (0, 255, 0), -1)
        cv2.line(img, (0, 0), (400, 300), (0, 0, 255), 5)
        return img

def compress_image_pca(image, n_components_list):
    """Compress image using PCA with different numbers of components"""
    # Convert image to grayscale if it's color
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Reshape image to a 2D array
    h, w = gray_image.shape
    flattened_image = gray_image.reshape(-1, w)
    
    # Create figure for visualization
    n_plots = len(n_components_list) + 1  # +1 for original image
    fig, axes = plt.subplots(1, n_plots, figsize=(20, 4))
    
    # Display original image
    axes[0].imshow(gray_image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Compress and display images with different n_components
    for i, n_comp in enumerate(n_components_list):
        # Create and fit PCA model
        pca = PCAFromScratch(n_components=n_comp)
        pca.fit(flattened_image)
        
        # Transform and inverse transform for compression
        compressed = pca.transform(flattened_image)
        reconstructed = pca.inverse_transform(compressed)
        
        # Reshape back to image format
        reconstructed_image = reconstructed.reshape(h, w)
        
        # Calculate compression ratio and MSE
        original_size = flattened_image.shape[0] * flattened_image.shape[1]
        compressed_size = compressed.shape[0] * compressed.shape[1] + pca.components_.size
        compression_ratio = original_size / compressed_size
        mse = np.mean((gray_image - reconstructed_image) ** 2)
        
        # Plot reconstructed image
        axes[i+1].imshow(reconstructed_image, cmap='gray')
        axes[i+1].set_title(f'n_comp={n_comp}\nRatio: {compression_ratio:.2f}\nMSE: {mse:.2f}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

##############################################
# Part 4: Anomaly Detection in Images using PCA
##############################################

def generate_normal_and_anomalous_images(n_normal=100, n_anomalous=20):
    """Generate synthetic normal and anomalous images for demonstration"""
    image_size = 20
    normal_images = []
    
    # Generate normal images (squares)
    for _ in range(n_normal):
        img = np.zeros((image_size, image_size))
        # Create a square with random position and size
        size = np.random.randint(5, 10)
        pos_x = np.random.randint(0, image_size - size)
        pos_y = np.random.randint(0, image_size - size)
        img[pos_y:pos_y+size, pos_x:pos_x+size] = 1
        # Add some noise
        img += np.random.normal(0, 0.1, img.shape)
        img = np.clip(img, 0, 1)
        normal_images.append(img.flatten())
    
    # Generate anomalous images (circles, triangles, different patterns)
    anomalous_images = []
    for i in range(n_anomalous):
        img = np.zeros((image_size, image_size))
        
        if i % 3 == 0:  # Circle
            center = (image_size // 2, image_size // 2)
            radius = np.random.randint(5, 9)
            y, x = np.ogrid[:image_size, :image_size]
            dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            mask = dist_from_center <= radius
            img[mask] = 1
        elif i % 3 == 1:  # Triangle
            pts = np.array([[image_size//2, 5], 
                            [5, image_size-5], 
                            [image_size-5, image_size-5]])
            cv2.fillPoly(img, [pts], 1)
        else:  # Random noise
            img = np.random.rand(image_size, image_size)
        
        # Add some noise
        img += np.random.normal(0, 0.1, img.shape)
        img = np.clip(img, 0, 1)
        anomalous_images.append(img.flatten())
    
    # Convert to numpy arrays
    X_normal = np.array(normal_images)
    X_anomalous = np.array(anomalous_images)
    
    return X_normal, X_anomalous, image_size

def visualize_images(images, image_size, title):
    """Visualize a set of images"""
    n = min(10, len(images))
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    
    for i in range(n):
        axes[i].imshow(images[i].reshape(image_size, image_size), cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def anomaly_detection_demo():
    """Demonstrate anomaly detection in images using PCA reconstruction error"""
    # Generate dataset
    X_normal, X_anomalous, image_size = generate_normal_and_anomalous_images()
    
    # Visualize normal and anomalous images
    visualize_images(X_normal, image_size, "Normal Images (Squares)")
    visualize_images(X_anomalous, image_size, "Anomalous Images")
    
    # Split normal data into train and test
    X_train, X_test = train_test_split(X_normal, test_size=0.2, random_state=42)
    
    # Apply PCA (fit only on normal training data)
    n_components = 10
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    
    # Function to compute reconstruction error
    def reconstruction_error(X, pca):
        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        # Mean squared error per sample
        return np.mean((X - X_reconstructed) ** 2, axis=1)
    
    # Compute reconstruction errors
    errors_train = reconstruction_error(X_train, pca)
    errors_test = reconstruction_error(X_test, pca)
    errors_anomalous = reconstruction_error(X_anomalous, pca)
    
    # Find threshold for anomaly detection (e.g., 95th percentile of training errors)
    threshold = np.percentile(errors_train, 95)
    
    # Plot reconstruction errors
    plt.figure(figsize=(12, 6))
    plt.hist(errors_train, bins=20, alpha=0.5, label='Normal (Train)')
    plt.hist(errors_test, bins=20, alpha=0.5, label='Normal (Test)')
    plt.hist(errors_anomalous, bins=20, alpha=0.5, label='Anomalous')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.legend()
    plt.title('PCA Reconstruction Error for Normal vs Anomalous Images')
    plt.show()
    
    # Calculate detection metrics
    anomalies_test = errors_test > threshold
    anomalies_anomalous = errors_anomalous > threshold
    
    print(f"False positive rate (normal images classified as anomalies): {100 * np.mean(anomalies_test):.2f}%")
    print(f"True positive rate (anomalous images correctly identified): {100 * np.mean(anomalies_anomalous):.2f}%")
    
    # Visualize some results
    # 1. Normal images with highest reconstruction error
    high_error_normal_idx = np.argsort(errors_test)[-5:]
    high_error_normal = X_test[high_error_normal_idx]
    
    # 2. Anomalous images with highest reconstruction error
    high_error_anomalous_idx = np.argsort(errors_anomalous)[-5:]
    high_error_anomalous = X_anomalous[high_error_anomalous_idx]
    
    # 3. Anomalous images with lowest reconstruction error (hardest to detect)
    low_error_anomalous_idx = np.argsort(errors_anomalous)[:5]
    low_error_anomalous = X_anomalous[low_error_anomalous_idx]
    
    # Visualize
    visualize_images(high_error_normal, image_size, "Normal Images with Highest Reconstruction Error")
    visualize_images(high_error_anomalous, image_size, "Anomalous Images with Highest Reconstruction Error")
    visualize_images(low_error_anomalous, image_size, "Anomalous Images with Lowest Reconstruction Error (Hard to Detect)")

##############################################
# Main function to run all demos
##############################################

def main():
    print("Welcome to PCA for Image Detection Project!")
    print("\n1. Starting Face Recognition Demo...")
    face_recognition_demo()
    
    print("\n2. Starting Image Compression Demo...")
    sample_image = load_sample_image()
    compress_image_pca(sample_image, [5, 10, 20, 50])
    
    print("\n3. Starting Anomaly Detection Demo...")
    anomaly_detection_demo()

if __name__ == "__main__":
    main()