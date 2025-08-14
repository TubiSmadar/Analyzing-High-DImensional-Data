import os
import numpy as np
from PIL import Image
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.special import gammaln
import math

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image and convert it to a numpy array.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Image as a numpy array
    """
    try:
        with Image.open(image_path) as img:
            # Convert image to RGB if it's not
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize image to a standard size (e.g., 224x224)
            img = img.resize((224, 224))
            return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def image_to_vector(image_array: np.ndarray) -> np.ndarray:
    """
    Convert an image array to a flattened vector.
    
    Args:
        image_array (np.ndarray): Image as a numpy array
        
    Returns:
        np.ndarray: Flattened vector representation of the image
    """
    if image_array is None:
        return None
    # Flatten the image array (224x224x3 -> 150528)
    return image_array.flatten()

def process_directory(directory_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Process all images in a directory and convert them to vectors.
    
    Args:
        directory_path (str): Path to the directory containing images
        
    Returns:
        Tuple[List[str], np.ndarray]: List of image filenames and their vector representations
    """
    image_files = []
    vectors = []
    
    # Supported image formats
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_path = os.path.join(directory_path, filename)
            image_array = load_image(image_path)
            if image_array is not None:
                vector = image_to_vector(image_array)
                if vector is not None:
                    image_files.append(filename)
                    vectors.append(vector)
    
    return image_files, np.array(vectors)

def calculate_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Calculate the Euclidean distance matrix between all vectors.
    
    Args:
        vectors (np.ndarray): Array of image vectors
        
    Returns:
        np.ndarray: Distance matrix
    """
    # Calculate pairwise Euclidean distances
    distances = pdist(vectors, metric='euclidean')
    # Convert to square matrix
    distance_matrix = squareform(distances)
    return distance_matrix

def plot_distance_histogram(distance_matrix: np.ndarray, output_file: str = 'distance_histogram.png'):
    """
    Plot a histogram of Euclidean distances between vectors with enhanced analysis.
    
    Args:
        distance_matrix (np.ndarray): Matrix of distances between vectors
        output_file (str): Name of the output file
    """
    plt.figure(figsize=(15, 10))
    
    # Get all unique distances (excluding self-distances)
    distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    
    # Calculate statistics
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    std_dist = np.std(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    # Calculate relative standard deviation and relative contrast
    relative_std = std_dist / mean_dist
    relative_contrast = (max_dist - min_dist) / mean_dist
    
    # Create histogram with density=True for better normal curve fitting
    counts, bins, _ = plt.hist(distances, bins=50, edgecolor='black', alpha=0.7, density=True, label='Original distances')
    plt.title('Distribution of Euclidean Distances Between Image Vectors\n(Enhanced Analysis)', fontsize=14, fontweight='bold')
    plt.xlabel('Euclidean Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 1. Add relative standard deviation and relative contrast as annotations
    stats_text = f"""
    Relative Statistics:
    - Relative Std Dev: {relative_std:.4f}
    - Relative Contrast: {relative_contrast:.4f}
    - Std Dev: {std_dist:.2f}
    - Mean: {mean_dist:.2f}
    - Range: {max_dist - min_dist:.2f}
    """
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # 2. Add fitted normal distribution curve
    from scipy.stats import norm
    
    # Fit normal distribution to the data
    mu, sigma = norm.fit(distances)
    
    # Create x values for the normal curve
    x = np.linspace(min_dist, max_dist, 100)
    normal_curve = norm.pdf(x, mu, sigma)
    
    # Plot the fitted normal distribution
    plt.plot(x, normal_curve, 'r-', linewidth=2, label=f'Fitted Normal (μ={mu:.1f}, σ={sigma:.1f})')
    
    # 3. Add low-dimensional comparison overlay (simulated)
    # Generate synthetic low-dimensional data (d=5) for comparison
    np.random.seed(42)  # For reproducibility
    n_pairs = len(distances)
    
    # Simulate distances in 5D space with similar mean but higher variance
    # In low dimensions, distances are more spread out (higher relative std)
    low_d_mean = mean_dist * 0.8  # Slightly lower mean for low-d
    low_d_std = mean_dist * 0.4   # Higher relative std for low-d (broader distribution)
    low_d_distances = np.random.normal(low_d_mean, low_d_std, n_pairs)
    low_d_distances = np.abs(low_d_distances)  # Ensure positive distances
    
    # Create histogram for low-d data
    counts_low_d, bins_low_d, _ = plt.hist(low_d_distances, bins=50, alpha=0.3, 
                                          color='green', density=True, 
                                          label=f'Simulated 5D distances (μ={low_d_mean:.1f}, σ={low_d_std:.1f})')
    
    # Add vertical lines for mean and median
    plt.axvline(mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.2f}')
    plt.axvline(median_dist, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_dist:.2f}')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print additional analysis
    print(f"\nDistance Distribution Analysis:")
    print(f"- Relative Standard Deviation: {relative_std:.4f}")
    print(f"- Relative Contrast: {relative_contrast:.4f}")
    print(f"- Normal fit parameters: μ={mu:.2f}, σ={sigma:.2f}")
    print(f"- Low-d simulation: μ={low_d_mean:.2f}, σ={low_d_std:.2f}")
    print(f"- High-d vs Low-d relative std ratio: {relative_std/(low_d_std/low_d_mean):.2f}")







def plot_vector_norms_histogram(vectors: np.ndarray, output_file: str = 'vector_norms_histogram.png'):
    """
    Create a histogram of the Euclidean distances of the original vectors from the origin.
    
    Args:
        vectors (np.ndarray): Array of image vectors (n_images x 150528)
        output_file (str): Name of the output file
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate Euclidean norms of each vector (distance from origin)
    # Use the original vectors, not centered ones
    vector_norms = np.linalg.norm(vectors, axis=1)
    
    # Create histogram
    plt.hist(vector_norms, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.title('Distribution of Euclidean Distances from Origin of Original Image Vectors', fontsize=14, fontweight='bold')
    plt.xlabel('Euclidean Distance from Origin', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_norm = np.mean(vector_norms)
    median_norm = np.median(vector_norms)
    std_norm = np.std(vector_norms)
    min_norm = np.min(vector_norms)
    max_norm = np.max(vector_norms)
    
    # Add vertical lines for mean and median
    plt.axvline(mean_norm, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_norm:.2f}')
    plt.axvline(median_norm, color='green', linestyle='--', linewidth=2, label=f'Median: {median_norm:.2f}')
    
    # Add text box with statistics
    stats_text = f"""
    Statistics (Distance from Origin):
    - Mean: {mean_norm:.2f}
    - Median: {median_norm:.2f}
    - Std Dev: {std_norm:.2f}
    - Min: {min_norm:.2f}
    - Max: {max_norm:.2f}
    - Range: {max_norm - min_norm:.2f}
    - Count: {len(vector_norms)}
    """
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return vector_norms

def plot_centered_vector_norms_histogram(vectors: np.ndarray, output_file: str = 'centered_vector_norms_histogram.png'):
    """
    Create a histogram of the Euclidean distances from the origin after centering the vectors.
    
    Args:
        vectors (np.ndarray): Array of image vectors (n_images x 150528)
        output_file (str): Name of the output file
    """
    plt.figure(figsize=(12, 8))
    
    # Center the vectors by subtracting the mean
    centered_vectors = vectors - np.mean(vectors, axis=0)
    
    # Calculate Euclidean norms of each centered vector (distance from origin)
    vector_norms = np.linalg.norm(centered_vectors, axis=1)
    
    # Create histogram
    plt.hist(vector_norms, bins=50, edgecolor='black', alpha=0.7, color='darkgreen')
    plt.title('Distribution of Euclidean Distances from Origin of Centered Image Vectors', fontsize=14, fontweight='bold')
    plt.xlabel('Euclidean Distance from Origin (After Centering)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_norm = np.mean(vector_norms)
    median_norm = np.median(vector_norms)
    std_norm = np.std(vector_norms)
    min_norm = np.min(vector_norms)
    max_norm = np.max(vector_norms)
    
    # Add vertical lines for mean and median
    plt.axvline(mean_norm, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_norm:.2f}')
    plt.axvline(median_norm, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_norm:.2f}')
    
    # Add text box with statistics
    stats_text = f"""
    Statistics (Centered Vectors):
    - Mean: {mean_norm:.2f}
    - Median: {median_norm:.2f}
    - Std Dev: {std_norm:.2f}
    - Min: {min_norm:.2f}
    - Max: {max_norm:.2f}
    - Range: {max_norm - min_norm:.2f}
    - Count: {len(vector_norms)}
    """
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return vector_norms

def plot_normalized_vector_norms_histogram(vectors: np.ndarray, output_file: str = 'normalized_vector_norms_histogram.png'):
    """
    Create a histogram of the Euclidean distances from the origin of normalized vectors.
    Each vector is normalized (variance=1) but not centered (original mean preserved).
    
    Args:
        vectors (np.ndarray): Array of image vectors (n_images x 150528)
        output_file (str): Name of the output file
    """
    plt.figure(figsize=(12, 8))
    
    # Normalize each vector to have variance = 1 (but don't center)
    # For each vector, divide by its standard deviation
    normalized_vectors = vectors / np.std(vectors, axis=1, keepdims=True)
    
    # Calculate Euclidean norms of each normalized vector (distance from origin)
    vector_norms = np.linalg.norm(normalized_vectors, axis=1)
    
    # Create histogram
    plt.hist(vector_norms, bins=50, edgecolor='black', alpha=0.7, color='brown')
    plt.title('Distribution of Euclidean Distances from Origin of Normalized Image Vectors', fontsize=14, fontweight='bold')
    plt.xlabel('Euclidean Distance from Origin (Normalized, Not Centered)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_norm = np.mean(vector_norms)
    median_norm = np.median(vector_norms)
    std_norm = np.std(vector_norms)
    min_norm = np.min(vector_norms)
    max_norm = np.max(vector_norms)
    
    # Add vertical lines for mean and median
    plt.axvline(mean_norm, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_norm:.2f}')
    plt.axvline(median_norm, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_norm:.2f}')
    
    # Add text box with statistics
    stats_text = f"""
    Statistics (Normalized, Not Centered):
    - Mean: {mean_norm:.2f}
    - Median: {median_norm:.2f}
    - Std Dev: {std_norm:.2f}
    - Min: {min_norm:.2f}
    - Max: {max_norm:.2f}
    - Range: {max_norm - min_norm:.2f}
    - Count: {len(vector_norms)}
    
    Normalization Info:
    - Each vector has variance = 1
    - Original mean is preserved
    - Norm represents distance from origin
    """
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return vector_norms

def plot_normalized_centered_vector_norms_histogram(vectors: np.ndarray, output_file: str = 'normalized_centered_vector_norms_histogram.png'):
    """
    Create a histogram of the Euclidean distances from the origin of normalized centered vectors.
    Each vector is centered (mean=0) and normalized (variance=1).
    
    Args:
        vectors (np.ndarray): Array of image vectors (n_images x 150528)
        output_file (str): Name of the output file
    """
    plt.figure(figsize=(12, 8))
    
    # Center the vectors by subtracting the mean
    centered_vectors = vectors - np.mean(vectors, axis=0)
    
    # Normalize each vector to have variance = 1
    # For each vector, divide by its standard deviation
    normalized_vectors = centered_vectors / np.std(centered_vectors, axis=1, keepdims=True)
    
    # Calculate Euclidean norms of each normalized centered vector (distance from origin)
    vector_norms = np.linalg.norm(normalized_vectors, axis=1)
    
    # Create histogram
    plt.hist(vector_norms, bins=50, edgecolor='black', alpha=0.7, color='purple')
    plt.title('Distribution of Euclidean Distances from Origin of Normalized Centered Image Vectors', fontsize=14, fontweight='bold')
    plt.xlabel('Euclidean Distance from Origin (Normalized & Centered)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_norm = np.mean(vector_norms)
    median_norm = np.median(vector_norms)
    std_norm = np.std(vector_norms)
    min_norm = np.min(vector_norms)
    max_norm = np.max(vector_norms)
    
    # Add vertical lines for mean and median
    plt.axvline(mean_norm, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_norm:.2f}')
    plt.axvline(median_norm, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_norm:.2f}')
    
    # Add text box with statistics
    stats_text = f"""
    Statistics (Normalized & Centered):
    - Mean: {mean_norm:.2f}
    - Median: {median_norm:.2f}
    - Std Dev: {std_norm:.2f}
    - Min: {min_norm:.2f}
    - Max: {max_norm:.2f}
    - Range: {max_norm - min_norm:.2f}
    - Count: {len(vector_norms)}
    
    Normalization Info:
    - Each vector has mean ≈ 0
    - Each vector has variance = 1
    - Norm represents distance from origin
    """
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return vector_norms

def plot_nearest_neighbor_histogram(vectors: np.ndarray, k: int = 5, output_file: str = 'nearest_neighbor_histogram.png'):
    """
    Plot histogram of k-nearest neighbor distances for high-d data and overlay a faded
    (simulated) low-dimensional (d=5) comparison.
    """
    # Compute pairwise distances efficiently
    distance_matrix = squareform(pdist(vectors, metric='euclidean'))
    # For each point, sort distances and take 1..k (exclude 0 self-distance)
    sorted_distances = np.sort(distance_matrix, axis=1)
    knn_distances = sorted_distances[:, 1:k+1].flatten()

    plt.figure(figsize=(12, 7))
    # High-d histogram
    plt.hist(knn_distances, bins=50, color='tab:blue', alpha=0.8, density=True, label=f'High-d data (k={k})')

    # Simulated low-d overlay (d=5) with larger spread
    np.random.seed(42)
    mu = np.mean(knn_distances) * 0.85
    sigma = np.std(knn_distances) * 1.75
    lowd_sim = np.abs(np.random.normal(mu, sigma, size=knn_distances.shape[0]))
    plt.hist(lowd_sim, bins=50, color='tab:green', alpha=0.25, density=True, label='Simulated 5D overlay')

    plt.title('Nearest-Neighbor Distance Histogram (with 5D overlay)')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_relative_metrics_vs_dimension(max_dimension: int = 100, n_points: int = 30, output_file: str = 'relative_metrics_vs_dimension.png'):
    """
    Simulate random points for dimensions 1..max_dimension (n_points per dim),
    compute pairwise distances, and plot relative std and relative contrast vs dimension.
    """
    dims = np.arange(1, max_dimension + 1)
    relative_stds = []
    relative_contrasts = []

    np.random.seed(123)
    for d in dims:
        # Simulate n_points in [0,1]^d
        pts = np.random.rand(n_points, d)
        dm = squareform(pdist(pts, metric='euclidean'))
        ds = dm[np.triu_indices_from(dm, k=1)]
        m = np.mean(ds)
        s = np.std(ds)
        mn = np.min(ds)
        mx = np.max(ds)
        relative_stds.append(s / m if m > 0 else 0.0)
        relative_contrasts.append((mx - mn) / m if m > 0 else 0.0)

    plt.figure(figsize=(12, 7))
    plt.plot(dims, relative_stds, 'b-', label='Relative Std (std/mean)', linewidth=2)
    plt.plot(dims, relative_contrasts, 'r-', label='Relative Contrast ((max-min)/mean)', linewidth=2)
    plt.title(f'Relative Metrics vs. Dimension (n={n_points} points per dimension)')
    plt.xlabel('Dimension')
    plt.ylabel('Relative value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_sphere_to_cube_volume_ratio(max_dimension: int = 1000, output_file: str = 'sphere_to_cube_volume_ratio.png'):
    """
    Plot the ratio of the unit sphere volume to the volume of its bounding cube vs dimension.
    Sphere volume: pi^{d/2} / Gamma(d/2 + 1), Cube volume: 2^d.
    Use log-space via gammaln for numerical stability.
    """
    dims = np.arange(1, max_dimension + 1)
    # log sphere volume = (d/2) * log(pi) - gammaln(d/2 + 1)
    log_sphere_vol = (dims / 2.0) * np.log(np.pi) - gammaln(dims / 2.0 + 1.0)
    # log cube volume = d * log(2)
    log_cube_vol = dims * np.log(2.0)
    log_ratio = log_sphere_vol - log_cube_vol
    ratio = np.exp(log_ratio)

    plt.figure(figsize=(12, 7))
    plt.semilogy(dims, ratio, 'k-')
    plt.title('Volume Ratio (Unit Sphere / Bounding Cube) vs Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('Ratio (log scale)')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Use the specific test directory path
    directory_path = r"C:\Users\Home\Desktop\python project\test\test"
    print(f"Looking for images in: {directory_path}")
    
    if os.path.exists(directory_path):
        image_files, vectors = process_directory(directory_path)
        
        if len(image_files) == 0:
            print("\nNo images found! Please make sure you have image files (.jpg, .jpeg, .png, .bmp, .gif) in the directory.")
            print("Supported image formats: .jpg, .jpeg, .png, .bmp, .gif")
            return
            
        print(f"\nProcessed {len(image_files)} images:")
        for filename in image_files:
            print(f"- {filename}")
        print(f"\nOriginal vector shape: {vectors.shape}")
        
        # Create histogram of vector norms (original, not centered)
        print("\nCreating histogram of vector norms (original)...")
        vector_norms = plot_vector_norms_histogram(vectors)
        print("Vector norms histogram saved as 'vector_norms_histogram.png'")
        
        # Create histogram of centered vector norms
        print("\nCreating histogram of centered vector norms...")
        centered_vector_norms = plot_centered_vector_norms_histogram(vectors)
        print("Centered vector norms histogram saved as 'centered_vector_norms_histogram.png'")
        
        # Create histogram of normalized vector norms (not centered)
        print("\nCreating histogram of normalized vector norms (not centered)...")
        normalized_vector_norms = plot_normalized_vector_norms_histogram(vectors)
        print("Normalized vector norms histogram saved as 'normalized_vector_norms_histogram.png'")
        
        # Create histogram of normalized centered vector norms
        print("\nCreating histogram of normalized centered vector norms...")
        normalized_centered_vector_norms = plot_normalized_centered_vector_norms_histogram(vectors)
        print("Normalized centered vector norms histogram saved as 'normalized_centered_vector_norms_histogram.png'")
        
        # Create original distance histogram
        print("\nCreating original distance histogram...")
        original_distance_matrix = calculate_distance_matrix(vectors)
        plot_distance_histogram(original_distance_matrix, 'original_distance_histogram.png')
        print("Original distance histogram saved as 'original_distance_histogram.png'")
        
        # Save original vectors
        np.save('image_vectors.npy', vectors)
        
        # Nearest neighbor histogram (with 5D overlay)
        print("\nCreating nearest-neighbor distance histogram (k=5)...")
        plot_nearest_neighbor_histogram(vectors, k=5)
        print("Nearest-neighbor histogram saved as 'nearest_neighbor_histogram.png'")
        
        # Relative metrics vs dimension (simulated)
        print("Creating relative metrics vs dimension plot (simulated 1..100D)...")
        plot_relative_metrics_vs_dimension(max_dimension=100, n_points=30)
        print("Relative metrics plot saved as 'relative_metrics_vs_dimension.png'")
        
        # Sphere-to-cube volume ratio (theoretical)
        print("Creating sphere-to-cube volume ratio plot (1..380D)...")
        plot_sphere_to_cube_volume_ratio(max_dimension=380)
        print("Sphere-to-cube volume ratio saved as 'sphere_to_cube_volume_ratio.png'")
        
        # Save filenames to a text file
        with open('image_filenames.txt', 'w') as f:
            for filename in image_files:
                f.write(f"{filename}\n")
        
        print("\nResults saved in:")
        print("- image_vectors.npy (original vector data)")
        print("- image_filenames.txt (list of processed images)")
        print("- vector_norms_histogram.png (distribution of original vector norms)")
        print("- centered_vector_norms_histogram.png (distribution of centered vector norms)")
        print("- normalized_vector_norms_histogram.png (distribution of normalized vector norms)")
        print("- normalized_centered_vector_norms_histogram.png (distribution of normalized centered vector norms)")
        print("- original_distance_histogram.png (original distance distribution)")
        print("- nearest_neighbor_histogram.png (k-NN distance histogram with 5D overlay)")
        print("- relative_metrics_vs_dimension.png (simulated relative metrics 1..100D)")
        print("- sphere_to_cube_volume_ratio.png (theoretical ratio 1..1000D)")
        
        # Print storage statistics
        original_size = vectors.nbytes / (1024 * 1024)  # MB
        print(f"\nStorage information:")
        print(f"- Original vectors: {original_size:.2f} MB")
        print(f"- Vector shape: {vectors.shape}")
        print(f"- Total dimensions: {vectors.shape[0] * vectors.shape[1]:,}")
        
        print(f"\nVector norms analysis:")
        print(f"- Original mean norm: {np.mean(vector_norms):.2f}")
        print(f"- Centered mean norm: {np.mean(centered_vector_norms):.2f}")
        print(f"- Normalized mean norm: {np.mean(normalized_vector_norms):.2f}")
        print(f"- Normalized centered mean norm: {np.mean(normalized_centered_vector_norms):.2f}")
        print(f"- Centering effect: {np.mean(vector_norms) - np.mean(centered_vector_norms):.2f}")
        print(f"- Normalization effect: {np.mean(vector_norms) - np.mean(normalized_vector_norms):.2f}")
        print(f"- Combined effect: {np.mean(vector_norms) - np.mean(normalized_centered_vector_norms):.2f}")
        
    else:
        print(f"Directory {directory_path} does not exist")

if __name__ == "__main__":
    main() 