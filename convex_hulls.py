import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import math

def generate_random_vectors(n_vectors: int = 25, n_dimensions: int = 3, seed: int = 42) -> np.ndarray:
    """
    Generate random vectors with specified dimensions.
    
    Args:
        n_vectors (int): Number of vectors to generate
        n_dimensions (int): Number of dimensions for each vector
        seed (int): Random seed for reproducibility
        
    Returns:
        np.ndarray: Array of random vectors (n_vectors x n_dimensions)
    """
    np.random.seed(seed)
    
    # Generate all vectors from uniform distribution on [0,1]
    vectors = np.random.uniform(0, 1, (n_vectors, n_dimensions))
    
    print(f"Generated {vectors.shape[0]} vectors with {vectors.shape[1]} dimensions each")
    print(f"All vectors from uniform distribution U[0,1]")
    print(f"Vector statistics:")
    print(f"- Mean: {np.mean(vectors):.4f}")
    print(f"- Std: {np.std(vectors):.4f}")
    print(f"- Min: {np.min(vectors):.4f}")
    print(f"- Max: {np.max(vectors):.4f}")
    
    return vectors

def compute_convex_hull_statistics(vectors: np.ndarray) -> dict:
    """
    Compute various statistics about the convex hull of the vectors.
    
    Args:
        vectors (np.ndarray): Array of vectors
        
    Returns:
        dict: Dictionary containing hull statistics
    """
    try:
        hull = ConvexHull(vectors)
        
        # Calculate additional statistics
        hull_vertices = vectors[hull.vertices]
        hull_volume = hull.volume
        hull_area = hull.area if hasattr(hull, 'area') else None
        
        # Calculate distances from origin to hull vertices
        vertex_distances = np.linalg.norm(hull_vertices, axis=1)
        
        # Calculate distances from origin to all points
        all_distances = np.linalg.norm(vectors, axis=1)
        
        # Find points inside vs outside convex hull
        # For high dimensions, we'll use a simpler approach
        # Points that are hull vertices are definitely on the boundary
        is_vertex = np.zeros(len(vectors), dtype=bool)
        is_vertex[hull.vertices] = True
        
        num_boundary_points = np.sum(is_vertex)
        num_interior_points = len(vectors) - num_boundary_points
        fraction_boundary = num_boundary_points / len(vectors)
        
        stats = {
            'hull_volume': hull_volume,
            'hull_area': hull_area,
            'num_vertices': len(hull.vertices),
            'num_simplices': len(hull.simplices),
            'vertex_distances_mean': np.mean(vertex_distances),
            'vertex_distances_std': np.std(vertex_distances),
            'all_distances_mean': np.mean(all_distances),
            'all_distances_std': np.std(all_distances),
            'num_boundary_points': num_boundary_points,
            'num_interior_points': num_interior_points,
            'fraction_boundary': fraction_boundary,
            'hull_vertices_indices': hull.vertices.tolist()
        }
        
        return stats
        
    except Exception as e:
        print(f"Error computing convex hull: {e}")
        return None

def plot_hull_analysis(vectors: np.ndarray, hull_stats: dict, output_file: str = 'random_vectors_hull_analysis.png'):
    """
    Create a comprehensive analysis plot of the convex hull.
    
    Args:
        vectors (np.ndarray): Array of vectors
        hull_stats (dict): Statistics from convex hull computation
        output_file (str): Name of the output file
    """
    if hull_stats is None:
        print("No hull statistics available for plotting")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Distribution of distances from origin
    all_distances = np.linalg.norm(vectors, axis=1)
    vertex_distances = np.linalg.norm(vectors[hull_stats['hull_vertices_indices']], axis=1)
    
    ax1.hist(all_distances, bins=50, alpha=0.7, label='All points', color='skyblue')
    ax1.hist(vertex_distances, bins=30, alpha=0.8, label='Hull vertices', color='red')
    ax1.set_title('Distribution of Distances from Origin')
    ax1.set_xlabel('Distance from Origin')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot of first two dimensions with hull vertices highlighted
    ax2.scatter(vectors[:, 0], vectors[:, 1], alpha=0.6, s=20, color='skyblue', label='All points')
    ax2.scatter(vectors[hull_stats['hull_vertices_indices'], 0], 
                vectors[hull_stats['hull_vertices_indices'], 1], 
                c='red', s=100, zorder=5, label='Hull vertices')
    ax2.set_title('2D Projection with Hull Vertices Highlighted')
    ax2.set_xlabel('Dimension 0')
    ax2.set_ylabel('Dimension 1')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Component-wise distribution
    ax3.boxplot([vectors[:, i] for i in range(min(10, vectors.shape[1]))], tick_labels=[f'Dim {i}' for i in range(min(10, vectors.shape[1]))])
    ax3.set_title('Distribution of Vector Components')
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics summary
    ax4.axis('off')
    stats_text = f"""
    Convex Hull Analysis
    
    Hull Statistics:
    - Volume: {hull_stats['hull_volume']:.4f}
    - Number of vertices: {hull_stats['num_vertices']}
    - Number of simplices: {hull_stats['num_simplices']}
    - Boundary points: {hull_stats['num_boundary_points']}
    - Interior points: {hull_stats['num_interior_points']}
    - Fraction on boundary: {hull_stats['fraction_boundary']:.3f} ({hull_stats['fraction_boundary']*100:.1f}%)
    
    Distance Statistics:
    - Mean distance (all): {hull_stats['all_distances_mean']:.4f}
    - Std distance (all): {hull_stats['all_distances_std']:.4f}
    - Mean distance (vertices): {hull_stats['vertex_distances_mean']:.4f}
    - Std distance (vertices): {hull_stats['vertex_distances_std']:.4f}
    
    Vector Statistics:
    - Total vectors: {len(vectors)}
    - Dimensions: {vectors.shape[1]}
    - Mean component: {np.mean(vectors):.4f}
    - Std component: {np.std(vectors):.4f}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_multiple_dimensions(n_vectors: int = 25, dimensions: list = [3, 4, 5, 6, 7], seed: int = 42):
    """
    Analyze convex hull properties across multiple dimensions.
    
    Args:
        n_vectors (int): Number of vectors to generate
        dimensions (list): List of dimensions to analyze
        seed (int): Random seed for reproducibility
    """
    results = []
    
    for dim in dimensions:
        print(f"\n{'='*50}")
        print(f"Analyzing {n_vectors} vectors in {dim} dimensions")
        print(f"{'='*50}")
        
        # Generate vectors for this dimension
        vectors = generate_random_vectors(n_vectors=n_vectors, n_dimensions=dim, seed=seed)
        
        # Compute hull statistics
        hull_stats = compute_convex_hull_statistics(vectors)
        
        if hull_stats:
            print(f"Hull volume: {hull_stats['hull_volume']:.4f}")
            print(f"Number of hull vertices: {hull_stats['num_vertices']}")
            print(f"Number of hull simplices: {hull_stats['num_simplices']}")
            print(f"Boundary points: {hull_stats['num_boundary_points']}")
            print(f"Interior points: {hull_stats['num_interior_points']}")
            print(f"Fraction on boundary: {hull_stats['fraction_boundary']:.3f} ({hull_stats['fraction_boundary']*100:.1f}%)")
            
            # Store results
            results.append({
                'dimension': dim,
                'hull_volume': hull_stats['hull_volume'],
                'num_vertices': hull_stats['num_vertices'],
                'num_simplices': hull_stats['num_simplices'],
                'fraction_boundary': hull_stats['fraction_boundary'],
                'all_distances_mean': hull_stats['all_distances_mean'],
                'vertex_distances_mean': hull_stats['vertex_distances_mean'],
                'all_distances_std': hull_stats['all_distances_std'],
                'vertex_distances_std': hull_stats['vertex_distances_std']
            })
            
            # Save vectors
            np.save(f'random_vectors_{dim}d.npy', vectors)
            print(f"Vectors saved as 'random_vectors_{dim}d.npy'")
        else:
            print(f"Could not compute convex hull statistics for {dim} dimensions")
    
    return results

def plot_dimension_comparison(results: list, output_file: str = 'hull_properties_vs_dimension.png'):
    """
    Create charts comparing hull properties across different dimensions.
    
    Args:
        results (list): List of results from analyze_multiple_dimensions
        output_file (str): Name of the output file (not used, kept for compatibility)
    """
    if not results:
        print("No results to plot")
        return
    
    dimensions = [r['dimension'] for r in results]
    
    # Plot 1: Hull volume vs dimension
    plt.figure(figsize=(10, 6))
    volumes = [r['hull_volume'] for r in results]
    plt.semilogy(dimensions, volumes, 'bo-', linewidth=3, markersize=10)
    plt.title('Convex Hull Volume vs Dimension', fontsize=14, fontweight='bold')
    plt.xlabel('Dimension', fontsize=12)
    plt.ylabel('Hull Volume (log scale)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on the volume plot
    for i, (dim, vol) in enumerate(zip(dimensions, volumes)):
        plt.annotate(f'{vol:.3f}', (dim, vol), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('hull_volume_vs_dimension.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Number of simplices vs dimension
    plt.figure(figsize=(10, 6))
    simplices = [r['num_simplices'] for r in results]
    plt.semilogy(dimensions, simplices, 'ro-', linewidth=3, markersize=10)
    plt.title('Number of Hull Facets vs Dimension', fontsize=14, fontweight='bold')
    plt.xlabel('Dimension', fontsize=12)
    plt.ylabel('Number of Facets (log scale)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on the simplices plot
    for i, (dim, simp) in enumerate(zip(dimensions, simplices)):
        plt.annotate(f'{simp:.0e}', (dim, simp), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('hull_facets_vs_dimension.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Fraction of boundary points vs dimension
    plt.figure(figsize=(10, 6))
    fractions = [r['fraction_boundary'] for r in results]
    plt.plot(dimensions, fractions, 'go-', linewidth=3, markersize=10)
    plt.title('Fraction of Points on Boundary vs Dimension', fontsize=14, fontweight='bold')
    plt.xlabel('Dimension', fontsize=12)
    plt.ylabel('Fraction on Boundary', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (dim, frac) in enumerate(zip(dimensions, fractions)):
        plt.annotate(f'{frac*100:.1f}%', (dim, frac), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('boundary_fraction_vs_dimension.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE: Hull Properties vs Dimension")
    print(f"{'='*80}")
    print(f"{'Dim':<6} {'Volume':<15} {'Simplices':<15} {'Fraction':<12}")
    print(f"{'':<6} {'(log10)':<15} {'(log10)':<15} {'Boundary':<12}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['dimension']:<6} {np.log10(r['hull_volume']):<15.2f} {np.log10(r['num_simplices']):<15.2f} "
              f"{r['fraction_boundary']:<12.3f}")

def main():
    print("Analyzing convex hull properties across multiple dimensions...")
    
    # Analyze multiple dimensions
    dimensions = [3, 4, 5, 6, 7]
    results = analyze_multiple_dimensions(n_vectors=25, dimensions=dimensions, seed=42)
    
    # Create comparison charts
    print("\nCreating dimension comparison charts...")
    plot_dimension_comparison(results)
    print("Dimension comparison charts saved as:")
    print("- hull_volume_vs_dimension.png")
    print("- hull_facets_vs_dimension.png") 
    print("- boundary_fraction_vs_dimension.png")
    
    print(f"\nAnalysis complete! Generated files:")
    for dim in dimensions:
        print(f"- random_vectors_{dim}d.npy")
    print(f"- hull_volume_vs_dimension.png")
    print(f"- hull_facets_vs_dimension.png")
    print(f"- boundary_fraction_vs_dimension.png")

if __name__ == "__main__":
    main() 