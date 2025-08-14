import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


def generate_and_plot(n_vectors: int, dimension: int, rng: np.random.Generator) -> None:
    points = rng.standard_normal((n_vectors, dimension)).astype(np.float32)
    np.save(f'gaussian_points_{n_vectors}x{dimension}.npy', points)
    norms = np.linalg.norm(points, axis=1)

    sqrt_d = float(np.sqrt(dimension))

    # Print stats
    print(f"d={dimension}: norm mean={norms.mean():.4f}, norm std={norms.std():.4f}, sqrt(d)={sqrt_d:.4f}")

    # Plot histogram of norms (original units)
    plt.figure(figsize=(10, 6))
    plt.hist(norms, bins=40, color='steelblue', alpha=0.85, edgecolor='black')
    plt.axvline(sqrt_d, color='red', linestyle='--', linewidth=2, label=f'{sqrt_d:.2f}')
    plt.title(f'Vector Norms for {n_vectors} vectors in {dimension}D')
    plt.xlabel('norm')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = f'gaussian_vector_norms_d{dimension}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_relative_metrics_across_dimensions(dims: list[int], n_vectors: int, rng: np.random.Generator) -> None:
    # Ensure ascending order for readability
    dims_sorted = sorted(dims)
    rel_std = []
    rel_contrast = []
    skewnesses = []
    kurtoses = []  # excess kurtosis (Fisher)

    for d in dims_sorted:
        path = f'gaussian_points_{n_vectors}x{d}.npy'
        try:
            pts = np.load(path, mmap_mode=None)
        except FileNotFoundError:
            # Fallback: generate if missing
            pts = rng.standard_normal((n_vectors, d)).astype(np.float32)
            np.save(path, pts)
        norms = np.linalg.norm(pts, axis=1)
        m = float(np.mean(norms))
        s = float(np.std(norms))
        mn = float(np.min(norms))
        mx = float(np.max(norms))
        rel_std.append(s / m if m > 0 else 0.0)
        rel_contrast.append((mx - mn) / m if m > 0 else 0.0)
        skewnesses.append(skew(norms, bias=False))
        kurtoses.append(kurtosis(norms, fisher=True, bias=False))
        print(f"metrics d={d}: rel_std={rel_std[-1]:.6f}, rel_contrast={rel_contrast[-1]:.6f}, skew={skewnesses[-1]:.6f}, kurt(excess)={kurtoses[-1]:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(dims_sorted, rel_std, 'o-b', linewidth=2, label='Relative Std (std/mean)')
    plt.plot(dims_sorted, rel_contrast, 'o-r', linewidth=2, label='Relative Contrast ((max-min)/mean)')
    plt.plot(dims_sorted, skewnesses, 's-g', linewidth=2, label='Skewness')
    plt.plot(dims_sorted, kurtoses, 'd-m', linewidth=2, label='Kurtosis (excess)')
    plt.title(f'Relative Metrics vs Dimension (n={n_vectors} vectors per d)')
    plt.xlabel('Dimension d')
    plt.ylabel('Relative / Shape metric')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = 'gaussian_relative_metrics_across_dims.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    rng = np.random.default_rng(42)

    # Dimensions requested
    dims = [10000, 5000, 1000, 500, 100]
    n_vectors = 10000

    for d in dims:
        generate_and_plot(n_vectors=n_vectors, dimension=d, rng=rng)

    # Relative metrics chart across dimensions
    plot_relative_metrics_across_dimensions(dims=dims, n_vectors=n_vectors, rng=rng)


if __name__ == "__main__":
    main() 