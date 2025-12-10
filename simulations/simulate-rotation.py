import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import vonmises_fisher
from tqdm import tqdm
import ot
from scipy.spatial.transform import Rotation


def generate_vonmises_fisher_samples(n_samples, kappa, mu=None):
    """Generate samples from von Mises-Fisher distribution on unit sphere."""
    if mu is None:
        mu = np.array([1.0, 0.0, 0.0])  # Default mean direction

    mu = mu / np.linalg.norm(mu)  # Normalize mu
    vmf = vonmises_fisher(mu, kappa)

    return vmf.rvs(n_samples)


def create_elongated_distribution(n_samples, kappa, scale_factor=3.0, axis=0):
    """Create an elongated von Mises-Fisher distribution on the sphere."""
    # Generate base samples with lower concentration to make elongation visible
    mu = np.array([1.0, 0.0, 0.0])
    vmf = vonmises_fisher(mu, kappa)  # Lower concentration for more spread
    vectors = vmf.rvs(n_samples)

    # Apply scaling to create elongation
    scaled_vectors = vectors.copy()
    scaled_vectors[:, axis] *= scale_factor

    # Renormalize to unit sphere
    norms = np.linalg.norm(scaled_vectors, axis=1, keepdims=True)
    scaled_vectors = scaled_vectors / norms

    return scaled_vectors


def rotate_vectors(vectors, angle_degrees, axis="z"):
    """Rotate vectors by specified angle around given axis."""
    rotation = Rotation.from_euler(axis, angle_degrees, degrees=True)
    return rotation.apply(vectors)


def rotate_vectors_around_mean(vectors, angle_degrees):
    """Rotate vectors by specified angle around their mean direction."""
    # Calculate mean direction
    mean_dir = np.array([1.0, 0.0, 0.0])

    # Create rotation matrix around the mean direction
    # Use Rodrigues' rotation formula
    angle_rad = np.radians(angle_degrees)

    rotation_axis = mean_dir

    # Rodrigues' rotation formula
    K = np.array(
        [
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0],
        ]
    )

    R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * np.dot(K, K)

    # Apply rotation to vectors (centered around mean)
    centered_vectors = vectors - mean_dir
    rotated_centered = np.dot(centered_vectors, R.T)
    rotated_vectors = rotated_centered + mean_dir

    # Renormalize to unit sphere
    norms = np.linalg.norm(rotated_vectors, axis=1, keepdims=True)
    rotated_vectors = rotated_vectors / norms

    return rotated_vectors


def apd_distance(vectors1, vectors2):
    """Average pairwise distance using mean vectors."""
    v_t1 = np.mean(vectors1, axis=0)
    v_t2 = np.mean(vectors2, axis=0)
    return 1 - np.dot(v_t1, v_t2)


def ot_distance(vectors1, vectors2):
    """Optimal transport distance using cosine distance."""
    cost_matrix = 1 - np.dot(vectors1, vectors2.T)
    ot_dist = ot.emd2([], [], cost_matrix)
    return ot_dist


def simulate_rotation_effects(kappa=10.0, n_samples=100, n_simulations=100):
    """Simulate how APD and OT distances depend on rotation."""
    rotation_angles = list(range(0, 181, 15))  # Every 15 degrees from 0 to 180

    apd_results = []
    ot_results = []

    for _ in tqdm(range(n_simulations), desc="Running simulations"):
        # Create elongated distribution T1
        t1_vectors = create_elongated_distribution(
            n_samples, kappa, scale_factor=4.0, axis=1
        )

        # Create rotated versions T2
        apd_vals = []
        ot_vals = []

        for angle in rotation_angles:
            # Rotate T1 to create T2 around its mean
            t2_vectors = create_elongated_distribution(
                n_samples, kappa, scale_factor=4.0, axis=1
            )
            t2_vectors = rotate_vectors_around_mean(t2_vectors, angle)

            # Calculate distances
            apd_vals.append(apd_distance(t1_vectors, t2_vectors))
            ot_vals.append(ot_distance(t1_vectors, t2_vectors))

        apd_results.append(apd_vals)
        ot_results.append(ot_vals)

    return (
        np.array(apd_results),
        np.array(ot_results),
        rotation_angles,
    )


def plot_rotation_effects(apd_results, ot_results, rotation_angles):
    """Plot how distances change with rotation angle."""
    # Calculate statistics
    apd_mean = np.mean(apd_results, axis=0)
    apd_std = np.std(apd_results, axis=0)
    ot_mean = np.mean(ot_results, axis=0)
    ot_std = np.std(ot_results, axis=0)

    # Create plot
    plt.figure(figsize=(12, 8))

    plt.errorbar(
        rotation_angles,
        apd_mean,
        yerr=apd_std,
        label="APD",
        marker="o",
        capsize=5,
        alpha=0.7,
        linestyle="-",
        color="tab:blue",
    )
    plt.errorbar(
        rotation_angles,
        ot_mean,
        yerr=ot_std,
        label="OT",
        marker="s",
        capsize=5,
        alpha=0.7,
        linestyle=":",
        color="tab:orange",
    )

    plt.xlabel("Rotation Angle (degrees)", fontsize=14)
    plt.ylabel("Distribution Distance", fontsize=14)
    plt.xticks(np.arange(0, 181, 30), fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig("rotation_vs_distance.png", dpi=300, bbox_inches="tight")
    plt.savefig("rotation_vs_distance.eps", dpi=300, bbox_inches="tight")
    plt.show()

    return apd_mean, ot_mean


def plot_disc_subplot(ax, t1_vectors, t2_vectors, angle):
    """Plot a single 2D disc in given subplot."""
    # Calculate rotation matrix to align distributions properly:
    # 1. (mean1 + mean2)/2 orthogonal to canvas (out of page)
    # 2. mean1 left, mean2 right along x-axis
    mean1 = np.mean(t1_vectors, axis=0)
    mean2 = np.mean(t2_vectors, axis=0)
    avg_mean = (mean1 + mean2) / 2
    avg_mean = avg_mean / np.linalg.norm(avg_mean)

    # First rotation: align average with z-axis
    target_z = np.array([0, 0, 1])
    v1 = np.cross(avg_mean, target_z)
    s1 = np.linalg.norm(v1)
    c1 = np.dot(avg_mean, target_z)

    if s1 != 0:
        K1 = np.array([[0, -v1[2], v1[1]], [v1[2], 0, -v1[0]], [-v1[1], v1[0], 0]])
        R1 = np.eye(3) + K1 + K1 @ K1 * (1 - c1) / (s1 * s1)
    else:
        R1 = np.eye(3)

    # Apply first rotation
    mean1_temp = R1 @ mean1
    mean2_temp = R1 @ mean2

    # Second rotation: align mean1-mean2 with x-axis
    diff = mean2_temp - mean1_temp
    angle_rot = np.arctan2(diff[1], diff[0])

    # Rotation around z-axis
    cos_a = np.cos(-angle_rot)
    sin_a = np.sin(-angle_rot)
    R2 = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

    # Combined rotation
    R = R2 @ R1

    # Apply rotation to both distributions
    t1_rot = (R @ t1_vectors.T).T
    t2_rot = (R @ t2_vectors.T).T
    mean1_rot = R @ mean1
    mean2_rot = R @ mean2

    # Plot filled disk
    disk = plt.Circle(
        (0, 0),
        1,
        fill=True,
        facecolor="lightgray",
        alpha=0.3,
        edgecolor="lightgray",
        linewidth=1,
    )
    ax.add_patch(disk)

    # Plot samples and means
    ax.scatter(t1_rot[:, 0], t1_rot[:, 1], c="blue", s=10, alpha=0.6, marker="o")
    ax.scatter(t2_rot[:, 0], t2_rot[:, 1], c="red", s=10, alpha=0.6, marker="^")
    ax.scatter(mean1_rot[0], mean1_rot[1], c="darkblue", s=50, marker="o")
    ax.scatter(mean2_rot[0], mean2_rot[1], c="darkred", s=50, marker="^")

    # Set equal aspect and limits
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_aspect("equal")
    ax.axis("off")


def visualize_distributions(apd_results, ot_results, rotation_angles):
    """Visualize the base distribution and its rotations using first simulation."""
    # Use the same parameters as the main simulation
    kappa = 80.0
    n_samples = 100
    t1_vectors = create_elongated_distribution(
        n_samples, kappa, scale_factor=4.0, axis=1
    )

    # Only show 3 specific rotations for visualization
    selected_angles = [30, 60, 90]
    labels = ["A", "B", "C"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (angle, label) in enumerate(zip(selected_angles, labels)):
        # Create rotated version
        t2_vectors = rotate_vectors_around_mean(t1_vectors, angle)

        # Plot disc
        plot_disc_subplot(axes[i], t1_vectors, t2_vectors, angle)

        # Add label
        axes[i].text(
            0.1,
            0.9,
            label,
            transform=axes[i].transAxes,
            fontsize=22,
            fontweight="bold",
            va="top",
        )
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].spines["bottom"].set_visible(False)
        axes[i].spines["left"].set_visible(False)

    plt.tight_layout(pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("rotation_visualization.png", dpi=300, bbox_inches="tight")
    plt.savefig("rotation_visualization.eps", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    sns.set_style("whitegrid")

    # Configuration
    kappa = 80.0
    n_samples = 100
    n_simulations = 200

    print("Simulating rotation effects on distance metrics...")
    print(
        f"Configuration: κ={kappa}, n_samples={n_samples}, n_simulations={n_simulations}"
    )

    # Run simulation
    apd_results, ot_results, rotation_angles = simulate_rotation_effects(
        kappa, n_samples, n_simulations
    )

    # Plot results
    print("\nPlotting results...")
    apd_mean, ot_mean = plot_rotation_effects(apd_results, ot_results, rotation_angles)

    # Print summary statistics
    print("\nDistance Metrics by Rotation Angle:")
    print("Angle (°) |   APD   |   OT")
    print("-" * 30)
    for i, angle in enumerate(rotation_angles):
        print(f"{angle:8d} | {apd_mean[i]:7.4f} | {ot_mean[i]:7.4f}")

    print(f"\nAPD range: {apd_mean.min():.4f} - {apd_mean.max():.4f}")
    print(f"OT range: {ot_mean.min():.4f} - {ot_mean.max():.4f}")

    # Generate visualization
    print("\nGenerating distribution visualization...")
    visualize_distributions(apd_results, ot_results, rotation_angles)

    print("\nSimulation complete!")
    print("Files saved:")
    print("- rotation_vs_distance.png/.eps")
    print("- rotation_visualization.png/.eps")


if __name__ == "__main__":
    main()
