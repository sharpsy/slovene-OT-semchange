import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import vonmises_fisher
from tqdm import tqdm
import ot
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


def generate_vonmises_fisher_samples(n_samples, kappa, mu=None):
    """Generate samples from von Mises-Fisher distribution on unit sphere."""
    if mu is None:
        mu = np.array([1.0, 0.0, 0.0])  # Default mean direction

    mu = mu / np.linalg.norm(mu)  # Normalize mu
    vmf = vonmises_fisher(mu, kappa)

    return vmf.rvs(n_samples)


def apd_distance(vectors1, vectors2):
    """Average pairwise distance using mean vectors."""
    # Then calculate mean vectors
    v_t1 = np.mean(vectors1, axis=0)
    v_t2 = np.mean(vectors2, axis=0)

    # APD = 1 - cosine similarity
    return 1 - np.dot(v_t1, v_t2)


def ot_distance(vectors1, vectors2):
    """Optimal transport distance using cosine distance."""
    # Compute cost matrix (cosine distances)
    cost_matrix = 1 - np.dot(vectors1, vectors2.T)

    # Compute optimal transport distance
    ot_dist = ot.emd2([], [], cost_matrix)

    return ot_dist


def ot_entropic_distance(vectors1, vectors2, reg=0.1):
    """Optimal transport distance with entropic regularization."""
    # Compute cost matrix (cosine distances)
    cost_matrix = 1 - np.dot(vectors1, vectors2.T)

    # Uniform distributions
    a = np.ones(len(vectors1)) / len(vectors1)
    b = np.ones(len(vectors2)) / len(vectors2)

    # Compute entropic optimal transport distance
    ot_dist = ot.sinkhorn2(a, b, cost_matrix, reg)

    return ot_dist


def jsd_distance(mu1, mu2, kappa, n_grid=20):
    """Jensen-Shannon divergence between two von Mises-Fisher distributions."""
    # Create grid on sphere
    u = np.linspace(0, np.pi, n_grid)
    v = np.linspace(0, 2 * np.pi, n_grid)
    u_grid, v_grid = np.meshgrid(u, v)
    vertices = np.stack(
        [
            np.cos(v_grid) * np.sin(u_grid),
            np.sin(v_grid) * np.sin(u_grid),
            np.cos(u_grid),
        ],
        axis=2,
    )

    # Calculate PDFs
    vmf1 = vonmises_fisher(mu1, kappa)
    vmf2 = vonmises_fisher(mu2, kappa)

    pdf1 = vmf1.pdf(vertices)
    pdf2 = vmf2.pdf(vertices)

    # Normalize to create probability distributions
    pdf1 = pdf1 / np.sum(pdf1)
    pdf2 = pdf2 / np.sum(pdf2)

    # Calculate Jensen-Shannon divergence
    jsd_div = jensenshannon(pdf1.flatten(), pdf2.flatten())

    return jsd_div


def simulate_mean_distance(kappa=80, n_samples=50, n_simulations=1000):
    """Simulate APD and OT metrics for different mean distances."""
    # Define angular distances (in radians) - every 15 degrees
    angular_distances = np.deg2rad(
        np.arange(0, 181, 15)
    )  # 0 to 180 degrees in 15° steps

    apd_results = []
    ot_results = []
    ot_entropic_results = []
    js_results = []

    for angle in tqdm(angular_distances, desc="Simulating mean distances"):
        apd_vals = []
        ot_vals = []
        ot_entropic_vals = []

        for _ in tqdm(
            range(n_simulations), desc=f"θ={np.degrees(angle):.1f}°", leave=False
        ):
            # Generate two von Mises-Fisher distributions with different means
            mu1 = np.array([1.0, 0.0, 0.0])
            mu2 = np.array([np.cos(angle), np.sin(angle), 0.0])

            vectors1 = generate_vonmises_fisher_samples(n_samples, kappa, mu1)
            vectors2 = generate_vonmises_fisher_samples(n_samples, kappa, mu2)

            # Calculate metrics
            apd_vals.append(apd_distance(vectors1, vectors2))
            ot_vals.append(ot_distance(vectors1, vectors2))
            ot_entropic_vals.append(ot_entropic_distance(vectors1, vectors2))

        apd_results.append(apd_vals)
        ot_results.append(ot_vals)
        ot_entropic_results.append(ot_entropic_vals)

        # Calculate JSD divergence (deterministic, no need for simulation)
        jsd_dist = jsd_distance(mu1, mu2, kappa)
        js_results.append([jsd_dist] * n_simulations)

    return (
        np.array(apd_results),
        np.array(ot_results),
        np.array(ot_entropic_results),
        np.array(js_results),
        angular_distances,
    )


def plot_vmf_samples(ax, vectors1, vectors2, mu1, mu2, angle):
    """Plot von Mises-Fisher samples on unit sphere."""
    # Create unit sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, linewidth=0, alpha=0.1, color="lightgray"
    )

    # Plot samples
    ax.scatter(
        vectors1[:, 0],
        vectors1[:, 1],
        vectors1[:, 2],
        c="blue",
        s=10,
        alpha=0.6,
        marker="o",
        label="Distribution 1",
    )
    ax.scatter(
        vectors2[:, 0],
        vectors2[:, 1],
        vectors2[:, 2],
        c="red",
        s=10,
        alpha=0.6,
        marker="^",
        label="Distribution 2",
    )

    # Plot mean directions
    ax.scatter(mu1[0], mu1[1], mu1[2], c="darkblue", s=50, marker="o")
    ax.scatter(mu2[0], mu2[1], mu2[2], c="darkred", s=50, marker="^")

    ax.set_aspect("equal")
    # Position view such that center is between the two means
    center_direction = (mu1 + mu2) / 2
    center_direction = center_direction / np.linalg.norm(center_direction)
    azim = np.degrees(np.arctan2(center_direction[1], center_direction[0]))
    elev = np.degrees(np.arcsin(center_direction[2]))
    ax.view_init(azim=azim, elev=elev)
    ax.axis("off")
    ax.set_title(rf"$\theta={np.degrees(angle):.1f}°$")


def main():
    sns.set_style("whitegrid")

    # Configuration
    kappa = 80  # High concentration as requested
    n_samples = 50
    n_simulations = 1000

    print(f"Running simulation with κ={kappa}...")
    apd_results, ot_results, ot_entropic_results, js_results, angular_distances = (
        simulate_mean_distance(kappa, n_samples, n_simulations)
    )

    # Calculate statistics
    apd_mean = np.mean(apd_results, axis=1)
    apd_std = np.std(apd_results, axis=1)
    ot_mean = np.mean(ot_results, axis=1)
    ot_std = np.std(ot_results, axis=1)
    ot_entropic_mean = np.mean(ot_entropic_results, axis=1)
    ot_entropic_std = np.std(ot_entropic_results, axis=1)
    js_mean = np.mean(js_results, axis=1)
    js_std = np.std(js_results, axis=1)

    # Create main plot
    plt.figure(figsize=(12, 8))

    # Plot means with error bars using same style as variance script
    plt.errorbar(
        np.degrees(angular_distances),
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
        np.degrees(angular_distances),
        ot_mean,
        yerr=ot_std,
        label="OT",
        marker="s",
        capsize=5,
        alpha=0.7,
        linestyle=":",
        color="tab:orange",
    )
    plt.plot(
        np.degrees(angular_distances),
        js_mean,
        label="JSD",
        marker="^",
        alpha=0.7,
        linestyle="--",
        color="tab:green",
    )

    plt.xlabel("Angular Distance Between Means (degrees)", fontsize=14)
    plt.ylabel("Distribution Distance", fontsize=14)
    plt.xticks(np.arange(0, 181, 15), fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("mean_distance_vs_distance.png", dpi=300, bbox_inches="tight")
    plt.savefig("mean_distance_vs_distance.eps", dpi=300, bbox_inches="tight")
    plt.show()

    # Generate visualization samples for selected angles
    print("\nGenerating visualization samples...")

    mu1 = np.array([1.0, 0.0, 0.0])
    selected_angles = [
        0,
        np.pi / 6,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4,
        np.pi,
    ]  # Selected angles for visualization

    for angle in selected_angles:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        mu2 = np.array([np.cos(angle), np.sin(angle), 0.0])

        vectors1 = generate_vonmises_fisher_samples(n_samples, kappa, mu1)
        vectors2 = generate_vonmises_fisher_samples(n_samples, kappa, mu2)

        plot_vmf_samples(ax, vectors1, vectors2, mu1, mu2, angle)

        # Find the closest angle in our simulation results
        idx = np.argmin(np.abs(angular_distances - angle))

        # Print metrics info
        print(
            f"θ={np.degrees(angle):.1f}°: APD={apd_mean[idx]:.3f}, OT={ot_mean[idx]:.3f}, OT-Entropic={ot_entropic_mean[idx]:.3f}, JSD={js_mean[idx]:.3f}"
        )

        plt.tight_layout()

        # Save in both formats
        filename_base = f"rotation_angle_{np.degrees(angle):.0f}"
        plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{filename_base}.eps", dpi=300, bbox_inches="tight")
        plt.close()  # Close figure to free memory

        print(f"Saved {filename_base}.png and {filename_base}.eps")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Angular distance range: 0° - 180°")
    print(f"APD range: {apd_mean.min():.3f} - {apd_mean.max():.3f}")
    print(f"OT range: {ot_mean.min():.3f} - {ot_mean.max():.3f}")
    print(
        f"OT-Entropic range: {ot_entropic_mean.min():.3f} - {ot_entropic_mean.max():.3f}"
    )
    print(f"JSD range: {js_mean.min():.3f} - {js_mean.max():.3f}")
    print(
        f"Spearman correlation between APD and OT: {spearmanr(apd_mean, ot_mean)[0]:.3f}"
    )
    print(
        f"Spearman correlation between APD and OT-Entropic: {spearmanr(apd_mean, ot_entropic_mean)[0]:.3f}"
    )
    print(
        f"Spearman correlation between APD and JSD: {spearmanr(apd_mean, js_mean)[0]:.3f}"
    )
    print(
        f"Spearman correlation between OT and OT-Entropic: {spearmanr(ot_mean, ot_entropic_mean)[0]:.3f}"
    )
    print(
        f"Spearman correlation between OT and JSD: {spearmanr(ot_mean, js_mean)[0]:.3f}"
    )
    print(
        f"Spearman correlation between OT-Entropic and JSD: {spearmanr(ot_entropic_mean, js_mean)[0]:.3f}"
    )


if __name__ == "__main__":
    main()
