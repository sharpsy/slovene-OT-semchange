import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import vonmises_fisher
from tqdm import tqdm
import ot
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, pearsonr


def generate_vonmises_fisher_samples(n_samples, kappa, mu=None):
    """Generate samples from von Mises-Fisher distribution on unit sphere."""
    if mu is None:
        mu = np.array([1.0, 0.0, 0.0])

    mu = mu / np.linalg.norm(mu)
    vmf = vonmises_fisher(mu, kappa)

    return vmf.rvs(n_samples)


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


def jsd_distance(mu1, mu2, kappa, n_grid=20):
    """Jensen-Shannon divergence between two von Mises-Fisher distributions."""
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

    vmf1 = vonmises_fisher(mu1, kappa)
    vmf2 = vonmises_fisher(mu2, kappa)

    pdf1 = vmf1.pdf(vertices)
    pdf2 = vmf2.pdf(vertices)

    pdf1 = pdf1 / np.sum(pdf1)
    pdf2 = pdf2 / np.sum(pdf2)

    return jensenshannon(pdf1.flatten(), pdf2.flatten())


def simulate_metrics(kappa_values, mu1, mu2, n_samples=50, n_simulations=1000):
    """Simulate APD and OT metrics for different concentration parameters."""
    apd_results = []
    ot_results = []
    js_results = []

    for kappa in tqdm(kappa_values, desc="Simulating kappa values"):
        apd_vals = []
        ot_vals = []

        for _ in tqdm(range(n_simulations), desc=f"κ={kappa:.2f}", leave=False):
            vectors1 = generate_vonmises_fisher_samples(n_samples, kappa, mu1)
            vectors2 = generate_vonmises_fisher_samples(n_samples, kappa, mu2)

            apd_vals.append(apd_distance(vectors1, vectors2))
            ot_vals.append(ot_distance(vectors1, vectors2))

        apd_results.append(apd_vals)
        ot_results.append(ot_vals)

        jsd_dist = jsd_distance(mu1, mu2, kappa)
        js_results.append([jsd_dist] * n_simulations)

    return np.array(apd_results), np.array(ot_results), np.array(js_results)


def plot_vmf_samples(ax, vectors1, vectors2, mu1, mu2, kappa):
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
    )
    ax.scatter(
        vectors2[:, 0],
        vectors2[:, 1],
        vectors2[:, 2],
        c="red",
        s=10,
        alpha=0.6,
        marker="^",
    )

    # Plot mean directions
    ax.scatter(mu1[0], mu1[1], mu1[2], c="darkblue", s=50, marker="o")
    ax.scatter(mu2[0], mu2[1], mu2[2], c="darkred", s=50, marker="^")

    # Set tight axis limits around the sphere
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])

    # Remove margins and set equal aspect
    ax.margins(0)
    ax.set_box_aspect([1, 1, 1])

    # Position view such that center is between the two means
    center_direction = (mu1 + mu2) / 2
    center_direction = center_direction / np.linalg.norm(center_direction)
    ax.axis("off")


def plot_disc_subplot(ax, kappa, mu1, mu2, n_samples):
    """Plot a single 2D disc in given subplot."""
    # Generate samples
    vectors1 = generate_vonmises_fisher_samples(n_samples, kappa, mu1)
    vectors2 = generate_vonmises_fisher_samples(n_samples, kappa, mu2)

    # Calculate rotation matrix to align distributions properly:
    # 1. (mu1 + mu2)/2 orthogonal to canvas (out of page)
    # 2. mu1 left, mu2 right along x-axis
    avg_mean = (mu1 + mu2) / 2
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
    mu1_temp = R1 @ mu1
    mu2_temp = R1 @ mu2

    # Second rotation: align mu1-mu2 with x-axis
    diff = mu2_temp - mu1_temp
    angle = np.arctan2(diff[1], diff[0])

    # Rotation around z-axis
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    R2 = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

    # Combined rotation
    R = R2 @ R1

    # Apply rotation to both distributions
    vectors1_rot = (R @ vectors1.T).T
    vectors2_rot = (R @ vectors2.T).T
    mu1_rot = R @ mu1
    mu2_rot = R @ mu2

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
    ax.scatter(
        vectors1_rot[:, 0], vectors1_rot[:, 1], c="blue", s=10, alpha=0.6, marker="o"
    )
    ax.scatter(
        vectors2_rot[:, 0], vectors2_rot[:, 1], c="red", s=10, alpha=0.6, marker="^"
    )
    ax.scatter(mu1_rot[0], mu1_rot[1], c="darkblue", s=50, marker="o")
    ax.scatter(mu2_rot[0], mu2_rot[1], c="darkred", s=50, marker="^")

    # Set equal aspect and limits
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    sns.set_style("whitegrid")

    kappa_values = np.array([1, 2, 5, 10, 20, 50, 100])
    n_samples = 50
    n_simulations = 1000

    # Define mean directions globally for reuse
    mu1 = np.array([1.0, 0.0, 0.0])
    mu2 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])

    print("Running simulation...")
    apd_results, ot_results, js_results = simulate_metrics(
        kappa_values, mu1, mu2, n_samples, n_simulations
    )

    apd_mean = np.mean(apd_results, axis=1)
    apd_std = np.std(apd_results, axis=1)
    ot_mean = np.mean(ot_results, axis=1)
    ot_std = np.std(ot_results, axis=1)
    js_mean = np.mean(js_results, axis=1)

    # Main plot
    plt.figure(figsize=(12, 8))
    plt.errorbar(
        kappa_values,
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
        kappa_values,
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
        kappa_values,
        js_mean,
        label="JSD",
        marker="^",
        alpha=0.7,
        linestyle="--",
        color="tab:green",
    )
    plt.xscale("log")
    plt.xlabel("Concentration Parameter κ (log scale)", fontsize=14)
    plt.ylabel("Distribution Distance", fontsize=14)
    plt.xticks(kappa_values, [f"{k}" for k in kappa_values], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("apd_vs_ot_simulation.png", dpi=300, bbox_inches="tight")
    plt.savefig("apd_vs_ot_simulation.eps", dpi=300, bbox_inches="tight")
    plt.show()

    # 2D disc visualization
    print("\nGenerating 2D disc visualization for κ=10, 50, 100...")
    selected_kappas = [10, 50, 100]
    labels = ["A", "B", "C"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.set_style("white")

    for i, (kappa, label) in enumerate(zip(selected_kappas, labels)):
        print(f"Creating disc for κ={kappa} ({label})...")
        plot_disc_subplot(axes[i], kappa, mu1, mu2, n_samples)
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

    # Print metrics once
    for kappa, label in zip(selected_kappas, labels):
        kappa_idx = np.where(kappa_values == kappa)[0][0]
        print(
            f"κ={kappa:.2f}: APD={apd_mean[kappa_idx]:.3f}, OT={ot_mean[kappa_idx]:.3f}, JSD={js_mean[kappa_idx]:.3f}"
        )

    plt.tight_layout(pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("vmf_combined.png", dpi=300, bbox_inches="tight")
    plt.savefig("vmf_combined.eps", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved vmf_combined.png and vmf_combined.eps")

    # Summary statistics
    print("\nSummary Statistics:")
    print(f"Kappa range: {kappa_values.min():.2f} - {kappa_values.max():.2f}")
    print(f"APD range: {apd_mean.min():.3f} - {apd_mean.max():.3f}")
    print(f"OT range: {ot_mean.min():.3f} - {ot_mean.max():.3f}")
    print(f"JSD range: {js_mean.min():.3f} - {js_mean.max():.3f}")
    print(
        f"Spearman correlation between APD and OT: {spearmanr(apd_mean, ot_mean)[0]:.3f}"
    )
    print(
        f"Pearson correlation between APD and OT: {pearsonr(apd_mean, ot_mean)[0]:.3f}"
    )
    print(
        f"Spearman correlation between APD and JSD: {spearmanr(apd_mean, js_mean)[0]:.3f}"
    )
    print(
        f"Pearson correlation between APD and JSD: {pearsonr(apd_mean, js_mean)[0]:.3f}"
    )
    print(
        f"Spearman correlation between OT and JSD: {spearmanr(ot_mean, js_mean)[0]:.3f}"
    )
    print(
        f"Pearson correlation between OT and JSD: {pearsonr(ot_mean, js_mean)[0]:.3f}"
    )


if __name__ == "__main__":
    main()
