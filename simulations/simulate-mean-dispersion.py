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


def jsd_distance(mu1, kappa1, mu2, kappa2, n_grid=20):
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

    vmf1 = vonmises_fisher(mu1, kappa1)
    vmf2 = vonmises_fisher(mu2, kappa2)

    pdf1 = vmf1.pdf(vertices)
    pdf2 = vmf2.pdf(vertices)

    pdf1 = pdf1 / np.sum(pdf1)
    pdf2 = pdf2 / np.sum(pdf2)

    return jensenshannon(pdf1.flatten(), pdf2.flatten())


def simulate_metrics(kappa_values, mu, kappa_fixed=50, n_samples=50, n_simulations=100):
    """Simulate APD, OT and JSD metrics for different concentration parameters."""
    apd_results = []
    ot_results = []
    js_results = []

    for kappa in tqdm(kappa_values, desc="Simulating kappa values"):
        apd_vals = []
        ot_vals = []

        for _ in tqdm(range(n_simulations), desc=f"κ={kappa:.2f}", leave=False):
            vectors1 = generate_vonmises_fisher_samples(n_samples, kappa_fixed, mu)
            vectors2 = generate_vonmises_fisher_samples(n_samples, kappa, mu)

            apd_vals.append(apd_distance(vectors1, vectors2))
            ot_vals.append(ot_distance(vectors1, vectors2))

        apd_results.append(apd_vals)
        ot_results.append(ot_vals)

        jsd_dist = jsd_distance(mu, kappa_fixed, mu, kappa)
        js_results.append([jsd_dist] * n_simulations)

    return np.array(apd_results), np.array(ot_results), np.array(js_results)


def plot_disc_subplot(ax, kappa1, kappa2, mu, n_samples):
    """Plot a single 2D disc in given subplot."""
    # Generate samples
    vectors1 = generate_vonmises_fisher_samples(n_samples, kappa1, mu)
    vectors2 = generate_vonmises_fisher_samples(n_samples, kappa2, mu)

    # For same mean, we just project onto 2D plane
    # Use x-y plane for visualization
    vectors1_2d = vectors1[:, :2]
    vectors2_2d = vectors2[:, :2]
    mu_2d = mu[:2]

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
        vectors1_2d[:, 0], vectors1_2d[:, 1], c="blue", s=10, alpha=0.6, marker="o"
    )
    ax.scatter(
        vectors2_2d[:, 0], vectors2_2d[:, 1], c="red", s=10, alpha=0.6, marker="^"
    )
    ax.scatter(mu_2d[0], mu_2d[1], c="black", s=50, marker="*")

    # Set equal aspect and limits
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    sns.set_style("whitegrid")

    kappa_values = np.array([1, 2, 5, 10, 20, 50, 80, 100])
    kappa_fixed = 10
    n_samples = 100
    n_simulations = 100

    # Define mean direction (same for both distributions)
    mu = np.array([1.0, 0.0, 0.0])

    print("Running simulation...")
    apd_results, ot_results, js_results = simulate_metrics(
        kappa_values, mu, kappa_fixed, n_samples, n_simulations
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
    plt.xlabel("Variable Concentration Parameter κ (log scale)", fontsize=14)
    plt.ylabel("Distribution Distance", fontsize=14)
    plt.xticks(kappa_values, [f"{k}" for k in kappa_values], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("mean_dispersion_simulation.png", dpi=300, bbox_inches="tight")
    plt.savefig("mean_dispersion_simulation.eps", dpi=300, bbox_inches="tight")
    plt.show()

    # 2D disc visualization for selected kappa values
    print("\nGenerating 2D disc visualization...")
    selected_kappas = [1, 10, 50, 100]
    labels = ["A", "B", "C", "D"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    sns.set_style("white")

    for i, (kappa, label) in enumerate(zip(selected_kappas, labels)):
        print(f"Creating disc for κ={kappa} ({label})...")
        plot_disc_subplot(axes[i], kappa_fixed, kappa, mu, n_samples)
        axes[i].text(
            0.1,
            0.9,
            label,
            transform=axes[i].transAxes,
            fontsize=22,
            fontweight="bold",
            va="top",
        )
        axes[i].set_title(f"κ={kappa}", fontsize=14)
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
    plt.savefig("mean_dispersion_combined.png", dpi=300, bbox_inches="tight")
    plt.savefig("mean_dispersion_combined.eps", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved mean_dispersion_combined.png and mean_dispersion_combined.eps")

    # Summary statistics
    print("\nSummary Statistics:")
    print(f"Variable kappa range: {kappa_values.min():.2f} - {kappa_values.max():.2f}")
    print(f"Fixed kappa: {kappa_fixed}")
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
