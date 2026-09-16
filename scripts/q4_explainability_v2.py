import matplotlib.pyplot as plt

import q4_explainability as base


def make_stability_figure_fixed(stab):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    data = [
        stab["importance_spearman_original_vs_perturbed"].to_numpy(float),
        stab["random_permutation_spearman_control"].to_numpy(float),
    ]
    ax.boxplot(data, tick_labels=["0.5% source-SD perturbation", "random permutation control"], showmeans=True)
    ax.set_ylabel("Spearman correlation of 28-feature explanation")
    ax.set_title("Explanation stability under small input perturbations")
    ax.axhline(0, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(base.FIG / "explanation_stability.png", dpi=180)
    plt.close(fig)


base.make_stability_figure = make_stability_figure_fixed

if __name__ == "__main__":
    base.main()
