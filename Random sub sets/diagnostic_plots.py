import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_diagnostic_plots(diagnostics, save_plots=True, save_dir="plots"):
    """
    Creates and optionally saves a variety of diagnostic plots using the
    evaluation results from `evaluate()`.

    Parameters
    ----------
    diagnostics : dict
        The dictionary returned by the `evaluate()` function, containing:
        {
            'predictions': np.ndarray,
            'targets': np.ndarray,
            'sizes': np.ndarray,
            'omegas': np.ndarray,
            'deltas': np.ndarray,
            ...
        }
    save_plots : bool
        If True, saves plots as PNG files in `save_dir`. Otherwise, just shows them.
    save_dir : str
        Directory to store the plot images if `save_plots=True`.
    """

    # Make sure the output directory exists if saving
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

    # Extract arrays
    preds = diagnostics['predictions']
    trues = diagnostics['targets']
    sizes = diagnostics['sizes']
    omegas = diagnostics['omegas']
    deltas = diagnostics['deltas']
    residuals = preds - trues

    # 1) True vs. Predicted (All Data)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=trues, y=preds, alpha=0.6, edgecolor='none')
    plt.plot([trues.min(), trues.max()], [trues.min(), trues.max()], 'r--', lw=1)
    plt.xlabel("True Entropy")
    plt.ylabel("Predicted Entropy")
    plt.title("All Samples: True vs Predicted")
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_dir, "true_vs_pred_all.png"), dpi=120)
    plt.show()

    # 2) Residuals vs. System Size
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=sizes, y=residuals, alpha=0.6, edgecolor='none')
    plt.axhline(0.0, color='r', linestyle='--')
    plt.xlabel("System Size")
    plt.ylabel("Residual (Prediction - Target)")
    plt.title("Residuals vs System Size")
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_dir, "residuals_vs_size.png"), dpi=120)
    plt.show()

    # 3) True vs. Predicted for each System Size
    unique_sz = np.unique(sizes)
    ncols = 4
    nrows = int(np.ceil(len(unique_sz) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax_idx, sz in enumerate(unique_sz):
        mask = (sizes == sz)
        ax = axes[ax_idx]
        ax.scatter(trues[mask], preds[mask], alpha=0.6, edgecolor='none')
        # identity line
        min_val = min(trues[mask].min(), preds[mask].min())
        max_val = max(trues[mask].max(), preds[mask].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
        ax.set_title(f"Size={int(sz)}")
        ax.set_xlabel("True S")
        ax.set_ylabel("Pred S")

    # Hide any unused subplots
    for i in range(ax_idx + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_dir, "true_vs_pred_by_size.png"), dpi=120)
    plt.show()

    # 4) Residual distribution (histogram)
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=30, kde=True, color='blue', alpha=0.7)
    plt.xlabel("Residual (Pred - True)")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_dir, "residual_distribution.png"), dpi=120)
    plt.show()

    # 5) Residual in (Delta, Omega) space
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(deltas, omegas, c=residuals, cmap='coolwarm', alpha=0.6, edgecolor='none')
    cbar = plt.colorbar(sc)
    cbar.set_label("Residual")
    plt.xlabel("Delta")
    plt.ylabel("Omega")
    plt.title("Residuals in (Delta, Omega) Space")
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_dir, "residuals_delta_omega.png"), dpi=120)
    plt.show()

    # 6) Additional: Boxplot of residuals by System Size
    #    This helps see the distribution of errors grouped by size
    plt.figure(figsize=(8, 4))
    data_to_plot = []
    labels = []
    for sz in unique_sz:
        data_to_plot.append(residuals[sizes == sz])
        labels.append(str(int(sz)))
    sns.boxplot(data=data_to_plot)
    plt.xticks(np.arange(len(labels)), labels)
    plt.xlabel("System Size")
    plt.ylabel("Residual")
    plt.title("Residual Boxplot by System Size")
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_dir, "residual_boxplot_by_size.png"), dpi=120)
    plt.show()

    # 7) Additional: 2D histogram (true vs. pred), for large data
    #    Helps see the density of points
    plt.figure(figsize=(6, 5))
    # Clip to avoid infinite bins if some outliers exist
    min_true_pred = min(trues.min(), preds.min())
    max_true_pred = max(trues.max(), preds.max())
    plt.hist2d(trues, preds, bins=50, range=[[min_true_pred, max_true_pred], [min_true_pred, max_true_pred]], cmap='inferno')
    plt.plot([min_true_pred, max_true_pred], [min_true_pred, max_true_pred], 'r--', lw=1)
    plt.colorbar(label="Count")
    plt.xlabel("True Entropy")
    plt.ylabel("Predicted Entropy")
    plt.title("2D Histogram: True vs Pred")
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_dir, "hist2d_true_vs_pred.png"), dpi=120)
    plt.show()

    # You can add more specialized plots as needed here.
