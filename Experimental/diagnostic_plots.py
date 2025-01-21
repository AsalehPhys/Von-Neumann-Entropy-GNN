import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_diagnostic_plots(diagnostics, save_plots=True, save_dir="plots"):
    """
    Creates enhanced diagnostic plots using experimental quantities.
    
    Parameters
    ----------
    diagnostics : dict
        Dictionary containing:
            'predictions': np.ndarray - model predictions
            'targets': np.ndarray - true values
            'sizes': np.ndarray - system sizes
            'rydberg_density': np.ndarray - experimental Rydberg densities
            'total_density': np.ndarray - total atomic densities
    save_plots : bool
        Whether to save plots to disk
    save_dir : str
        Directory for saved plots
    """
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

    # Extract arrays
    preds = diagnostics['predictions']
    trues = diagnostics['targets']
    sizes = diagnostics['sizes']
    rydberg = diagnostics['rydberg_density']
    density = diagnostics['total_density']
    residuals = preds - trues
    rel_error = np.abs(residuals / (trues + 1e-10)) * 100

    # Calculate global metrics
    r2 = r2_score(trues, preds)
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    mape = np.mean(np.abs(residuals / (trues + 1e-10))) * 100

    # 1) Enhanced True vs. Predicted
    plt.figure(figsize=(8, 6))
    g = sns.jointplot(
        x=trues, y=preds, 
        kind='scatter',
        joint_kws={'alpha': 0.4},
        marginal_kws={'color': 'blue'},
        height=8
    )
    g.fig.suptitle(f'True vs Predicted\nR² = {r2:.4f}, MAPE = {mape:.2f}%')
    g.ax_joint.plot([trues.min(), trues.max()], [trues.min(), trues.max()], 'r--', lw=1)
    if save_plots:
        plt.savefig(os.path.join(save_dir, "true_vs_pred_enhanced.png"), dpi=120, bbox_inches='tight')
    plt.close()

    # 2) Size-specific Predictions Matrix
    unique_sz = np.unique(sizes)
    ncols = min(4, len(unique_sz))
    nrows = int(np.ceil(len(unique_sz) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sz in enumerate(unique_sz):
        i, j = divmod(idx, ncols)
        mask = (sizes == sz)
        sz_preds = preds[mask]
        sz_trues = trues[mask]
        
        # Calculate size-specific metrics
        sz_r2 = r2_score(sz_trues, sz_preds)
        sz_mape = np.mean(np.abs((sz_preds - sz_trues) / (sz_trues + 1e-10))) * 100
        
        # Create density scatter
        sns.kdeplot(
            x=sz_trues, y=sz_preds,
            cmap='viridis',
            fill=True,
            ax=axes[i, j]
        )
        axes[i, j].scatter(sz_trues, sz_preds, alpha=0.2, color='black', s=10)
        axes[i, j].plot([sz_trues.min(), sz_trues.max()], 
                       [sz_trues.min(), sz_trues.max()], 
                       'r--', lw=1)
        axes[i, j].set_title(f'Size={int(sz)}\nR²={sz_r2:.3f}, MAPE={sz_mape:.1f}%')

    for idx in range(len(unique_sz), nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].axis('off')

    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_dir, "size_specific_density.png"), dpi=120, bbox_inches='tight')
    plt.close()

    # 3) Phase Space Analysis
    plt.figure(figsize=(10, 8))
    plt.subplot(221)
    sc = plt.scatter(density, rydberg, c=preds, cmap='viridis', alpha=0.6)
    plt.colorbar(sc, label='Predicted Entropy')
    plt.xlabel('Total Density')
    plt.ylabel('Rydberg Density')
    plt.title('Phase Space: Predictions')

    plt.subplot(222)
    sc = plt.scatter(density, rydberg, c=trues, cmap='viridis', alpha=0.6)
    plt.colorbar(sc, label='True Entropy')
    plt.xlabel('Total Density')
    plt.ylabel('Rydberg Density')
    plt.title('Phase Space: True Values')

    plt.subplot(223)
    sc = plt.scatter(density, rydberg, c=residuals, cmap='coolwarm', alpha=0.6)
    plt.colorbar(sc, label='Residuals')
    plt.xlabel('Total Density')
    plt.ylabel('Rydberg Density')
    plt.title('Phase Space: Residuals')

    plt.subplot(224)
    sc = plt.scatter(density, rydberg, c=rel_error, cmap='Reds', alpha=0.6)
    plt.colorbar(sc, label='Relative Error (%)')
    plt.xlabel('Total Density')
    plt.ylabel('Rydberg Density')
    plt.title('Phase Space: Relative Error')

    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_dir, "phase_space_analysis.png"), dpi=120, bbox_inches='tight')
    plt.close()

    # 4) Error Analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[0,0], color='blue', alpha=0.6)
    axes[0,0].axvline(0, color='r', linestyle='--')
    axes[0,0].set_title('Residual Distribution')
    axes[0,0].set_xlabel('Residual (Pred - True)')
    
    # Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot of Residuals')
    
    # Residuals vs Predictions
    axes[1,0].scatter(preds, residuals, alpha=0.4)
    axes[1,0].axhline(0, color='r', linestyle='--')
    axes[1,0].set_xlabel('Predicted Values')
    axes[1,0].set_ylabel('Residuals')
    axes[1,0].set_title('Residuals vs Predictions')
    
    # Relative Error Distribution
    sns.histplot(rel_error, kde=True, ax=axes[1,1], color='green', alpha=0.6)
    axes[1,1].set_title('Relative Error Distribution')
    axes[1,1].set_xlabel('Relative Error (%)')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_dir, "error_analysis.png"), dpi=120, bbox_inches='tight')
    plt.close()

    # 5) Size and Density Effects
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    sns.boxplot(x=sizes.astype(int), y=rel_error)
    plt.xlabel('System Size')
    plt.ylabel('Relative Error (%)')
    plt.title('Error vs System Size')
    
    plt.subplot(132)
    plt.scatter(rydberg, rel_error, c=sizes, cmap='viridis', alpha=0.6)
    plt.colorbar(label='System Size')
    plt.xlabel('Rydberg Density')
    plt.ylabel('Relative Error (%)')
    plt.yscale('log')
    plt.title('Error vs Rydberg Density')
    
    plt.subplot(133)
    plt.scatter(density, rel_error, c=sizes, cmap='viridis', alpha=0.6)
    plt.colorbar(label='System Size')
    plt.xlabel('Total Density')
    plt.ylabel('Relative Error (%)')
    plt.yscale('log')
    plt.title('Error vs Total Density')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_dir, "size_density_effects.png"), dpi=120, bbox_inches='tight')
    plt.close()

    # 6) Error Heatmaps
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    unique_sizes = np.sort(np.unique(sizes))
    unique_densities = np.linspace(density.min(), density.max(), 20)
    error_matrix = np.zeros((len(unique_sizes), len(unique_densities)))
    count_matrix = np.zeros_like(error_matrix)
    
    for i, sz in enumerate(unique_sizes):
        for j, d in enumerate(unique_densities[:-1]):
            mask = (sizes == sz) & (density >= d) & (density < unique_densities[j+1])
            if mask.any():
                error_matrix[i,j] = np.mean(rel_error[mask])
                count_matrix[i,j] = np.sum(mask)
    
    sns.heatmap(error_matrix, cmap='Reds', 
                xticklabels=[f'{d:.2f}' for d in unique_densities[:-1]],
                yticklabels=unique_sizes.astype(int))
    plt.xlabel('Total Density')
    plt.ylabel('System Size')
    plt.title('Mean Relative Error')
    
    plt.subplot(132)
    sns.heatmap(np.log10(count_matrix + 1), cmap='viridis',
                xticklabels=[f'{d:.2f}' for d in unique_densities[:-1]],
                yticklabels=unique_sizes.astype(int))
    plt.xlabel('Total Density')
    plt.ylabel('System Size')
    plt.title('Log10(Sample Count)')
    
    plt.subplot(133)
    reliability = np.zeros_like(error_matrix)
    for i, sz in enumerate(unique_sizes):
        for j, d in enumerate(unique_densities[:-1]):
            mask = (sizes == sz) & (density >= d) & (density < unique_densities[j+1])
            if mask.any():
                reliability[i,j] = np.mean(rel_error[mask] < 10)  # % within 10% error
    
    sns.heatmap(reliability, cmap='RdYlGn',
                xticklabels=[f'{d:.2f}' for d in unique_densities[:-1]],
                yticklabels=unique_sizes.astype(int))
    plt.xlabel('Total Density')
    plt.ylabel('System Size')
    plt.title('Reliability (% within 10% error)')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_dir, "error_heatmaps.png"), dpi=120, bbox_inches='tight')
    plt.close()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Overall R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4e}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.4e}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print("\nSize-specific MAPE:")
    for sz in unique_sz:
        mask = sizes == sz
        sz_mape = np.mean(np.abs((preds[mask] - trues[mask]) / (trues[mask] + 1e-10))) * 100
        print(f"Size {int(sz)}: {sz_mape:.2f}%")

if __name__ == "__main__":
    print("This is a plotting utility module. Import and use create_diagnostic_plots().")