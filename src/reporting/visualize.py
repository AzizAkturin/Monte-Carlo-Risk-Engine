from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_simulation_paths(
    simulation_df: np.ndarray,
    initial_value: float = 1.0,
    max_paths: int = 250,
    figsize: tuple[int, int] = (12, 7),
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot Monte Carlo simulation paths (fan chart).
    
    Args:
        simulation_df: Array of shape (n_paths, horizon_days) with portfolio values
        initial_value: Starting portfolio value
        max_paths: Maximum number of paths to plot (for performance)
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    n_paths, horizon = simulation_df.shape
    
    # Subsample if too many paths
    if n_paths > max_paths:
        indices = np.random.choice(n_paths, max_paths, replace=False)
        paths_to_plot = simulation_df[indices]
    else:
        paths_to_plot = simulation_df
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot individual paths with transparency
    for i in range(len(paths_to_plot)):
        ax.plot(paths_to_plot[i], alpha=0.3, linewidth=0.8)
    
    # Plot mean path in bold
    mean_path = simulation_df.mean(axis=0)
    ax.plot(mean_path, color='black', linewidth=2.5, label=f'Mean Path', zorder=10)
    
    # Add percentile bands
    p5 = np.percentile(simulation_df, 5, axis=0)
    p95 = np.percentile(simulation_df, 95, axis=0)
    ax.fill_between(range(horizon), p5, p95, alpha=0.2, color='gray', label='5th-95th Percentile')
    
    # Add initial value line
    ax.axhline(y=initial_value, color='red', linestyle='--', linewidth=1.5, label='Initial Value', alpha=0.7)
    
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Portfolio Value', fontsize=12)
    ax.set_title(f'Monte Carlo Simulation - {n_paths:,} Paths', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_pnl_distribution(
    pnl: np.ndarray,
    var_95: float,
    cvar_95: float,
    var_99: float,
    cvar_99: float,
    figsize: tuple[int, int] = (12, 6),
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot P&L distribution with VaR/CVaR markers.
    
    Args:
        pnl: Array of terminal P&L values
        var_95, cvar_95, var_99, cvar_99: Risk metrics
        figsize: Figure size
        save_path: Optional path to save the figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram of P&L
    ax.hist(pnl, bins=100, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    
    # Add VaR/CVaR lines
    ax.axvline(-var_95, color='orange', linestyle='--', linewidth=2, label=f'VaR 95%: {var_95:.4f}')
    ax.axvline(-var_99, color='red', linestyle='--', linewidth=2, label=f'VaR 99%: {var_99:.4f}')
    ax.axvline(-cvar_95, color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f'CVaR 95%: {cvar_95:.4f}')
    ax.axvline(-cvar_99, color='red', linestyle=':', linewidth=2, alpha=0.7, label=f'CVaR 99%: {cvar_99:.4f}')
    
    # Add zero line
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Profit & Loss', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Terminal P&L Distribution with Risk Metrics', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_risk_dashboard(
    simulation_df: np.ndarray,
    pnl: np.ndarray,
    var_95: float,
    cvar_95: float,
    var_99: float,
    cvar_99: float,
    initial_value: float = 1.0,
    max_paths: int = 250,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create a comprehensive risk dashboard with multiple subplots.
    
    Args:
        simulation_df: Array of shape (n_paths, horizon_days) with portfolio values
        pnl: Array of terminal P&L values
        var_95, cvar_95, var_99, cvar_99: Risk metrics
        initial_value: Starting portfolio value
        max_paths: Maximum paths to plot
        save_path: Optional path to save the figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Simulation Paths (top, spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])
    n_paths, horizon = simulation_df.shape
    
    if n_paths > max_paths:
        indices = np.random.choice(n_paths, max_paths, replace=False)
        paths_to_plot = simulation_df[indices]
    else:
        paths_to_plot = simulation_df
    
    for i in range(len(paths_to_plot)):
        ax1.plot(paths_to_plot[i], alpha=0.3, linewidth=0.8)
    
    mean_path = simulation_df.mean(axis=0)
    ax1.plot(mean_path, color='black', linewidth=2.5, label='Mean', zorder=10)
    ax1.axhline(y=initial_value, color='red', linestyle='--', linewidth=1.5, label='Initial', alpha=0.7)
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Portfolio Value')
    ax1.set_title(f'Monte Carlo Simulation - {n_paths:,} Paths', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. P&L Distribution (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(pnl, bins=80, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(-var_95, color='orange', linestyle='--', linewidth=2, label=f'VaR 95%')
    ax2.axvline(-var_99, color='red', linestyle='--', linewidth=2, label=f'VaR 99%')
    ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Terminal P&L')
    ax2.set_ylabel('Frequency')
    ax2.set_title('P&L Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Percentile Bands (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        path = np.percentile(simulation_df, p, axis=0)
        ax3.plot(path, label=f'{p}th', linewidth=2)
    ax3.axhline(y=initial_value, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Portfolio Value')
    ax3.set_title('Percentile Trajectories', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Risk Metrics Table (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    metrics_data = [
        ['Metric', 'Value'],
        ['VaR 95%', f'{var_95:.4f}'],
        ['CVaR 95%', f'{cvar_95:.4f}'],
        ['VaR 99%', f'{var_99:.4f}'],
        ['CVaR 99%', f'{cvar_99:.4f}'],
        ['Mean P&L', f'{pnl.mean():.4f}'],
        ['Std Dev P&L', f'{pnl.std():.4f}'],
        ['Prob(Loss)', f'{(pnl < 0).mean():.2%}'],
    ]
    
    table = ax4.table(cellText=metrics_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Risk Metrics Summary', fontweight='bold', pad=20)
    
    # 5. Drawdown Analysis (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    running_max = np.maximum.accumulate(simulation_df, axis=1)
    drawdowns = (simulation_df - running_max) / running_max
    worst_dd = drawdowns.min(axis=1)
    
    ax5.hist(worst_dd * 100, bins=80, alpha=0.7, color='crimson', edgecolor='black')
    ax5.axvline(np.percentile(worst_dd * 100, 5), color='red', linestyle='--', 
                linewidth=2, label=f'5th Percentile: {np.percentile(worst_dd * 100, 5):.2f}%')
    ax5.set_xlabel('Maximum Drawdown (%)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Worst Drawdown Distribution', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('Monte Carlo Risk Dashboard', fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig
