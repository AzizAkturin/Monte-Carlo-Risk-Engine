from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from pathlib import Path

# Known crypto market stress events shown as annotations on the rolling VaR chart
_CRASH_EVENTS: list[tuple[str, str]] = [
    ("2022-01-21", "Crypto\nWinter"),
    ("2022-05-09", "Luna\nCrash"),
    ("2022-11-08", "FTX\nCollapse"),
    ("2024-08-05", "Yen Carry\nUnwind"),
]


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
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # 1. Simulation Paths (top, spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])
    n_paths, horizon = simulation_df.shape
    
    if n_paths > max_paths:
        indices = np.random.choice(n_paths, max_paths, replace=False)
        paths_to_plot = simulation_df[indices]
    else:
        paths_to_plot = simulation_df
    
    # Plot paths with better coloring
    for i in range(len(paths_to_plot)):
        ax1.plot(paths_to_plot[i], alpha=0.25, linewidth=0.6, color='steelblue')
    
    # Add confidence bands
    p5 = np.percentile(simulation_df, 5, axis=0)
    p95 = np.percentile(simulation_df, 95, axis=0)
    ax1.fill_between(range(horizon), p5, p95, alpha=0.15, color='orange', label='90% Confidence Band')
    
    mean_path = simulation_df.mean(axis=0)
    ax1.plot(mean_path, color='darkgreen', linewidth=3, label='Average Outcome', zorder=10)
    ax1.axhline(y=initial_value, color='red', linestyle='--', linewidth=2, label=f'Starting Value (${initial_value:,.0f})', alpha=0.8)
    
    ax1.set_xlabel('Days into Future', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'📈 Portfolio Simulation: {n_paths:,} Possible Futures Over {horizon} Days', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    # 2. P&L Distribution (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.hist(pnl, bins=80, alpha=0.75, color='mediumseagreen', edgecolor='black', linewidth=0.5)
    ax2.axvline(-var_95, color='darkorange', linestyle='--', linewidth=2.5, 
                label=f'VaR 95%: ${var_95:,.0f} loss', zorder=5)
    ax2.axvline(-var_99, color='darkred', linestyle='--', linewidth=2.5, 
                label=f'VaR 99%: ${var_99:,.0f} loss', zorder=5)
    ax2.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.6, label='Break Even')
    
    # Add shading for loss region
    ax2.axvspan(pnl.min(), 0, alpha=0.1, color='red')
    ax2.axvspan(0, pnl.max(), alpha=0.1, color='green')
    
    ax2.set_xlabel('Profit/Loss ($)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Scenarios', fontsize=11, fontweight='bold')
    ax2.set_title('💰 Final Outcomes After 30 Days\n(What you could gain or lose)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.25, linestyle='--')
    
    # Add text annotation
    prob_loss = (pnl < 0).mean()
    ax2.text(0.98, 0.97, f'Chance of Loss: {prob_loss:.1%}', 
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, fontweight='bold')
    
    ax2.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    # 3. Percentile Bands (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    
    percentiles = [5, 25, 50, 75, 95]
    colors = ['darkred', 'orangered', 'gold', 'limegreen', 'darkgreen']
    labels = [
        '5th (Very Bad: 1 in 20)',
        '25th (Below Average)',
        '50th (Median/Typical)',
        '75th (Above Average)', 
        '95th (Very Good: 1 in 20)'
    ]
    
    for p, color, label in zip(percentiles, colors, labels):
        path = np.percentile(simulation_df, p, axis=0)
        ax3.plot(path, label=label, linewidth=2.5, color=color)
    
    ax3.axhline(y=initial_value, color='black', linestyle='--', linewidth=2, 
                alpha=0.7, label='Starting Value')
    ax3.set_xlabel('Days into Future', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
    ax3.set_title('📊 Best to Worst Case Scenarios\n(How good or bad it could get)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='best', fontsize=8.5, framealpha=0.9)
    ax3.grid(True, alpha=0.25, linestyle='--')
    ax3.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    # 4. Risk Metrics Table (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    metrics_data = [
        ['Risk Metric', 'Value', 'What It Means'],
        ['VaR 95%', f'${var_95:,.0f}', '95% chance loss ≤ this'],
        ['CVaR 95%', f'${cvar_95:,.0f}', 'Avg loss if in worst 5%'],
        ['VaR 99%', f'${var_99:,.0f}', '99% chance loss ≤ this'],
        ['CVaR 99%', f'${cvar_99:,.0f}', 'Avg loss if in worst 1%'],
        ['Expected P&L', f'${pnl.mean():+,.0f}', 'Average outcome'],
        ['Volatility', f'${pnl.std():,.0f}', 'Typical swing size'],
        ['Loss Probability', f'{(pnl < 0).mean():.1%}', 'Chance of losing money'],
    ]
    
    table = ax4.table(cellText=metrics_data, cellLoc='left', loc='center',
                     colWidths=[0.35, 0.25, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.8)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#2c5282')
        table[(0, i)].set_text_props(weight='bold', color='white', size=10)
    
    # Color code risk levels
    for row in range(1, 5):
        table[(row, 0)].set_facecolor('#fff5f5')
        table[(row, 1)].set_facecolor('#fff5f5')
        table[(row, 1)].set_text_props(weight='bold', color='darkred')
    
    ax4.set_title('⚠️  Key Risk Metrics - Read This First!', 
                  fontweight='bold', fontsize=12, pad=15)
    
    # 5. Drawdown Analysis (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    running_max = np.maximum.accumulate(simulation_df, axis=1)
    drawdowns = (simulation_df - running_max) / running_max
    worst_dd = drawdowns.min(axis=1)
    
    ax5.hist(worst_dd * 100, bins=80, alpha=0.75, color='indianred', edgecolor='darkred', linewidth=0.5)
    
    dd_5th = np.percentile(worst_dd * 100, 5)
    dd_median = np.percentile(worst_dd * 100, 50)
    
    ax5.axvline(dd_5th, color='darkred', linestyle='--', linewidth=2.5, 
                label=f'Worst 5%: {dd_5th:.1f}%', zorder=5)
    ax5.axvline(dd_median, color='orange', linestyle='--', linewidth=2.5,
                label=f'Typical: {dd_median:.1f}%', zorder=5)
    
    ax5.set_xlabel('Maximum Drawdown (%)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Number of Scenarios', fontsize=11, fontweight='bold')
    ax5.set_title('📉 Maximum Drawdown\n(Biggest peak-to-trough drop)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax5.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax5.grid(True, alpha=0.25, linestyle='--')
    
    # Add explanation box
    ax5.text(0.98, 0.65, 
             'Drawdown = How much\nyour portfolio drops\nfrom its highest point', 
             transform=ax5.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
             fontsize=8.5)
    
    plt.suptitle('Portfolio Risk Dashboard - 30-Day Monte Carlo Simulation\n' + 
                 f'Analyzing {n_paths:,} possible futures starting with ${initial_value:,.0f}',
                 fontsize=15, fontweight='bold', y=0.998)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_regime_comparison(
    mu_flat: np.ndarray,
    mu_ewma: np.ndarray,
    cov_flat: np.ndarray,
    cov_ewma: np.ndarray,
    pnl_flat: np.ndarray,
    pnl_ewma: np.ndarray,
    var_95_flat: float,
    var_95_ewma: float,
    var_99_flat: float,
    var_99_ewma: float,
    asset_names: list[str],
    initial_value: float = 1.0,
    ewma_span: int = 60,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Three-panel figure showing how EWMA regime-awareness shifts the
    simulation relative to the flat (equal-weight) historical baseline.

    Panel 1 – Annualised drift per asset (flat vs EWMA)
    Panel 2 – Annualised volatility per asset (flat vs EWMA)
    Panel 3 – Overlaid terminal P&L distributions with VaR markers
    """
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.3)

    ax_drift = fig.add_subplot(gs[0, 0])
    ax_vol   = fig.add_subplot(gs[0, 1])
    ax_pnl   = fig.add_subplot(gs[1, :])

    n = len(asset_names)
    x = np.arange(n)
    bar_w = 0.35

    # ── Panel 1: Annualised drift ─────────────────────────────────────
    drift_flat = mu_flat * 252 * 100
    drift_ewma = mu_ewma * 252 * 100

    bars_f = ax_drift.bar(x - bar_w / 2, drift_flat, bar_w,
                          label='Flat (full history)', color='steelblue', alpha=0.85)
    bars_e = ax_drift.bar(x + bar_w / 2, drift_ewma, bar_w,
                          label=f'EWMA (span={ewma_span}d)', color='darkorange', alpha=0.85)

    ax_drift.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_drift.set_xticks(x)
    ax_drift.set_xticklabels(asset_names, fontsize=9)
    ax_drift.set_ylabel('Annualised Return (%)', fontsize=10, fontweight='bold')
    ax_drift.set_title('Drift: Current Regime vs Long-Run Average',
                       fontsize=11, fontweight='bold', pad=10)
    ax_drift.legend(fontsize=9, framealpha=0.9)
    ax_drift.grid(True, axis='y', alpha=0.25, linestyle='--')
    ax_drift.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:+.1f}%'))

    # Label each bar with its value
    for bar in list(bars_f) + list(bars_e):
        h = bar.get_height()
        ax_drift.text(bar.get_x() + bar.get_width() / 2, h + (1.5 if h >= 0 else -3.5),
                      f'{h:+.1f}%', ha='center', va='bottom', fontsize=7.5)

    # ── Panel 2: Annualised volatility ────────────────────────────────
    vol_flat = np.sqrt(np.diag(cov_flat)) * np.sqrt(252) * 100
    vol_ewma = np.sqrt(np.diag(cov_ewma)) * np.sqrt(252) * 100

    bars_vf = ax_vol.bar(x - bar_w / 2, vol_flat, bar_w,
                         label='Flat (full history)', color='steelblue', alpha=0.85)
    bars_ve = ax_vol.bar(x + bar_w / 2, vol_ewma, bar_w,
                         label=f'EWMA (span={ewma_span}d)', color='darkorange', alpha=0.85)

    ax_vol.set_xticks(x)
    ax_vol.set_xticklabels(asset_names, fontsize=9)
    ax_vol.set_ylabel('Annualised Volatility (%)', fontsize=10, fontweight='bold')
    ax_vol.set_title('Volatility: Current Regime vs Long-Run Average',
                     fontsize=11, fontweight='bold', pad=10)
    ax_vol.legend(fontsize=9, framealpha=0.9)
    ax_vol.grid(True, axis='y', alpha=0.25, linestyle='--')
    ax_vol.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:.1f}%'))

    for bar in list(bars_vf) + list(bars_ve):
        h = bar.get_height()
        ax_vol.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=7.5)

    # ── Panel 3: Overlaid P&L distributions ──────────────────────────
    pnl_min = min(pnl_flat.min(), pnl_ewma.min())
    pnl_max = max(pnl_flat.max(), pnl_ewma.max())
    bins = np.linspace(pnl_min, pnl_max, 100)

    ax_pnl.hist(pnl_flat, bins=bins, alpha=0.45, color='steelblue',
                label='Flat (full history)', density=True)
    ax_pnl.hist(pnl_ewma, bins=bins, alpha=0.45, color='darkorange',
                label=f'EWMA (span={ewma_span}d)', density=True)

    # VaR markers for both
    ax_pnl.axvline(-var_95_flat, color='steelblue', linestyle='--', linewidth=2,
                   label=f'VaR 95% flat: ${var_95_flat:,.0f}')
    ax_pnl.axvline(-var_95_ewma, color='darkorange', linestyle='--', linewidth=2,
                   label=f'VaR 95% EWMA: ${var_95_ewma:,.0f}')
    ax_pnl.axvline(-var_99_flat, color='steelblue', linestyle=':', linewidth=2,
                   label=f'VaR 99% flat: ${var_99_flat:,.0f}')
    ax_pnl.axvline(-var_99_ewma, color='darkorange', linestyle=':', linewidth=2,
                   label=f'VaR 99% EWMA: ${var_99_ewma:,.0f}')
    ax_pnl.axvline(0, color='black', linewidth=1.2, alpha=0.6)

    # Annotate shift in expected P&L
    delta_mean = pnl_ewma.mean() - pnl_flat.mean()
    ax_pnl.text(0.01, 0.97,
                f'EWMA mean P&L shift vs flat: ${delta_mean:+,.0f}',
                transform=ax_pnl.transAxes, va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax_pnl.set_xlabel('Terminal Profit / Loss ($)', fontsize=11, fontweight='bold')
    ax_pnl.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax_pnl.set_title('P&L Distribution: How Regime-Awareness Shifts the Simulation',
                     fontsize=12, fontweight='bold', pad=10)
    ax_pnl.legend(fontsize=9, framealpha=0.9, ncol=2)
    ax_pnl.grid(True, alpha=0.25, linestyle='--')
    ax_pnl.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'${v:,.0f}'))

    plt.suptitle(
        f'Regime Comparison — Flat (full history) vs EWMA (recent {ewma_span} days)\n'
        f'Starting portfolio: ${initial_value:,.0f}',
        fontsize=14, fontweight='bold', y=1.01,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Regime comparison saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_rolling_var(
    rolling_var: pd.DataFrame,
    current_var_95: float,
    current_var_99: float,
    window: int = 60,
    horizon_days: int = 30,
    initial_value: float = 1.0,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot rolling parametric VaR (95% and 99%) over time.

    Shows whether portfolio risk is rising, falling, or historically unusual
    compared to the current MC estimate. Known crypto stress events are
    annotated so you can see how the model reacted to real crashes.

    Args:
        rolling_var:    DataFrame with DatetimeIndex and columns var_95, var_99.
        current_var_95: Today's MC-based 95% VaR (horizontal reference line).
        current_var_99: Today's MC-based 99% VaR (horizontal reference line).
        window:         Rolling look-back window in days (for axis label).
        horizon_days:   Simulation horizon (for axis label).
        initial_value:  Starting portfolio value.
        save_path:      Optional file path to save the figure.
        show:           Whether to call plt.show().
    """
    dates = rolling_var.index
    if hasattr(dates, 'tz') and dates.tz is not None:
        dates = dates.tz_localize(None)
        rolling_var = rolling_var.copy()
        rolling_var.index = dates

    fig, ax = plt.subplots(figsize=(16, 6))

    # ── Fill between VaR 95 and 99 to highlight uncertainty band ─────
    ax.fill_between(dates, rolling_var["var_95"], rolling_var["var_99"],
                    alpha=0.18, color="red", label="VaR 95–99% band")

    # ── Rolling VaR lines ─────────────────────────────────────────────
    ax.plot(dates, rolling_var["var_95"], color="darkorange", linewidth=1.8,
            label=f"Rolling VaR 95% ({window}d window)")
    ax.plot(dates, rolling_var["var_99"], color="darkred", linewidth=1.8,
            label=f"Rolling VaR 99% ({window}d window)")

    # ── Current MC VaR reference lines ───────────────────────────────
    ax.axhline(current_var_95, color="darkorange", linestyle="--", linewidth=1.5,
               alpha=0.75, label=f"Current MC VaR 95%: ${current_var_95:,.0f}")
    ax.axhline(current_var_99, color="darkred", linestyle="--", linewidth=1.5,
               alpha=0.75, label=f"Current MC VaR 99%: ${current_var_99:,.0f}")

    # ── Annotate known crash events that fall within the date range ───
    y_max = rolling_var["var_99"].max()
    y_min = rolling_var["var_95"].min()
    y_range = y_max - y_min

    for date_str, label in _CRASH_EVENTS:
        event_date = pd.Timestamp(date_str)
        if dates[0] <= event_date <= dates[-1]:
            ax.axvline(event_date, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
            ax.text(event_date, y_max + y_range * 0.04, label,
                    ha="center", va="bottom", fontsize=7.5, color="gray",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    # ── Shade periods where rolling VaR 99 is in the top quartile ────
    q75 = rolling_var["var_99"].quantile(0.75)
    high_risk = rolling_var["var_99"] >= q75
    ax.fill_between(dates, 0, rolling_var["var_99"],
                    where=high_risk, alpha=0.07, color="red",
                    label="Elevated risk periods (top 25%)")

    ax.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{horizon_days}-Day VaR ($)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Rolling {horizon_days}-Day Parametric VaR  —  {window}-Day Look-Back Window\n"
        f"Starting portfolio: ${initial_value:,.0f}  |  Higher = more risk",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Rolling VaR chart saved to {save_path}")

    if show:
        plt.show()

    return fig
