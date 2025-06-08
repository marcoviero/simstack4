"""
Complete Plotting and visualization for Simstack4

This module creates publication-quality plots from saved stacking results,
maintaining modularity so plot generation is separate from the main algorithm.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

from .results import SimstackResults, SEDResults
from .utils import setup_logging
from .exceptions.simstack_exceptions import PlotError

logger = setup_logging()


class SimstackPlots:
    """
    Create plots and visualizations from stacking results

    This class generates publication-quality plots from saved SimstackResults,
    maintaining modularity by working with saved results rather than requiring
    the full pipeline.
    """

    def __init__(self, results_object: Optional[SimstackResults] = None,
                 results_file: Optional[Path] = None, style: str = 'default'):
        """
        Initialize plotting class

        Args:
            results_object: Processed SimstackResults object
            results_file: Path to saved results file
            style: Plotting style ('default', 'publication', 'presentation')
        """
        if not HAS_MATPLOTLIB:
            raise PlotError("matplotlib is required for plotting")

        # Load results
        if results_object is not None:
            self.results = results_object
        elif results_file is not None:
            self.results = SimstackResults.load_results(results_file)
        else:
            raise PlotError("Either results_object or results_file must be provided")

        # Set up plotting style
        self.style = style
        self._setup_plotting_style()

        logger.info(f"SimstackPlots initialized with {len(self.results.sed_results)} populations")

    def _setup_plotting_style(self) -> None:
        """Set up matplotlib plotting style"""
        if self.style == 'publication':
            # Publication-ready style
            plt.style.use('seaborn-v0_8-whitegrid' if HAS_SEABORN else 'seaborn-whitegrid')
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 11,
                'figure.titlesize': 18,
                'lines.linewidth': 2,
                'lines.markersize': 6,
                'figure.figsize': [8, 6],
                'savefig.dpi': 300,
                'savefig.bbox': 'tight'
            })
        elif self.style == 'presentation':
            # Presentation style (larger fonts)
            plt.rcParams.update({
                'font.size': 16,
                'axes.labelsize': 18,
                'axes.titlesize': 20,
                'xtick.labelsize': 16,
                'ytick.labelsize': 16,
                'legend.fontsize': 14,
                'figure.titlesize': 22,
                'lines.linewidth': 3,
                'lines.markersize': 8,
                'figure.figsize': [10, 8]
            })

    def plot_stacked_seds(self, population_ids: Optional[List[str]] = None,
                         normalize_by_sources: bool = False,
                         show_errors: bool = True,
                         save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot stacked SEDs for selected populations

        Args:
            population_ids: List of population IDs to plot (None for all)
            normalize_by_sources: Whether to normalize by number of sources
            show_errors: Whether to show error bars
            save_path: Path to save plot

        Returns:
            Figure object
        """
        if population_ids is None:
            population_ids = list(self.results.sed_results.keys())

        fig, ax = plt.subplots(figsize=(10, 8))

        # Define colors for different populations
        colors = plt.cm.tab10(np.linspace(0, 1, len(population_ids)))

        for i, pop_id in enumerate(population_ids):
            if pop_id not in self.results.sed_results:
                logger.warning(f"Population {pop_id} not found in results")
                continue

            sed = self.results.sed_results[pop_id]

            # Get plotting data
            wavelengths = sed.wavelengths
            flux_densities = sed.flux_densities.copy()
            flux_errors = sed.flux_errors.copy()

            # Normalize by number of sources if requested
            if normalize_by_sources and sed.n_sources > 0:
                flux_densities /= sed.n_sources
                flux_errors /= sed.n_sources

            # Create label
            label = f"{pop_id} (N={sed.n_sources})"
            if len(label) > 40:
                label = label[:37] + "..."

            # Plot
            if show_errors:
                ax.errorbar(wavelengths, flux_densities, yerr=flux_errors,
                           color=colors[i], marker='o', linestyle='-',
                           label=label, capsize=3, capthick=1)
            else:
                ax.plot(wavelengths, flux_densities, color=colors[i],
                       marker='o', linestyle='-', label=label)

        # Formatting
        ax.set_xlabel('Observed Wavelength [μm]')
        ylabel = 'Flux Density [Jy'
        if normalize_by_sources:
            ylabel += '/source'
        ylabel += ']'
        ax.set_ylabel(ylabel)
        ax.set_title('Stacked Spectral Energy Distributions')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SED plot saved to {save_path}")

        return fig

    def plot_luminosity_functions(self, rest_wavelength: float = 100.0,
                                 save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot rest-frame luminosity functions

        Args:
            rest_wavelength: Rest-frame wavelength to use (microns)
            save_path: Path to save plot

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get luminosities at specified wavelength
        luminosities = []
        redshifts = []
        masses = []
        pop_labels = []

        for pop_id, sed in self.results.sed_results.items():
            # Find closest wavelength
            wave_idx = np.argmin(np.abs(sed.wavelengths - rest_wavelength))
            closest_wave = sed.wavelengths[wave_idx]

            if abs(closest_wave - rest_wavelength) < rest_wavelength * 0.5:  # Within 50%
                lum = sed.rest_luminosities[wave_idx]
                if lum > 0:
                    luminosities.append(lum)
                    redshifts.append(sed.median_redshift)
                    masses.append(sed.median_mass)
                    pop_labels.append(pop_id)

        if not luminosities:
            logger.warning(f"No valid luminosities found at {rest_wavelength}μm")
            return fig

        luminosities = np.array(luminosities)
        redshifts = np.array(redshifts)
        masses = np.array(masses)

        # Create scatter plot colored by redshift
        scatter = ax.scatter(masses, luminosities, c=redshifts,
                           s=60, alpha=0.7, cmap='viridis')

        # Formatting
        ax.set_xlabel('log(M*/M☉)')
        ax.set_ylabel(f'L_{rest_wavelength:.0f}μm [L☉]')
        ax.set_title(f'Rest-frame {rest_wavelength:.0f}μm Luminosity vs Stellar Mass')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Median Redshift')

        # Add population labels for highest luminosities
        if len(luminosities) <= 10:  # Only if not too crowded
            for i, (mass, lum, label) in enumerate(zip(masses, luminosities, pop_labels)):
                ax.annotate(label[:10], (mass, lum), xytext=(5, 5),
                          textcoords='offset points', fontsize=8, alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Luminosity function plot saved to {save_path}")

        return fig

    def plot_sfr_vs_mass(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot star formation rate vs stellar mass (main sequence)

        Args:
            save_path: Path to save plot

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get data from results
        summary_df = self.results.get_population_summary()

        # Filter for valid SFR measurements
        valid_mask = (summary_df['sfr_msun_yr'] > 0) & (summary_df['median_log_mass'] > 0)
        plot_data = summary_df[valid_mask]

        if len(plot_data) == 0:
            logger.warning("No valid SFR/mass data found")
            ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=16)
            return fig

        # Create scatter plot
        masses = plot_data['median_log_mass']
        sfrs = plot_data['sfr_msun_yr']
        n_sources = plot_data['n_sources']

        # Size points by number of sources
        sizes = 30 + 100 * (n_sources / n_sources.max())

        scatter = ax.scatter(masses, sfrs, s=sizes, alpha=0.7,
                           c=plot_data['median_redshift'], cmap='plasma')

        # Plot main sequence relation (Whitaker et al. 2012)
        mass_range = np.linspace(masses.min(), masses.max(), 100)
        ms_sfr = 10**(0.7 * (mass_range - 10.5) - 0.13)  # Simplified relation
        ax.plot(mass_range, ms_sfr, 'k--', alpha=0.5, linewidth=2,
               label='Main Sequence (z~1)')

        # Formatting
        ax.set_xlabel('log(M*/M☉)')
        ax.set_ylabel('SFR [M☉/yr]')
        ax.set_title('Star Formation Rate vs Stellar Mass')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Median Redshift')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SFR vs mass plot saved to {save_path}")

        return fig

    def plot_fit_quality(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot fit quality metrics across bands

        Args:
            save_path: Path to save plot

        Returns:
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Extract fit quality data
        map_names = list(self.results.band_results.keys())
        wavelengths = [self.results.band_results[name]['wavelength_um'] for name in map_names]
        chi2_values = [self.results.band_results[name]['reduced_chi_squared'] for name in map_names]
        chi2_abs = [self.results.band_results[name]['chi_squared'] for name in map_names]

        # Plot 1: Reduced chi-squared vs wavelength
        ax1.scatter(wavelengths, chi2_values, s=80, alpha=0.7, color='steelblue')
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfect fit')
        ax1.axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='Acceptable fit')

        ax1.set_xlabel('Wavelength [μm]')
        ax1.set_ylabel('Reduced χ²')
        ax1.set_title('Fit Quality by Wavelength')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add band labels
        for wave, chi2, name in zip(wavelengths, chi2_values, map_names):
            ax1.annotate(name, (wave, chi2), xytext=(5, 5),
                        textcoords='offset points', fontsize=9, alpha=0.8)

        # Plot 2: Absolute chi-squared
        ax2.bar(range(len(map_names)), chi2_abs, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Band')
        ax2.set_ylabel('χ²')
        ax2.set_title('Absolute Chi-squared by Band')
        ax2.set_xticks(range(len(map_names)))
        ax2.set_xticklabels([f"{name}\n{wave:.0f}μm" for name, wave in zip(map_names, wavelengths)],
                           rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Fit quality plot saved to {save_path}")

        return fig

    def plot_population_comparison(self, populations: List[str],
                                 metric: str = 'flux_densities',
                                 save_path: Optional[Path] = None) -> plt.Figure:
        """
        Compare specific populations across bands

        Args:
            populations: List of population IDs to compare
            metric: Metric to compare ('flux_densities', 'rest_luminosities')
            save_path: Path to save plot

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, len(populations)))

        for i, pop_id in enumerate(populations):
            if pop_id not in self.results.sed_results:
                logger.warning(f"Population {pop_id} not found")
                continue

            sed = self.results.sed_results[pop_id]

            if metric == 'flux_densities':
                y_data = sed.flux_densities
                y_errors = sed.flux_errors
                ylabel = 'Flux Density [Jy]'
            elif metric == 'rest_luminosities':
                y_data = sed.rest_luminosities
                y_errors = sed.rest_luminosity_errors
                ylabel = 'Rest Luminosity [L☉]'
            else:
                raise PlotError(f"Unknown metric: {metric}")

            # Create label
            label = f"{pop_id} (z={sed.median_redshift:.2f}, N={sed.n_sources})"
            if len(label) > 50:
                label = label[:47] + "..."

            ax.errorbar(sed.wavelengths, y_data, yerr=y_errors,
                       color=colors[i], marker='o', linestyle='-',
                       label=label, capsize=3, markersize=6)

        ax.set_xlabel('Wavelength [μm]')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Population Comparison: {metric.replace("_", " ").title()}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Population comparison plot saved to {save_path}")

        return fig

    def plot_redshift_evolution(self, quantity: str = 'sfr',
                               save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot evolution of quantities with redshift

        Args:
            quantity: Quantity to plot ('sfr', 'total_ir_luminosity', 'specific_sfr')
            save_path: Path to save plot

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get data
        summary_df = self.results.get_population_summary()

        # Map quantity names
        quantity_map = {
            'sfr': ('sfr_msun_yr', 'SFR [M☉/yr]'),
            'total_ir_luminosity': ('total_ir_luminosity_lsun', 'L_IR [L☉]'),
            'specific_sfr': ('specific_sfr_yr', 'sSFR [yr⁻¹]')
        }

        if quantity not in quantity_map:
            raise PlotError(f"Unknown quantity: {quantity}")

        col_name, ylabel = quantity_map[quantity]

        # Filter valid data
        valid_mask = (summary_df[col_name] > 0) & (summary_df['median_redshift'] > 0)
        plot_data = summary_df[valid_mask]

        if len(plot_data) == 0:
            logger.warning(f"No valid data for {quantity}")
            return fig

        # Color code by stellar mass
        masses = plot_data['median_log_mass']
        redshifts = plot_data['median_redshift']
        values = plot_data[col_name]
        n_sources = plot_data['n_sources']

        # Size by number of sources
        sizes = 30 + 100 * (n_sources / n_sources.max())

        scatter = ax.scatter(redshifts, values, c=masses, s=sizes,
                           alpha=0.7, cmap='viridis')

        ax.set_xlabel('Redshift')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel.split("[")[0].strip()} Evolution with Redshift')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log(M*/M☉)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Redshift evolution plot saved to {save_path}")

        return fig

    def create_summary_dashboard(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive summary dashboard

        Args:
            save_path: Path to save plot

        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Plot 1: SEDs (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_seds_subplot(ax1)

        # Plot 2: SFR vs Mass (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_sfr_mass_subplot(ax2)

        # Plot 3: Fit quality (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_fit_quality_subplot(ax3)

        # Plot 4: Redshift evolution (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_redshift_evolution_subplot(ax4)

        # Plot 5: Population source counts (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_source_counts_subplot(ax5)

        # Plot 6: Luminosity distribution (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_luminosity_distribution_subplot(ax6)

        # Results summary table (bottom)
        ax7 = fig.add_subplot(gs[2, :])
        self._create_results_table(ax7)

        # Main title
        fig.suptitle('Simstack4 Results Dashboard', fontsize=20, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary dashboard saved to {save_path}")

        return fig

    def _plot_seds_subplot(self, ax: plt.Axes) -> None:
        """Helper method for SEDs subplot"""
        # Plot a few representative SEDs
        pop_ids = list(self.results.sed_results.keys())[:5]  # Limit to 5
        colors = plt.cm.tab10(np.linspace(0, 1, len(pop_ids)))

        for i, pop_id in enumerate(pop_ids):
            sed = self.results.sed_results[pop_id]
            label = f"{pop_id[:15]}..." if len(pop_id) > 15 else pop_id
            ax.plot(sed.wavelengths, sed.flux_densities, 'o-',
                   color=colors[i], label=label, markersize=4)

        ax.set_xlabel('λ [μm]')
        ax.set_ylabel('S [Jy]')
        ax.set_title('Stacked SEDs')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_sfr_mass_subplot(self, ax: plt.Axes) -> None:
        """Helper method for SFR vs mass subplot"""
        summary_df = self.results.get_population_summary()
        valid_mask = (summary_df['sfr_msun_yr'] > 0) & (summary_df['median_log_mass'] > 0)

        if np.sum(valid_mask) > 0:
            plot_data = summary_df[valid_mask]
            ax.scatter(plot_data['median_log_mass'], plot_data['sfr_msun_yr'],
                      alpha=0.7, s=30)
            ax.set_yscale('log')

        ax.set_xlabel('log(M*/M☉)')
        ax.set_ylabel('SFR [M☉/yr]')
        ax.set_title('Main Sequence')
        ax.grid(True, alpha=0.3)

    def _plot_fit_quality_subplot(self, ax: plt.Axes) -> None:
        """Helper method for fit quality subplot"""
        map_names = list(self.results.band_results.keys())
        chi2_values = [self.results.band_results[name]['reduced_chi_squared'] for name in map_names]
        wavelengths = [self.results.band_results[name]['wavelength_um'] for name in map_names]

        ax.scatter(wavelengths, chi2_values, s=60, alpha=0.7)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('λ [μm]')
        ax.set_ylabel('χ²_red')
        ax.set_title('Fit Quality')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    def _plot_redshift_evolution_subplot(self, ax: plt.Axes) -> None:
        """Helper method for redshift evolution subplot"""
        summary_df = self.results.get_population_summary()
        valid_mask = (summary_df['sfr_msun_yr'] > 0) & (summary_df['median_redshift'] > 0)

        if np.sum(valid_mask) > 0:
            plot_data = summary_df[valid_mask]
            ax.scatter(plot_data['median_redshift'], plot_data['sfr_msun_yr'],
                      alpha=0.7, s=30)
            ax.set_yscale('log')

        ax.set_xlabel('Redshift')
        ax.set_ylabel('SFR [M☉/yr]')
        ax.set_title('SFR Evolution')
        ax.grid(True, alpha=0.3)

    def _plot_source_counts_subplot(self, ax: plt.Axes) -> None:
        """Helper method for source counts subplot"""
        summary_df = self.results.get_population_summary()

        # Show top 10 populations by source count
        top_pops = summary_df.nlargest(10, 'n_sources')
        pop_labels = [pid[:10] + "..." if len(pid) > 10 else pid for pid in top_pops['population_id']]

        bars = ax.bar(range(len(pop_labels)), top_pops['n_sources'], alpha=0.7)
        ax.set_xlabel('Population')
        ax.set_ylabel('N sources')
        ax.set_title('Source Counts')
        ax.set_xticks(range(len(pop_labels)))
        ax.set_xticklabels(pop_labels, rotation=45, ha='right', fontsize=8)

    def _plot_luminosity_distribution_subplot(self, ax: plt.Axes) -> None:
        """Helper method for luminosity distribution subplot"""
        summary_df = self.results.get_population_summary()
        valid_lums = summary_df[summary_df['total_ir_luminosity_lsun'] > 0]['total_ir_luminosity_lsun']

        if len(valid_lums) > 0:
            ax.hist(np.log10(valid_lums), bins=15, alpha=0.7, edgecolor='black')

        ax.set_xlabel('log(L_IR/L☉)')
        ax.set_ylabel('N populations')
        ax.set_title('L_IR Distribution')
        ax.grid(True, alpha=0.3)

    def _create_results_table(self, ax: plt.Axes) -> None:
        """Helper method to create results summary table"""
        ax.axis('off')

        # Get summary statistics
        summary_df = self.results.get_population_summary()

        if len(summary_df) == 0:
            ax.text(0.5, 0.5, 'No results to display', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return

        # Create summary statistics
        stats_data = [
            ['Total Populations', f"{len(summary_df)}"],
            ['Total Sources', f"{summary_df['n_sources'].sum():,}"],
            ['Redshift Range', f"{summary_df['median_redshift'].min():.2f} - {summary_df['median_redshift'].max():.2f}"],
            ['Mass Range', f"{summary_df['median_log_mass'].min():.1f} - {summary_df['median_log_mass'].max():.1f}"],
            ['Mean SFR', f"{summary_df['sfr_msun_yr'].mean():.2e} M☉/yr"],
            ['Mean L_IR', f"{summary_df['total_ir_luminosity_lsun'].mean():.2e} L☉"]
        ]

        # Create table
        table = ax.table(cellText=stats_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.3, 0.3])

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

        # Style the table
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2' if i % 2 == 0 else 'white')

        ax.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

    def save_all_plots(self, output_dir: Path, populations: Optional[List[str]] = None) -> None:
        """
        Generate and save all standard plots

        Args:
            output_dir: Directory to save plots
            populations: List of populations for comparison plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating all plots in {output_dir}")

        # Standard plots
        plots_to_generate = [
            ('seds', lambda: self.plot_stacked_seds()),
            ('sfr_vs_mass', lambda: self.plot_sfr_vs_mass()),
            ('fit_quality', lambda: self.plot_fit_quality()),
            ('redshift_evolution_sfr', lambda: self.plot_redshift_evolution('sfr')),
            ('luminosity_functions', lambda: self.plot_luminosity_functions()),
            ('dashboard', lambda: self.create_summary_dashboard())
        ]

        for plot_name, plot_func in plots_to_generate:
            try:
                fig = plot_func()
                save_path = output_dir / f"{plot_name}.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved {plot_name} to {save_path}")
            except Exception as e:
                logger.error(f"Failed to generate {plot_name}: {e}")

        # Population comparison if populations specified
        if populations and len(populations) > 1:
            try:
                fig = self.plot_population_comparison(populations)
                save_path = output_dir / "population_comparison.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved population comparison to {save_path}")
            except Exception as e:
                logger.error(f"Failed to generate population comparison: {e}")

        logger.info("All plots generated successfully")


def create_plots_from_results(results_path: Path, output_dir: Path,
                            style: str = 'publication') -> SimstackPlots:
    """
    Convenience function to create plots from saved results

    Args:
        results_path: Path to saved results file
        output_dir: Directory to save plots
        style: Plotting style

    Returns:
        SimstackPlots object
    """
    plotter = SimstackPlots(results_file=results_path, style=style)
    plotter.save_all_plots(output_dir)
    return plotter