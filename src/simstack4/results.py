"""
Complete Results processing and analysis for Simstack4

This module handles processing of stacking results, applying cosmological corrections,
calculating derived quantities like luminosities and star formation rates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import warnings

from .algorithm import StackingResults
from .cosmology import CosmologyCalculator, Cosmology
from .populations import PopulationManager
from .config import SimstackConfig
from .utils import setup_logging
from .exceptions.simstack_exceptions import ResultsError

logger = setup_logging()


@dataclass
class SEDResults:
    """Spectral Energy Distribution results for a population"""
    population_id: str
    wavelengths: np.ndarray  # microns
    flux_densities: np.ndarray  # Jy
    flux_errors: np.ndarray  # Jy
    luminosity_distances: np.ndarray  # Mpc
    rest_luminosities: np.ndarray  # L_sun
    rest_luminosity_errors: np.ndarray  # L_sun
    median_redshift: float
    median_mass: float
    n_sources: int


@dataclass
class DerivedQuantities:
    """Container for derived astrophysical quantities"""
    total_ir_luminosity: float  # L_sun
    total_ir_luminosity_error: float  # L_sun
    star_formation_rate: float  # M_sun/yr
    star_formation_rate_error: float  # M_sun/yr
    specific_sfr: float  # yr^-1
    dust_temperature: Optional[float] = None  # K
    dust_mass: Optional[float] = None  # M_sun


class SimstackResults:
    """
    Process and analyze stacking results

    This class takes raw stacking results and applies cosmological corrections,
    calculates luminosities, star formation rates, and other derived quantities.
    """

    def __init__(self, config: SimstackConfig, stacking_results: StackingResults,
                 population_manager: PopulationManager, cosmology_calc: Optional[CosmologyCalculator] = None):
        """
        Initialize results processor

        Args:
            config: Simstack configuration
            stacking_results: Raw stacking results from algorithm
            population_manager: Population manager with catalog info
            cosmology_calc: Cosmology calculator (created if None)
        """
        self.config = config
        self.raw_results = stacking_results
        self.population_manager = population_manager

        # Set up cosmology
        if cosmology_calc is None:
            self.cosmology_calc = CosmologyCalculator(config.cosmology)
        else:
            self.cosmology_calc = cosmology_calc

        # Initialize processed results containers
        self.sed_results: Dict[str, SEDResults] = {}
        self.derived_quantities: Dict[str, DerivedQuantities] = {}
        self.band_results: Dict[str, Dict[str, Any]] = {}

        # Process results
        self._process_results()

    def _process_results(self) -> None:
        """Process raw results into SEDs and derived quantities"""
        logger.info("Processing stacking results...")

        # Create SEDs for each population
        for i, pop_label in enumerate(self.raw_results.population_labels):
            if pop_label == "foreground":
                continue  # Skip foreground layer

            try:
                sed_result = self._create_sed_for_population(pop_label, i)
                self.sed_results[pop_label] = sed_result

                # Calculate derived quantities
                derived = self._calculate_derived_quantities(sed_result)
                self.derived_quantities[pop_label] = derived

            except Exception as e:
                logger.warning(f"Failed to process population {pop_label}: {e}")

        # Process band-by-band results
        self._process_band_results()

        logger.info(f"Processed results for {len(self.sed_results)} populations")

    def _create_sed_for_population(self, pop_label: str, pop_index: int) -> SEDResults:
        """Create SED results for a single population"""

        # Get population info
        if pop_label not in self.population_manager.populations:
            raise ResultsError(f"Population {pop_label} not found in manager")

        pop_bin = self.population_manager.populations[pop_label]

        # Extract fluxes and errors for this population
        wavelengths = []
        flux_densities = []
        flux_errors = []

        for map_name in self.raw_results.map_names:
            # Get wavelength from config
            map_config = self.config.maps[map_name]
            wavelengths.append(map_config.wavelength)

            # Get flux and error
            flux = self.raw_results.flux_densities[map_name][pop_index]
            error = self.raw_results.flux_errors[map_name][pop_index]

            # Apply color correction
            flux *= map_config.color_correction
            error *= map_config.color_correction

            flux_densities.append(flux)
            flux_errors.append(error)

        # Convert to arrays and sort by wavelength
        wavelengths = np.array(wavelengths)
        flux_densities = np.array(flux_densities)
        flux_errors = np.array(flux_errors)

        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        flux_densities = flux_densities[sort_idx]
        flux_errors = flux_errors[sort_idx]

        # Calculate luminosity distance
        z_median = pop_bin.median_redshift
        cosmo_results = self.cosmology_calc.calculate_distances(z_median)
        luminosity_distance = cosmo_results.luminosity_distance

        # Convert to rest-frame luminosities
        rest_luminosities = np.zeros_like(flux_densities)
        rest_luminosity_errors = np.zeros_like(flux_errors)

        for i, (wave, flux, flux_err) in enumerate(zip(wavelengths, flux_densities, flux_errors)):
            if flux > 0:  # Only convert positive fluxes
                rest_lum = self.cosmology_calc.flux_to_luminosity(flux, z_median, wave)
                rest_lum_err = self.cosmology_calc.flux_to_luminosity(flux_err, z_median, wave)
                rest_luminosities[i] = rest_lum
                rest_luminosity_errors[i] = rest_lum_err

        return SEDResults(
            population_id=pop_label,
            wavelengths=wavelengths,
            flux_densities=flux_densities,
            flux_errors=flux_errors,
            luminosity_distances=np.full_like(wavelengths, luminosity_distance),
            rest_luminosities=rest_luminosities,
            rest_luminosity_errors=rest_luminosity_errors,
            median_redshift=z_median,
            median_mass=pop_bin.median_stellar_mass,
            n_sources=pop_bin.n_sources
        )

    def _calculate_derived_quantities(self, sed_result: SEDResults) -> DerivedQuantities:
        """Calculate derived astrophysical quantities from SED"""

        # Calculate total IR luminosity (integrate SED)
        # For now, use simple trapezoid rule integration
        # In practice, you might fit SED templates

        valid_mask = (sed_result.rest_luminosities > 0) & np.isfinite(sed_result.rest_luminosities)

        if np.sum(valid_mask) < 2:
            logger.warning(f"Insufficient valid points for integration in {sed_result.population_id}")
            return DerivedQuantities(
                total_ir_luminosity=0.0,
                total_ir_luminosity_error=0.0,
                star_formation_rate=0.0,
                star_formation_rate_error=0.0,
                specific_sfr=0.0
            )

        # Get valid data
        valid_waves = sed_result.wavelengths[valid_mask]
        valid_lums = sed_result.rest_luminosities[valid_mask]
        valid_errs = sed_result.rest_luminosity_errors[valid_mask]

        # Simple integration approach
        # Total IR luminosity from 8-1000 microns (approximate)
        ir_mask = (valid_waves >= 8) & (valid_waves <= 1000)

        if np.sum(ir_mask) >= 2:
            # Trapezoid integration
            ir_waves = valid_waves[ir_mask]
            ir_lums = valid_lums[ir_mask]
            ir_errs = valid_errs[ir_mask]

            # Convert to frequency space for integration
            freq_hz = 2.998e14 / ir_waves  # c/lambda in Hz
            lum_freq = ir_lums * ir_waves / 2.998e14  # Convert to L_sun/Hz

            # Sort by frequency for integration
            sort_idx = np.argsort(freq_hz)
            freq_hz_sorted = freq_hz[sort_idx]
            lum_freq_sorted = lum_freq[sort_idx]

            # Integrate using trapezoid rule
            total_ir_lum = np.trapz(lum_freq_sorted, freq_hz_sorted)

            # Simple error propagation (could be improved)
            rel_errors = ir_errs / ir_lums
            avg_rel_error = np.mean(rel_errors[ir_lums > 0])
            total_ir_lum_err = total_ir_lum * avg_rel_error

        else:
            # Fall back to sum of available IR points
            ir_band_mask = valid_waves >= 24  # At least include mid-IR
            if np.sum(ir_band_mask) > 0:
                total_ir_lum = np.sum(valid_lums[ir_band_mask])
                total_ir_lum_err = np.sqrt(np.sum(valid_errs[ir_band_mask]**2))
            else:
                total_ir_lum = 0.0
                total_ir_lum_err = 0.0

        # Convert IR luminosity to star formation rate
        # Use Kennicutt (1998) relation: SFR [M_sun/yr] = L_IR [L_sun] / 1e10
        sfr = total_ir_lum / 1e10
        sfr_err = total_ir_lum_err / 1e10

        # Calculate specific star formation rate
        stellar_mass = 10**sed_result.median_mass  # Convert from log mass
        specific_sfr = sfr / stellar_mass if stellar_mass > 0 else 0.0

        # Estimate dust temperature (very rough approximation)
        dust_temp = None
        if len(valid_waves) >= 2:
            # Use ratio of far-IR bands if available
            far_ir_mask = valid_waves >= 100
            if np.sum(far_ir_mask) >= 2:
                # Very simplified temperature estimate
                dust_temp = 20.0  # K - placeholder, would need proper SED fitting

        return DerivedQuantities(
            total_ir_luminosity=total_ir_lum,
            total_ir_luminosity_error=total_ir_lum_err,
            star_formation_rate=sfr,
            star_formation_rate_error=sfr_err,
            specific_sfr=specific_sfr,
            dust_temperature=dust_temp
        )

    def _process_band_results(self) -> None:
        """Process results for individual bands"""
        for map_name in self.raw_results.map_names:
            band_result = {
                'wavelength_um': self.config.maps[map_name].wavelength,
                'flux_densities_jy': self.raw_results.flux_densities[map_name],
                'flux_errors_jy': self.raw_results.flux_errors[map_name],
                'population_labels': self.raw_results.population_labels,
                'chi_squared': self.raw_results.chi_squared[map_name],
                'reduced_chi_squared': self.raw_results.reduced_chi_squared.get(map_name, np.nan),
                'n_sources_per_pop': [self.raw_results.n_sources.get(pop, 0)
                                     for pop in self.raw_results.population_labels]
            }
            self.band_results[map_name] = band_result

    def get_population_summary(self) -> pd.DataFrame:
        """Get summary table of all population results"""
        data = []

        for pop_id, sed_result in self.sed_results.items():
            derived = self.derived_quantities.get(pop_id)

            row = {
                'population_id': pop_id,
                'n_sources': sed_result.n_sources,
                'median_redshift': sed_result.median_redshift,
                'median_log_mass': sed_result.median_mass,
                'n_bands': len(sed_result.wavelengths),
                'total_ir_luminosity_lsun': derived.total_ir_luminosity if derived else 0,
                'sfr_msun_yr': derived.star_formation_rate if derived else 0,
                'specific_sfr_yr': derived.specific_sfr if derived else 0
            }
            data.append(row)

        return pd.DataFrame(data)

    def get_sed_table(self, population_id: str) -> pd.DataFrame:
        """Get SED data table for a specific population"""
        if population_id not in self.sed_results:
            raise ResultsError(f"Population {population_id} not found in results")

        sed = self.sed_results[population_id]

        data = {
            'wavelength_um': sed.wavelengths,
            'flux_density_jy': sed.flux_densities,
            'flux_error_jy': sed.flux_errors,
            'rest_luminosity_lsun': sed.rest_luminosities,
            'rest_luminosity_error_lsun': sed.rest_luminosity_errors
        }

        return pd.DataFrame(data)

    def save_results(self, output_path: Path, format: str = 'pickle') -> None:
        """
        Save processed results to file

        Args:
            output_path: Output file path
            format: Output format ('pickle', 'hdf5', 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'pickle':
            self._save_pickle(output_path)
        elif format == 'hdf5':
            self._save_hdf5(output_path)
        elif format == 'csv':
            self._save_csv(output_path)
        else:
            raise ResultsError(f"Unknown output format: {format}")

        logger.info(f"Results saved to {output_path}")

    def _save_pickle(self, output_path: Path) -> None:
        """Save results as pickle file"""
        results_dict = {
            'config': self.config,
            'raw_results': self.raw_results,
            'sed_results': self.sed_results,
            'derived_quantities': self.derived_quantities,
            'band_results': self.band_results,
            'population_summary': self.get_population_summary(),
            'cosmology_summary': self.cosmology_calc.get_cosmology_summary()
        }

        with open(output_path, 'wb') as f:
            pickle.dump(results_dict, f)

    def _save_hdf5(self, output_path: Path) -> None:
        """Save results as HDF5 file"""
        try:
            import h5py
        except ImportError:
            raise ResultsError("h5py required for HDF5 output")

        with h5py.File(output_path, 'w') as f:
            # Save population summary
            summary_df = self.get_population_summary()
            summary_grp = f.create_group('population_summary')
            for col in summary_df.columns:
                summary_grp.create_dataset(col, data=summary_df[col].values)

            # Save SEDs
            seds_grp = f.create_group('seds')
            for pop_id, sed in self.sed_results.items():
                sed_grp = seds_grp.create_group(pop_id)
                sed_grp.create_dataset('wavelengths', data=sed.wavelengths)
                sed_grp.create_dataset('flux_densities', data=sed.flux_densities)
                sed_grp.create_dataset('flux_errors', data=sed.flux_errors)
                sed_grp.create_dataset('rest_luminosities', data=sed.rest_luminosities)
                sed_grp.create_dataset('rest_luminosity_errors', data=sed.rest_luminosity_errors)

                # Add metadata
                sed_grp.attrs['median_redshift'] = sed.median_redshift
                sed_grp.attrs['median_mass'] = sed.median_mass
                sed_grp.attrs['n_sources'] = sed.n_sources

    def _save_csv(self, output_path: Path) -> None:
        """Save results as CSV files"""
        base_path = output_path.with_suffix('')

        # Save population summary
        summary_df = self.get_population_summary()
        summary_df.to_csv(f"{base_path}_summary.csv", index=False)

        # Save individual SEDs
        for pop_id, sed in self.sed_results.items():
            sed_df = self.get_sed_table(pop_id)
            safe_pop_id = pop_id.replace('__', '_').replace('.', 'p')
            sed_df.to_csv(f"{base_path}_sed_{safe_pop_id}.csv", index=False)

    def print_results_summary(self) -> None:
        """Print a formatted summary of results"""
        print("=== Simstack4 Results Summary ===")
        print(f"Processed {len(self.sed_results)} populations")
        print(f"Bands: {len(self.raw_results.map_names)}")
        print()

        # Print fit quality
        print("Fit Quality:")
        for map_name in self.raw_results.map_names:
            chi2 = self.raw_results.chi_squared[map_name]
            red_chi2 = self.raw_results.reduced_chi_squared.get(map_name, np.nan)
            wave = self.config.maps[map_name].wavelength
            print(f"  {map_name} ({wave}μm): χ²_red = {red_chi2:.2f}")
        print()

        # Print population results
        summary_df = self.get_population_summary()
        if len(summary_df) > 0:
            print("Population Results:")
            print(f"{'Population':<30} {'N_src':<8} {'z_med':<8} {'L_IR[L☉]':<12} {'SFR[M☉/yr]':<12}")
            print("-" * 80)

            for _, row in summary_df.iterrows():
                pop_id = row['population_id'][:29]  # Truncate long names
                n_src = int(row['n_sources'])
                z_med = row['median_redshift']
                l_ir = row['total_ir_luminosity_lsun']
                sfr = row['sfr_msun_yr']

                print(f"{pop_id:<30} {n_src:<8} {z_med:<8.2f} {l_ir:<12.2e} {sfr:<12.2e}")

    def plot_results_overview(self, save_path: Optional[Path] = None) -> None:
        """Create overview plots of results (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: SEDs
        ax1 = axes[0, 0]
        for pop_id, sed in self.sed_results.items():
            if sed.n_sources > 0:
                ax1.errorbar(sed.wavelengths, sed.flux_densities,
                           yerr=sed.flux_errors, label=pop_id[:20], marker='o')
        ax1.set_xlabel('Wavelength [μm]')
        ax1.set_ylabel('Flux Density [Jy]')
        ax1.set_title('Stacked SEDs')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: SFR vs stellar mass
        ax2 = axes[0, 1]
        summary_df = self.get_population_summary()
        if len(summary_df) > 0:
            mask = summary_df['sfr_msun_yr'] > 0
            if np.sum(mask) > 0:
                ax2.scatter(summary_df.loc[mask, 'median_log_mass'],
                          summary_df.loc[mask, 'sfr_msun_yr'])
        ax2.set_xlabel('log(M*/M☉)')
        ax2.set_ylabel('SFR [M☉/yr]')
        ax2.set_title('Star Formation Rate vs Stellar Mass')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Fit quality
        ax3 = axes[1, 0]
        map_names = list(self.band_results.keys())
        chi2_values = [self.band_results[name]['reduced_chi_squared'] for name in map_names]
        wavelengths = [self.band_results[name]['wavelength_um'] for name in map_names]

        ax3.scatter(wavelengths, chi2_values)
        ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Perfect fit')
        ax3.set_xlabel('Wavelength [μm]')
        ax3.set_ylabel('Reduced χ²')
        ax3.set_title('Fit Quality by Band')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Number of sources per population
        ax4 = axes[1, 1]
        if len(summary_df) > 0:
            pop_labels = [pid[:15] for pid in summary_df['population_id']]
            n_sources = summary_df['n_sources']

            bars = ax4.bar(range(len(pop_labels)), n_sources)
            ax4.set_xlabel('Population')
            ax4.set_ylabel('Number of Sources')
            ax4.set_title('Sources per Population')
            ax4.set_xticks(range(len(pop_labels)))
            ax4.set_xticklabels(pop_labels, rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Overview plot saved to {save_path}")
        else:
            plt.show()

    @classmethod
    def load_results(cls, file_path: Path) -> 'SimstackResults':
        """Load results from saved file"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise ResultsError(f"Results file not found: {file_path}")

        if file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                results_dict = pickle.load(f)

            # Reconstruct the results object
            # This is a simplified version - full implementation would
            # properly reconstruct all components
            logger.info(f"Loaded results from {file_path}")
            return results_dict

        else:
            raise ResultsError(f"Unsupported file format: {file_path.suffix}")


def create_results_processor(config: SimstackConfig, stacking_results: StackingResults,
                           population_manager: PopulationManager) -> SimstackResults:
    """
    Convenience function to create results processor

    Args:
        config: Simstack configuration
        stacking_results: Raw stacking results
        population_manager: Population manager

    Returns:
        Processed SimstackResults object
    """
    return SimstackResults(config, stacking_results, population_manager)