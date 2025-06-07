#!/usr/bin/env python3
"""
Test script for catalog loading functionality
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Create a mock catalog for testing
def create_test_catalog():
    """Create a small test catalog"""
    print("Creating test catalog...")
    
    # Create mock data
    n_sources = 1000
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic astronomical data
    ra = np.random.uniform(149.4, 150.8, n_sources)  # COSMOS field RA range
    dec = np.random.uniform(1.6, 2.8, n_sources)     # COSMOS field Dec range
    z_peak = np.random.lognormal(mean=0.5, sigma=0.8, size=n_sources)
    z_peak = np.clip(z_peak, 0.01, 6.0)  # Realistic redshift range
    
    # Stellar masses (log scale)
    lmass = np.random.normal(10.0, 0.8, n_sources)
    lmass = np.clip(lmass, 8.0, 12.5)
    
    # UVJ colors for classification
    rf_U_V = np.random.normal(1.0, 0.5, n_sources)
    rf_V_J = np.random.normal(0.8, 0.4, n_sources)
    
    # Create DataFrame
    catalog_data = {
        'ra': ra,
        'dec': dec,
        'z_peak': z_peak,
        'lmass': lmass,
        'rf_U_V': rf_U_V,
        'rf_V_J': rf_V_J
    }
    
    df = pd.DataFrame(catalog_data)
    
    # Save to CSV
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    catalog_path = test_dir / "test_catalog.csv"
    df.to_csv(catalog_path, index=False)
    
    print(f"Test catalog created: {catalog_path}")
    print(f"Sources: {len(df)}")
    print(f"Redshift range: {df['z_peak'].min():.2f} - {df['z_peak'].max():.2f}")
    print(f"Mass range: {df['lmass'].min():.2f} - {df['lmass'].max():.2f}")
    
    return catalog_path


def create_test_config():
    """Create a test configuration"""
    import tempfile
    import os
    
    # Create temporary config file
    config_content = f"""
# Test configuration for catalog loading
[binning]
stack_all_z_at_once = true
add_foreground = true
crop_circles = true

[error_estimator]
write_simmaps = false
randomize = false

[error_estimator.bootstrap]
enabled = true
iterations = 10  # Small number for testing
initial_seed = 1

cosmology = "Planck18"

[output]
folder = "./test_output"
shortname = "test_run"

[catalog]
path = "./test_data"
file = "test_catalog.csv"

[catalog.astrometry]
ra = "ra"
dec = "dec"

[catalog.classification]
split_type = "uvj"

[catalog.classification.redshift]
id = "z_peak"
bins = [0.01, 0.5, 1.0, 1.5, 2.0, 3.0, 6.0]

[catalog.classification.stellar_mass]
id = "lmass"
bins = [8.5, 9.5, 10.0, 10.5, 11.0, 12.0]

[catalog.classification.split_params]
id = "sfg"

[catalog.classification.split_params.bins]
"U-V" = "rf_U_V"
"V-J" = "rf_V_J"

# No maps section for catalog-only testing
[maps]
"""
    
    config_path = Path("test_data/test_config.toml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Test config created: {config_path}")
    return config_path


def test_catalog_loading():
    """Test the catalog loading functionality"""
    print("=== Testing Catalog Loading ===")
    
    try:
        # Import simstack4 modules
        from simstack4.config import load_config
        from simstack4.sky_catalogs import SkyCatalogs
        
        # Create test data
        catalog_path = create_test_catalog()
        config_path = create_test_config()
        
        # Load configuration
        print("\nLoading configuration...")
        config = load_config(config_path)
        print("✓ Configuration loaded successfully")
        
        # Test with pandas backend
        print("\n--- Testing with pandas backend ---")
        catalogs_pandas = SkyCatalogs(config.catalog, backend="pandas")
        catalogs_pandas.load_catalog()
        catalogs_pandas.print_catalog_summary()
        
        # Test with polars backend (if available)
        print("\n--- Testing with polars backend ---")
        try:
            catalogs_polars = SkyCatalogs(config.catalog, backend="polars")
            catalogs_polars.load_catalog()
            catalogs_polars.print_catalog_summary()
            print("✓ Polars backend works!")
        except Exception as e:
            print(f"⚠ Polars backend failed: {e}")
        
        # Test auto backend selection
        print("\n--- Testing auto backend selection ---")
        catalogs_auto = SkyCatalogs(config.catalog, backend="auto")
        catalogs_auto.load_catalog()
        print(f"✓ Auto-selected backend: {catalogs_auto.backend}")
        
        # Test population manager
        print("\n--- Testing Population Manager ---")
        if catalogs_auto.population_manager:
            n_pops = len(catalogs_auto.population_manager)
            print(f"✓ Created {n_pops} populations")
            
            # Test getting population data
            pop_ids = list(catalogs_auto.population_manager.populations.keys())
            if pop_ids:
                test_pop = pop_ids[0]
                pop_data = catalogs_auto.get_population_data(test_pop)
                print(f"✓ Retrieved data for population: {test_pop}")
                print(f"  - {len(pop_data['ra'])} sources")
                print(f"  - RA range: {pop_data['ra'].min():.3f} - {pop_data['ra'].max():.3f}")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_catalog_loading()
    if not success:
        exit(1)