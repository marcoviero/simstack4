"""Basic tests to verify simstack4 installation"""

def test_import():
    """Test that main modules can be imported"""
    import simstack4
    assert simstack4.__version__ == "0.1.0"

def test_config_loading():
    """Test configuration loading (with mock file)"""
    from simstack4.config import SimstackConfig
    # This would test actual config loading with a real file
    assert True  # Placeholder

def test_population_manager():
    """Test population manager basic functionality"""
    from simstack4.populations import PopulationManager
    from simstack4.config import ClassificationConfig, SplitType, ClassificationBins
    
    # Create minimal config for testing
    config = ClassificationConfig(
        split_type=SplitType.LABELS,
        redshift=ClassificationBins(id="z", bins=[0, 1, 2]),
        stellar_mass=ClassificationBins(id="mass", bins=[9, 10, 11])
    )
    
    pm = PopulationManager(config)
    assert len(pm.redshift_bins) == 2
    assert len(pm.stellar_mass_bins) == 2
