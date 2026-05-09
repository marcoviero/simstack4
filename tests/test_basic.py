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
    from simstack4.config import ClassificationConfig, SplitType, BinConfig

    config = ClassificationConfig(
        split_type=SplitType.LABELS,
        binning={
            "redshift": BinConfig(id="z", label="Redshift", bins=[0, 1, 2]),
            "stellar_mass": BinConfig(id="mass", label="Stellar Mass", bins=[9, 10, 11]),
        },
    )

    pm = PopulationManager(config, classification_config=config)
    assert len(pm.bin_configs["redshift"].bins) - 1 == 2
    assert len(pm.bin_configs["stellar_mass"].bins) - 1 == 2
    assert len(pm._create_bin_combinations()) == 4
