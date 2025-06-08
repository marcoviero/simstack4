"""
Debug script to identify and fix config issues
"""

import tomllib
from pathlib import Path
import sys

def debug_toml_config(config_path: str):
    """Debug TOML configuration issues"""
    config_path = Path(config_path)
    
    print(f"🔍 Debugging TOML config: {config_path}")
    print("=" * 60)
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'rb') as f:
            config_dict = tomllib.load(f)
        
        print("✅ TOML file loaded successfully")
        print(f"📋 Top-level sections: {list(config_dict.keys())}")
        print()
        
        # Check error_estimator.bootstrap section specifically
        if 'error_estimator' in config_dict:
            error_est = config_dict['error_estimator']
            print("🔧 error_estimator section:")
            print(f"   Keys: {list(error_est.keys())}")
            
            if 'bootstrap' in error_est:
                bootstrap = error_est['bootstrap']
                print(f"   bootstrap keys: {list(bootstrap.keys())}")
                print(f"   bootstrap content: {bootstrap}")
                
                # Check for problematic keys
                expected_keys = {'enabled', 'iterations', 'initial_seed', 'initial_bootstrap'}
                unexpected_keys = set(bootstrap.keys()) - expected_keys
                if unexpected_keys:
                    print(f"   ⚠️  Unexpected keys: {unexpected_keys}")
                    print("   💡 These might be causing the error")
            print()
        
        # Check all sections
        all_sections = {
            'binning': ['stack_all_z_at_once', 'add_foreground', 'crop_circles'],
            'error_estimator': ['bootstrap', 'write_simmaps', 'randomize'],
            'cosmology': None,  # Should be a string
            'output': ['folder', 'shortname'],
            'catalog': ['path', 'file', 'astrometry', 'classification'],
            'maps': None  # Dynamic section
        }
        
        for section_name, expected_keys in all_sections.items():
            if section_name in config_dict:
                section = config_dict[section_name]
                print(f"✅ {section_name}: {type(section).__name__}")
                
                if isinstance(section, dict) and expected_keys:
                    for key in expected_keys:
                        if key in section:
                            print(f"   ✅ {key}: {section[key]}")
                        else:
                            print(f"   ❌ Missing: {key}")
                elif not isinstance(section, dict):
                    print(f"   Value: {section}")
            else:
                print(f"❌ Missing section: {section_name}")
            print()
        
        # Show the full structure for debugging
        print("🗂️  Full structure:")
        print_dict_structure(config_dict, indent=0)
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading TOML: {e}")
        return False

def print_dict_structure(d, indent=0):
    """Recursively print dictionary structure"""
    for key, value in d.items():
        print("  " * indent + f"├── {key}: {type(value).__name__}")
        if isinstance(value, dict) and indent < 3:  # Limit depth
            print_dict_structure(value, indent + 1)
        elif isinstance(value, (list, tuple)):
            print("  " * (indent + 1) + f"└── [{len(value)} items]")
        else:
            # Truncate long values
            str_val = str(value)
            if len(str_val) > 50:
                str_val = str_val[:47] + "..."
            print("  " * (indent + 1) + f"└── {str_val}")

def create_minimal_config():
    """Create a minimal working config for testing"""
    minimal_config = """
[binning]
stack_all_z_at_once = true
add_foreground = true
crop_circles = true

[error_estimator]
write_simmaps = false
randomize = false

[error_estimator.bootstrap]
enabled = true
iterations = 150
initial_seed = 1

cosmology = "Planck18"

[output]
folder = "./output"
shortname = "test_run"

[catalog]
path = "test_data"
file = "test_catalog.csv"

[catalog.astrometry]
ra = "ra"
dec = "dec"

[catalog.classification]
split_type = "labels"

[catalog.classification.redshift]
id = "z_peak"
bins = [0.01, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

[catalog.classification.stellar_mass]
id = "lmass"
bins = [8.5, 9.5, 10.0, 10.5, 11.0, 12.0]

[maps.test_map]
wavelength = 100.0
color_correction = 1.0
path_map = "test_data/test_map.fits"
path_noise = ""

[maps.test_map.beam]
fwhm = 6.0
area = 1.0
"""
    
    with open('config/minimal_test.toml', 'w') as f:
        f.write(minimal_config)
    
    print("✅ Created minimal_test.toml for testing")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config/uvista.toml"
    
    print("Simstack4 Config Debugger")
    print("=" * 60)
    
    success = debug_toml_config(config_path)
    
    if not success:
        print("\n💡 Creating minimal config for testing...")
        create_minimal_config()
        print("\nTry: uv run python debug_config.py config/minimal_test.toml")
