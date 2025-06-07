#!/usr/bin/env python3
"""
Basic usage example for Simstack4
"""

import simstack4

def main():
    # Check system
    print("Checking system...")
    try:
        simstack4.print_system_info()
    except AttributeError:
        print("print_system_info not yet implemented")
        print("But simstack4 imported successfully!")
    
    # Example of loading config (requires actual config file)
    # config = simstack4.load_config("config/uvista_example.toml")
    # print(f"Loaded config with {len(config.maps)} maps")
    
    print("Simstack4 is ready to use!")

if __name__ == "__main__":
    main()
