#!/usr/bin/env python3
"""
COSMOS Stacking Pipeline - Stacking Only (FIXED VERSION)
Runs stacking analysis and saves results as self-contained JSON

Location: src/simstack4/scripts/cosmos_stacking_clean.py

Usage:
    uv run python src/simstack4/scripts/cosmos_stacking_clean.py --config config/cosmos25_luv_beta.toml
    python cosmos_stacking_clean.py --output-dir ./results
    python cosmos_stacking_clean.py --config path/to/config.toml --verbose
"""

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add the parent directory to Python path to import simstack4 modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simstack4.config import load_config  # noqa: E402
from simstack4.utils import setup_logging  # noqa: E402
from simstack4.wrapper import SimstackWrapper  # noqa: E402


def setup_paths(config_arg=None):
    """Find config file path with better error reporting"""
    if config_arg:
        config_path = Path(config_arg)
        print(f"🔍 Looking for config at: {config_path.absolute()}")

        if not config_path.exists():
            print(f"❌ Config file not found: {config_path.absolute()}")
            print(f"   Current working directory: {Path.cwd()}")
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print(f"✅ Found config file: {config_path.absolute()}")
        return config_path

    # Default config search with better reporting
    search_paths = [
        Path(__file__).parent.parent.parent.parent
        / "config"
        / "cosmos25_luv_beta.toml",
        Path("config/cosmos25_luv_beta.toml"),
        Path("../../../config/cosmos25_luv_beta.toml"),
        Path.cwd() / "config" / "cosmos25_luv_beta.toml",
        Path("cosmos25_luv_beta.toml"),  # Current directory
    ]

    print("🔍 Searching for default config file...")
    for search_path in search_paths:
        print(f"   Checking: {search_path.absolute()}")
        if search_path.exists():
            print(f"✅ Found config file: {search_path.absolute()}")
            return search_path

    print("❌ Could not find cosmos25_luv_beta.toml in any of these locations:")
    for path in search_paths:
        print(f"   {path.absolute()}")

    raise FileNotFoundError("Config file not found in any default location")


def validate_config(config_path: Path):
    """Validate config file before running"""
    print("🔍 Validating configuration...")

    try:
        config = load_config(config_path)

        # Check critical sections
        if not config.catalog:
            raise ValueError("No catalog configuration found")

        if not config.maps:
            raise ValueError("No maps configuration found")

        if not config.output:
            raise ValueError("No output configuration found")

        # Check file paths exist (with environment variable support)
        catalog_path = Path(config.catalog.path) / config.catalog.file
        if not catalog_path.exists() and "$" not in str(catalog_path):
            print(f"⚠️  Catalog file not found: {catalog_path}")
            print("   (This might be OK if using environment variables)")

        # Check map files
        missing_maps = []
        for map_name, map_config in config.maps.items():
            map_path = Path(map_config.path_map)
            if not map_path.exists() and "$" not in map_config.path_map:
                missing_maps.append(f"{map_name}: {map_path}")

        if missing_maps:
            print("⚠️  Some map files not found:")
            for missing in missing_maps:
                print(f"   {missing}")
            print("   (This might be OK if using environment variables)")

        print("✅ Configuration validation passed")
        print(f"   Catalog: {config.catalog.file}")
        print(f"   Maps: {len(config.maps)} configured")
        print(f"   Output: {config.output.shortname}")

        return config

    except (ValueError, AttributeError, TypeError) as e:
        print(f"❌ Configuration validation failed: {e}")
        raise


def run_stacking_pipeline(config_path: Path, output_dir: Path, verbose: bool = False):
    """Run stacking pipeline and save results as JSON with enhanced error handling"""
    print("🚀 COSMOS Stacking Pipeline")
    print("=" * 50)

    # Set up logging
    if verbose:
        import logging

        logging.basicConfig(level=logging.INFO)
        logger = setup_logging()
        logger.setLevel(logging.INFO)

    # Validate config first
    config = validate_config(config_path)

    print(f"📄 Config: {config_path}")
    print(f"📁 Output: {output_dir}")
    print(f"Configuration: {config.output.shortname}")
    print(f"Bootstrap enabled: {config.error_estimator.bootstrap.enabled}")
    print(f"Bootstrap iterations: {config.error_estimator.bootstrap.iterations}")
    print(f"Maps: {len(config.maps)}")
    print()

    start_time = time.time()
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    wrapper = None
    try:
        # Initialize wrapper step by step with progress reporting
        print("🔧 Initializing SimstackWrapper...")
        wrapper = SimstackWrapper(config=config_path)

        print("📚 Loading catalog...")
        wrapper._load_catalog()
        if wrapper.population_manager:
            print(f"✅ Catalog loaded: {len(wrapper.population_manager)} populations")

            # Print population summary
            for pop_id, pop in list(wrapper.population_manager.populations.items())[:5]:
                print(f"   {pop_id}: {getattr(pop, 'n_sources', 0)} sources")
            if len(wrapper.population_manager.populations) > 5:
                print(
                    f"   ... and {len(wrapper.population_manager.populations) - 5} more populations"
                )
        else:
            raise RuntimeError("Failed to create population manager")

        print("🗺️  Loading maps...")
        wrapper._load_maps()
        if wrapper.sky_maps:
            print(f"✅ Maps loaded: {len(wrapper.sky_maps)} maps")
            for map_name in wrapper.sky_maps.maps.keys():
                print(f"   {map_name}")
        else:
            raise RuntimeError("Failed to load maps")

        print("🧮 Running stacking computation...")
        wrapper._run_stacking()

        if not wrapper.stacking_results:
            raise RuntimeError("Stacking computation failed - no results generated")

        execution_time = time.time() - start_time

        # Save results with detailed reporting
        print("💾 Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stacking_file = (
            output_dir / f"{config.output.shortname}_stacking_{timestamp}.json"
        )

        # Ensure output directory exists
        stacking_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"   Saving to: {stacking_file}")
        wrapper.save_stacking_results(stacking_file)

        # Verify the file was saved correctly
        if stacking_file.exists():
            file_size = stacking_file.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ File saved successfully: {file_size:.1f} MB")

            # Quick validation
            try:
                import json

                with open(stacking_file) as f:
                    data = json.load(f)

                metadata = data.get("metadata", {})
                print(f"   Version: {metadata.get('version', 'unknown')}")
                print(f"   Self-contained: {metadata.get('self_contained', False)}")
                print(f"   Populations: {metadata.get('n_populations', 0)}")
                print(f"   Maps: {metadata.get('n_maps', 0)}")

            except (ValueError, TypeError, KeyError) as e:
                print(f"⚠️  File validation warning: {e}")
        else:
            raise RuntimeError(f"File was not created: {stacking_file}")

        print()
        print("✅ Stacking computation completed!")
        print(f"⏱️  Execution time: {execution_time / 60:.1f} minutes")
        print(f"💾 Results saved: {stacking_file}")
        print()
        print("📋 File contains:")
        print("  • Stacked fluxes and errors for all populations")
        print("  • Population data (coordinates, redshifts, masses)")
        print("  • Complete configuration parameters")
        print("  • Catalog metadata for reconstruction")
        print("  • Analysis-ready self-contained data")
        print()
        print("🔬 To analyze these results (no config file needed!):")
        print("   # In Python:")
        print("   from simstack4.wrapper import run_analysis_only")
        print(f"   results = run_analysis_only('{stacking_file}')")
        print()
        print("   # Or load manually:")
        print("   from simstack4.wrapper import SimstackWrapper")
        print("   wrapper = SimstackWrapper()")
        print(f"   wrapper.load_stacking_results('{stacking_file}')")
        print("   analysis_results = wrapper.run_analysis_only(use_mcmc=True)")

        return {
            "success": True,
            "execution_time": execution_time,
            "stacking_file": stacking_file,
            "file_size_mb": file_size,
            "n_populations": len(wrapper.population_manager)
            if wrapper.population_manager
            else 0,
        }

    except (RuntimeError, ValueError, ImportError, OSError) as e:
        execution_time = time.time() - start_time
        print(f"\n❌ Stacking failed after {execution_time / 60:.1f} minutes")
        print(f"Error: {str(e)}")

        # Print detailed error information
        if verbose:
            print("\n🔍 Detailed error information:")
            print(traceback.format_exc())

        # Save comprehensive error log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = output_dir / f"stacking_error_{timestamp}.txt"

        try:
            error_file.parent.mkdir(parents=True, exist_ok=True)
            with open(error_file, "w") as f:
                f.write("COSMOS Stacking Error Report\n")
                f.write("=" * 40 + "\n")
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write(f"Config: {config_path}\n")
                f.write(f"Output dir: {output_dir}\n")
                f.write(f"Execution time: {execution_time:.1f} seconds\n")
                f.write(f"Error: {str(e)}\n")
                f.write("\nConfiguration loaded:\n")
                try:
                    f.write(f"  Catalog: {config.catalog.file}\n")
                    f.write(f"  Maps: {list(config.maps.keys())}\n")
                    f.write(f"  Output: {config.output.shortname}\n")
                except (AttributeError, TypeError):
                    f.write("  (Config details not available)\n")

                f.write("\nWrapper state:\n")
                if wrapper:
                    f.write(f"  Catalog loaded: {wrapper.sky_catalogs is not None}\n")
                    f.write(f"  Maps loaded: {wrapper.sky_maps is not None}\n")
                    f.write(
                        f"  Population manager: {wrapper.population_manager is not None}\n"
                    )
                    f.write(
                        f"  Stacking results: {wrapper.stacking_results is not None}\n"
                    )
                else:
                    f.write("  Wrapper not initialized\n")

                f.write("\nFull traceback:\n")
                f.write(traceback.format_exc())

            print(f"📝 Error log saved: {error_file}")
        except OSError as log_error:
            print(f"⚠️  Could not save error log: {log_error}")

        return {
            "success": False,
            "error": str(e),
            "execution_time": execution_time,
            "error_file": error_file if error_file.exists() else None,
        }


def main():
    """Main function with enhanced argument parsing and error handling"""
    parser = argparse.ArgumentParser(
        description="COSMOS Stacking Pipeline - saves results as self-contained JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with config file
  uv run python cosmos_stacking_clean.py --config config/cosmos25_luv_beta.toml

  # With custom output directory
  python cosmos_stacking_clean.py --config config/cosmos25_luv_beta.toml --output-dir ./results

  # With verbose output for debugging
  python cosmos_stacking_clean.py --config config/cosmos25_luv_beta.toml --verbose
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file (default: searches for cosmos25_luv_beta.toml)",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory (default: from config file)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output for debugging",
    )

    args = parser.parse_args()

    print("🌌 COSMOS Stacking Pipeline")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📁 Working directory: {Path.cwd()}")
    print()

    try:
        # Setup paths with better error reporting
        config_path = setup_paths(args.config)

        # Load config to get output directory
        config = load_config(config_path)
        output_dir = args.output_dir or Path(config.output.folder)

        # Ensure output directory exists and is writable
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            # Test write permission
            test_file = (
                output_dir
                / f"test_write_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tmp"
            )
            test_file.write_text("test")
            test_file.unlink()
            print(f"✅ Output directory ready: {output_dir.absolute()}")
        except (OSError, PermissionError) as e:
            print(f"❌ Cannot write to output directory: {output_dir}")
            print(f"   Error: {e}")
            sys.exit(1)

        print()

        # Run stacking
        results = run_stacking_pipeline(config_path, output_dir, args.verbose)

        # Print final status
        print("\n" + "=" * 50)
        if results["success"]:
            print(
                f"✅ SUCCESS: Stacking completed in {results['execution_time'] / 60:.1f} minutes"
            )
            print(f"📊 {results['n_populations']} populations processed")
            print(
                f"💾 Results file: {results['stacking_file']} ({results.get('file_size_mb', 0):.1f} MB)"
            )
            print()
            print("🎉 Ready for analysis! The JSON file is completely self-contained.")
        else:
            print(f"❌ FAILED: {results['error']}")
            if results.get("error_file"):
                print(f"📝 Error log: {results['error_file']}")
            print(f"⏱️  Failed after {results['execution_time'] / 60:.1f} minutes")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(1)
    except (FileNotFoundError, ValueError, TypeError, OSError) as e:
        print(f"\n❌ SETUP FAILED: {str(e)}")
        if args.verbose:
            print("\n🔍 Detailed error:")
            print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
