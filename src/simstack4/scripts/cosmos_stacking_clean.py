#!/usr/bin/env python3
"""
COSMOS Stacking Pipeline - Stacking Only
Runs stacking analysis and saves results as self-contained JSON

Location: src/simstack4/scripts/cosmos_stacking.py

Usage:
    python cosmos_stacking.py                    # Run with default config
    python cosmos_stacking.py --output-dir ./    # Override output directory
    python cosmos_stacking.py --config path/to/config.toml  # Use different config

Before stacking, loads the catalog + maps and estimates peak memory and
wall-clock time on this machine; if the estimate exceeds
--max-memory-fraction of available RAM or --max-time-minutes, it pauses for
confirmation (or aborts outright in a non-interactive session without
--yes). Use --skip-preflight to bypass the estimate entirely, or --yes/-y
to proceed without the confirmation prompt.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the parent directory to Python path to import simstack4 modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simstack4.config import load_config  # noqa: E402
from simstack4.preflight import (  # noqa: E402
    confirm_or_abort,
    estimate_compute_requirements,
)
from simstack4.wrapper import SimstackWrapper  # noqa: E402


def setup_paths(config_arg=None):
    """Find config file path"""
    if config_arg:
        config_path = Path(config_arg)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return config_path

    # Default config search
    config_path = (
        Path(__file__).parent.parent.parent.parent / "config" / "cosmos25.toml"
    )

    if not config_path.exists():
        alt_paths = [
            "config/cosmos25.toml",
            "../../../config/cosmos25.toml",
            Path.cwd() / "config" / "cosmos25.toml",
        ]

        for alt_path in alt_paths:
            if Path(alt_path).exists():
                config_path = Path(alt_path)
                break
        else:
            print("❌ Could not find cosmos25.toml")
            raise FileNotFoundError("Config file not found")

    return config_path


def run_stacking_pipeline(
    config_path: Path,
    output_dir: Path,
    *,
    skip_preflight: bool = False,
    assume_yes: bool = False,
    max_memory_fraction: float = 0.8,
    max_time_minutes: float = 15.0,
):
    """Run stacking pipeline and save results as JSON"""
    print("🚀 COSMOS Stacking Pipeline")
    print("=" * 50)

    config = load_config(config_path)
    print(f"📄 Config: {config_path}")
    print(f"Configuration: {config.output.shortname}")
    print(f"Bootstrap enabled: {config.error_estimator.bootstrap.enabled}")
    print(f"Bootstrap iterations: {config.error_estimator.bootstrap.iterations}")
    print(f"Maps: {len(config.maps)}")
    print()

    start_time = time.time()
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Load catalog + maps first (real, unavoidable work) so we can
        # estimate compute requirements before the expensive stacking step.
        wrapper = SimstackWrapper(
            config=config_path,
            read_maps=True,
            read_catalog=True,
            stack_automatically=False,
            analyze_automatically=False,
        )

        if not skip_preflight:
            print("\n📐 Estimating compute requirements...")
            estimate = estimate_compute_requirements(
                wrapper.config, wrapper.population_manager, wrapper.sky_maps
            )
            print()
            if not confirm_or_abort(
                estimate,
                memory_fraction=max_memory_fraction,
                time_seconds=max_time_minutes * 60.0,
                assume_yes=assume_yes,
            ):
                print("\n🛑 Aborted before stacking (compute estimate exceeded budget).")
                return {"success": False, "error": "aborted_by_user_preflight"}
            print()

        # Run stacking (catalog/maps already loaded above, so this reuses them)
        wrapper._run_stacking()

        execution_time = time.time() - start_time

        if wrapper.stacking_results:
            # Save as self-contained JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stacking_file = (
                output_dir / f"{config.output.shortname}_stacking_{timestamp}.json"
            )
            wrapper.save_stacking_results(stacking_file)

            print("✅ Stacking computation completed!")
            print(f"⏱️  Execution time: {execution_time / 60:.1f} minutes")
            print(f"💾 Results saved: {stacking_file}")
            print()
            print("📋 File contains:")
            print("  • Stacked fluxes and errors")
            print("  • Population data (coordinates, redshifts, masses)")
            print("  • Configuration parameters")
            print("  • Metadata")
            print()
            print("🔬 To analyze these results:")
            print("   # In Python:")
            print("   from simstack4.wrapper import run_analysis_only")
            print(f"   results = run_analysis_only('{stacking_file}')")
            print()
            print("   # Or load manually:")
            print("   wrapper = SimstackWrapper()")
            print(f"   wrapper.load_stacking_results('{stacking_file}')")
            print("   analysis_results = wrapper.run_analysis_only()")

            return {
                "success": True,
                "execution_time": execution_time,
                "stacking_file": stacking_file,
                "n_populations": (
                    len(wrapper.population_manager) if wrapper.population_manager else 0
                ),
            }

        else:
            print("❌ Stacking failed - no results generated")
            return {"success": False, "error": "No stacking results"}

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ Stacking failed after {execution_time / 60:.1f} minutes")
        print(f"Error: {str(e)}")

        # Save error log
        error_file = (
            output_dir
            / f"stacking_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(error_file, "w") as f:
            f.write("Stacking Error Report\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Config: {config_path}\n")
            f.write(f"Error: {str(e)}\n")
            f.write("\nFull traceback:\n")
            import traceback

            f.write(traceback.format_exc())

        print(f"📝 Error log saved: {error_file}")
        return {"success": False, "error": str(e), "execution_time": execution_time}


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="COSMOS Stacking Pipeline - saves results as self-contained JSON"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file (default: searches for cosmos25.toml)",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory (default: from config file)"
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the compute-requirement estimate and run stacking immediately",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Don't pause for confirmation even if the estimate exceeds budget",
    )
    parser.add_argument(
        "--max-memory-fraction",
        type=float,
        default=0.8,
        metavar="FRACTION",
        help="Pause if estimated peak memory exceeds this fraction of "
             "available RAM (default: 0.8)",
    )
    parser.add_argument(
        "--max-time-minutes",
        type=float,
        default=15.0,
        metavar="MINUTES",
        help="Pause if estimated run time exceeds this many minutes (default: 15)",
    )

    args = parser.parse_args()

    try:
        # Setup paths
        config_path = setup_paths(args.config)
        config = load_config(config_path)

        output_dir = args.output_dir or Path(config.output.folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Output directory: {output_dir}")
        print()

        # Run stacking
        results = run_stacking_pipeline(
            config_path,
            output_dir,
            skip_preflight=args.skip_preflight,
            assume_yes=args.yes,
            max_memory_fraction=args.max_memory_fraction,
            max_time_minutes=args.max_time_minutes,
        )

        # Print final status
        if results["success"]:
            print(
                f"\n✅ SUCCESS: Stacking completed in {results['execution_time'] / 60:.1f} minutes"
            )
            print(f"📊 {results['n_populations']} populations processed")
            print(f"📁 Results file: {results['stacking_file']}")
        elif results["error"] == "aborted_by_user_preflight":
            sys.exit(0)
        else:
            print(f"\n❌ FAILED: {results['error']}")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ SETUP FAILED: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
