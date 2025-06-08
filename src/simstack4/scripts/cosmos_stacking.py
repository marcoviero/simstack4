#!/usr/bin/env python3
"""
COSMOS Stacking Pipeline with Bootstrap Error Estimation
Runs stacking analysis on COSMOS catalog with automatic result saving

Location: src/simstack4/scripts/cosmos_stacking.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add the parent directory to Python path to import simstack4 modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Now import simstack4 modules using relative imports
from simstack4.config import load_config
from simstack4.wrapper import SimstackWrapper


def run_cosmos_stacking_pipeline():
    """Run complete COSMOS stacking pipeline with result saving"""

    # Load configuration
    print("🚀 COSMOS Stacking Pipeline with Bootstrap")
    print("=" * 50)

    # Config path relative to project root (where you run the script from)
    config_path = (
        Path(__file__).parent.parent.parent.parent / "config" / "cosmos25.toml"
    )

    if not config_path.exists():
        # Alternative paths to try
        alt_paths = [
            "config/cosmos25.toml",  # If run from project root
            "../../../config/cosmos25.toml",  # Relative to script location
            Path.cwd() / "config" / "cosmos25.toml",  # From current working directory
        ]

        for alt_path in alt_paths:
            if Path(alt_path).exists():
                config_path = Path(alt_path)
                break
        else:
            print("❌ Could not find cosmos25.toml in any of these locations:")
            print(
                f"   • {Path(__file__).parent.parent.parent.parent / 'config' / 'cosmos25.toml'}"
            )
            for path in alt_paths:
                print(f"   • {path}")
            print(f"\nCurrent working directory: {Path.cwd()}")
            print(f"Script location: {Path(__file__).parent}")
            return {"success": False, "error": "Config file not found"}

    print(f"📄 Using config: {config_path}")
    config = load_config(config_path)

    # Remove crop_circles setting (use default from TOML)
    # config.binning.crop_circles = False  # REMOVED as requested

    # Enable bootstrap with 10 iterations
    config.error_estimator.bootstrap.enabled = True
    config.error_estimator.bootstrap.iterations = 10

    print(f"Configuration loaded: {config.output.shortname}")
    print(f"Crop circles: {config.binning.crop_circles}")
    print(f"Bootstrap enabled: {config.error_estimator.bootstrap.enabled}")
    print(f"Bootstrap iterations: {config.error_estimator.bootstrap.iterations}")
    print(
        f"Expected runs: {config.error_estimator.bootstrap.iterations} × {len(config.maps)} maps = {config.error_estimator.bootstrap.iterations * len(config.maps)} bootstrap runs"
    )
    print("Estimated time: ~15-20 minutes")
    print()

    # Create output directory
    output_dir = Path(config.output.folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Run stacking pipeline
    start_time = time.time()
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        wrapper = SimstackWrapper(
            config, read_maps=True, read_catalog=True, stack_automatically=True
        )

        execution_time = time.time() - start_time

        if wrapper.processed_results:
            print("✅ Bootstrap error estimation completed!")
            print(f"⏱️  Total execution time: {execution_time / 60:.1f} minutes")
            print()

            # Save results in multiple formats
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{config.output.shortname}_bootstrap_{timestamp}"

            # 1. Save detailed results (pickle format)
            pickle_path = output_dir / f"{base_filename}_detailed.pkl"
            wrapper.processed_results.save_results(pickle_path, format="pickle")
            print(f"💾 Detailed results saved: {pickle_path}")

            # 2. Save summary table (CSV format)
            summary_df = wrapper.processed_results.get_population_summary()
            csv_path = output_dir / f"{base_filename}_summary.csv"
            summary_df.to_csv(csv_path, index=False)
            print(f"📊 Summary table saved: {csv_path}")

            # 3. Save bootstrap flux results (CSV format)
            if hasattr(wrapper.processed_results, "get_bootstrap_results"):
                bootstrap_df = wrapper.processed_results.get_bootstrap_results()
                bootstrap_path = output_dir / f"{base_filename}_bootstrap_fluxes.csv"
                bootstrap_df.to_csv(bootstrap_path, index=False)
                print(f"🔄 Bootstrap fluxes saved: {bootstrap_path}")

            # 4. Save configuration used
            config_path_out = output_dir / f"{base_filename}_config.toml"
            try:
                import tomli_w

                with open(config_path_out, "wb") as f:
                    # Convert config back to dict for saving
                    config_dict = {
                        "binning": {
                            "stack_all_z_at_once": config.binning.stack_all_z_at_once,
                            "add_foreground": config.binning.add_foreground,
                            "crop_circles": config.binning.crop_circles,
                        },
                        "error_estimator": {
                            "bootstrap": {
                                "enabled": config.error_estimator.bootstrap.enabled,
                                "iterations": config.error_estimator.bootstrap.iterations,
                                "initial_seed": config.error_estimator.bootstrap.initial_seed,
                            }
                        },
                        "output": {
                            "folder": config.output.folder,
                            "shortname": config.output.shortname,
                        },
                    }
                    tomli_w.dump(config_dict, f)
                print(f"⚙️  Configuration saved: {config_path_out}")
            except ImportError:
                print("⚠️  tomli_w not available - skipping config save")

            print()
            print("=" * 50)
            print("📈 RESULTS SUMMARY")
            print("=" * 50)

            # Analysis of results
            measured = summary_df[summary_df["total_ir_luminosity_lsun"] > 0]
            print("📊 Bootstrap Results Summary:")
            print(f"  • Total populations: {len(summary_df):,}")
            print(f"  • Populations with L_IR > 0: {len(measured):,}")
            print(f"  • Detection rate: {len(measured) / len(summary_df) * 100:.1f}%")

            if len(measured) > 0:
                print()
                print("🌟 Top 5 IR Luminosity Detections:")
                print("-" * 80)
                top = measured.nlargest(5, "total_ir_luminosity_lsun")

                for i, (_, row) in enumerate(top.iterrows(), 1):
                    pop_id = row["population_id"][:35]  # Truncate long IDs
                    l_ir = row["total_ir_luminosity_lsun"]
                    l_ir_err = row.get("total_ir_luminosity_lsun_err", 0)
                    sfr = row["sfr_msun_yr"]
                    sfr_err = row.get("sfr_msun_yr_err", 0)
                    n_src = row["n_sources"]
                    z_med = row.get("median_redshift", 0)
                    mass_med = row.get("median_stellar_mass", 0)

                    print(f"{i:2d}. {pop_id}")
                    print(f"    L_IR: {l_ir:.2e} ± {l_ir_err:.2e} L☉")
                    print(f"    SFR:  {sfr:.2e} ± {sfr_err:.2e} M☉/yr")
                    print(
                        f"    Sources: {n_src:,} | z_med: {z_med:.2f} | M*_med: {mass_med:.1f}"
                    )
                    print()

                # Population type breakdown
                if "population_id" in summary_df.columns:
                    sf_pops = summary_df[
                        summary_df["population_id"].str.contains("split_0")
                    ]
                    q_pops = summary_df[
                        summary_df["population_id"].str.contains("split_1")
                    ]

                    sf_detected = sf_pops[sf_pops["total_ir_luminosity_lsun"] > 0]
                    q_detected = q_pops[q_pops["total_ir_luminosity_lsun"] > 0]

                    print("🔬 Population Type Analysis:")
                    print(
                        f"  Star-forming: {len(sf_detected)}/{len(sf_pops)} detected ({len(sf_detected) / len(sf_pops) * 100:.1f}%)"
                    )
                    print(
                        f"  Quiescent:    {len(q_detected)}/{len(q_pops)} detected ({len(q_detected) / len(q_pops) * 100:.1f}%)"
                    )

            print()
            print("=" * 50)
            print("📁 SAVED FILES")
            print("=" * 50)
            print(f"All results saved to: {output_dir}")
            print(f"• {pickle_path.name} - Complete results (for further analysis)")
            print(f"• {csv_path.name} - Population summary table")
            if "config_path_out" in locals():
                print(f"• {config_path_out.name} - Configuration used")
            if "bootstrap_path" in locals():
                print(f"• {bootstrap_path.name} - Bootstrap flux measurements")

            print()
            print("🎉 Pipeline completed successfully!")

            return {
                "success": True,
                "execution_time": execution_time,
                "n_populations": len(summary_df),
                "n_detections": len(measured),
                "output_files": {
                    "detailed": pickle_path,
                    "summary": csv_path,
                },
            }

        else:
            print("❌ Bootstrap failed - no results generated")
            return {"success": False, "error": "No processed results"}

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ Pipeline failed after {execution_time / 60:.1f} minutes")
        print(f"Error: {str(e)}")

        # Save error log
        error_path = (
            output_dir / f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(error_path, "w") as f:
            f.write("COSMOS Stacking Pipeline Error\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Execution time: {execution_time:.1f} seconds\n")
            f.write(f"Error: {str(e)}\n")

            import traceback

            f.write("\nFull traceback:\n")
            f.write(traceback.format_exc())

        print(f"💾 Error log saved: {error_path}")

        return {"success": False, "error": str(e), "execution_time": execution_time}


if __name__ == "__main__":
    results = run_cosmos_stacking_pipeline()

    if results["success"]:
        print(
            f"\n✅ SUCCESS: Pipeline completed in {results['execution_time'] / 60:.1f} minutes"
        )
        print(
            f"📊 {results['n_detections']}/{results['n_populations']} populations detected"
        )
    else:
        print(f"\n❌ FAILED: {results['error']}")
        sys.exit(1)
