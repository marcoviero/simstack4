#!/usr/bin/env python3
"""
COSMOS Analysis Pipeline - Analysis Only
Loads stacking results from JSON and runs SED fitting and bootstrap analysis

Location: src/simstack4/scripts/cosmos_analysis.py

Usage:
    python cosmos_analysis.py --results-json cosmos25_stacking_20250620_171049.json
    python cosmos_analysis.py --results-json stacking.json --output analysis_results.json
    python cosmos_analysis.py --results-json stacking.json --output analysis.pkl --format pkl
    python cosmos_analysis.py --results-json stacking.json --bootstrap-iterations 2000
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the parent directory to Python path to import simstack4 modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simstack4.wrapper import SimstackWrapper  # noqa: E402


def find_results_file(results_arg):
    """Find the results JSON file"""
    results_path = Path(results_arg)

    # If it's an absolute path or exists as-is, use it
    if results_path.is_absolute() or results_path.exists():
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        return results_path

    # Try common locations
    search_paths = [
        Path.cwd() / results_arg,  # Current directory
        Path("/Users/mviero/data/Astronomy/pickles/simstack/stacked_flux_densities")
        / results_arg,
        Path(__file__).parent / results_arg,  # Script directory
    ]

    for search_path in search_paths:
        if search_path.exists():
            return search_path

    # If not found, show where we looked
    print(f"❌ Results file '{results_arg}' not found in:")
    for path in search_paths:
        print(f"   • {path}")
    raise FileNotFoundError(f"Results file not found: {results_arg}")


def run_analysis_pipeline(
    results_file: Path,
    output_file: Path = None,
    output_format: str = "json",
    bootstrap_iterations: int = None,
):
    """Run analysis pipeline from saved stacking results"""
    print("🔬 COSMOS Analysis Pipeline")
    print("=" * 50)
    print(f"📁 Loading results: {results_file}")

    if bootstrap_iterations:
        print(f"🔄 Bootstrap iterations override: {bootstrap_iterations}")

    start_time = time.time()
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Load stacking results
        print("📂 Loading stacking results...")
        wrapper = SimstackWrapper()  # No config required!
        wrapper.load_stacking_results(results_file)

        # Override bootstrap iterations if requested
        if bootstrap_iterations and wrapper.config:
            print(
                f"🔄 Overriding bootstrap iterations: {wrapper.config.error_estimator.bootstrap.iterations} → {bootstrap_iterations}"
            )
            wrapper.config.error_estimator.bootstrap.iterations = bootstrap_iterations

        # Show what we loaded
        print("✅ Stacking results loaded:")
        if hasattr(wrapper.stacking_results, "flux_densities"):
            maps = list(wrapper.stacking_results.flux_densities.keys())
            print(f"  • Flux data for {len(maps)} maps: {maps}")

        if wrapper.population_manager:
            print(f"  • {len(wrapper.population_manager)} populations")

        print()

        # Run analysis
        print("🧮 Running analysis (SED fitting, bootstrap, etc.)...")
        analysis_results = wrapper.run_analysis_only()

        execution_time = time.time() - start_time

        if analysis_results:
            print("✅ Analysis completed!")
            print(f"⏱️  Analysis time: {execution_time / 60:.1f} minutes")
            print()

            # Determine output file
            if not output_file:
                # Auto-generate output filename based on input
                input_stem = results_file.stem
                if input_stem.endswith("_stacking"):
                    base_name = input_stem.replace("_stacking", "_analysis")
                else:
                    base_name = f"{input_stem}_analysis"

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if output_format == "pkl":
                    output_file = results_file.parent / f"{base_name}_{timestamp}.pkl"
                else:
                    output_file = results_file.parent / f"{base_name}_{timestamp}.csv"

            # Save results
            print(f"💾 Saving results to: {output_file}")
            if output_format == "pkl":
                try:
                    wrapper.save_analysis_results(output_file)
                    print(f"✅ Results saved as pickle: {output_file}")
                except Exception as e:
                    print(f"⚠️  Pickle save failed: {e}")
                    # Fall back to saving summary as CSV
                    csv_file = output_file.with_suffix(".csv")
                    summary_df = analysis_results.get_population_summary()
                    summary_df.to_csv(csv_file, index=False)
                    print(f"📊 Saved summary as CSV instead: {csv_file}")
                    output_file = csv_file
            else:
                # Save summary as CSV since JSON is complex for analysis results
                csv_file = output_file.with_suffix(".csv")
                summary_df = analysis_results.get_population_summary()
                summary_df.to_csv(csv_file, index=False)
                print(f"📊 Saved summary as CSV: {csv_file}")
                print(
                    "   Note: Full analysis results are complex - CSV contains population summary"
                )
                output_file = csv_file

            # Show summary
            show_analysis_summary(analysis_results)

            return {
                "success": True,
                "execution_time": execution_time,
                "output_file": output_file,
                "n_populations": len(wrapper.population_manager)
                if wrapper.population_manager
                else 0,
            }

        else:
            print("❌ Analysis failed - no results generated")
            return {"success": False, "error": "No analysis results"}

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ Analysis failed after {execution_time / 60:.1f} minutes")
        print(f"Error: {str(e)}")

        # Save error log
        error_file = (
            results_file.parent
            / f"analysis_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(error_file, "w") as f:
            f.write("Analysis Error Report\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Results file: {results_file}\n")
            f.write(f"Error: {str(e)}\n")
            f.write("\nFull traceback:\n")
            import traceback

            f.write(traceback.format_exc())

        print(f"📝 Error log saved: {error_file}")
        return {"success": False, "error": str(e), "execution_time": execution_time}


def show_analysis_summary(analysis_results):
    """Display analysis results summary"""
    try:
        summary_df = analysis_results.get_population_summary()
        measured = summary_df[summary_df["total_ir_luminosity_lsun"] > 0]

        print()
        print("=" * 50)
        print("📈 ANALYSIS RESULTS SUMMARY")
        print("=" * 50)
        print(f"📊 Total populations: {len(summary_df):,}")
        print(f"🌟 Populations with L_IR > 0: {len(measured):,}")
        print(f"📈 Detection rate: {len(measured) / len(summary_df) * 100:.1f}%")

        if len(measured) > 0:
            l_ir_values = measured["total_ir_luminosity_lsun"]
            sfr_values = measured["sfr_msun_yr"]

            print(f"🔆 L_IR range: {l_ir_values.min():.2e} - {l_ir_values.max():.2e} L☉")
            print(f"🔆 L_IR median: {l_ir_values.median():.2e} L☉")
            print(f"⭐ SFR range: {sfr_values.min():.2e} - {sfr_values.max():.2e} M☉/yr")
            print(f"⭐ SFR median: {sfr_values.median():.2e} M☉/yr")

            print()
            print("🏆 Top 3 Detections:")
            top = measured.nlargest(3, "total_ir_luminosity_lsun")
            for i, (_, row) in enumerate(top.iterrows(), 1):
                pop_id = row["population_id"][:50]
                l_ir = row["total_ir_luminosity_lsun"]
                l_ir_err = row.get("total_ir_luminosity_lsun_err", 0)
                sfr = row["sfr_msun_yr"]
                n_src = row["n_sources"]

                snr = l_ir / l_ir_err if l_ir_err > 0 else float("inf")
                print(f"  {i}. {pop_id}")
                print(f"     L_IR: {l_ir:.2e} ± {l_ir_err:.2e} L☉ (SNR: {snr:.1f})")
                print(f"     SFR: {sfr:.2e} M☉/yr | {n_src:,} sources")

    except Exception as e:
        print(f"⚠️  Could not display summary: {e}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="COSMOS Analysis Pipeline - loads stacking results and runs SED fitting analysis"
    )
    parser.add_argument(
        "--results-json", required=True, help="Path to stacking results JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: auto-generated based on input)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "pkl"],
        default="csv",
        help="Output format: csv for summary table, pkl for full results (default: csv)",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        help="Override bootstrap iterations from config",
    )

    args = parser.parse_args()

    try:
        # Find results file
        results_file = find_results_file(args.results_json)
        print(f"📁 Found results file: {results_file}")
        print()

        # Run analysis
        results = run_analysis_pipeline(
            results_file=results_file,
            output_file=args.output,
            output_format=args.format,
            bootstrap_iterations=args.bootstrap_iterations,
        )

        # Print final status
        if results["success"]:
            print(
                f"\n✅ SUCCESS: Analysis completed in {results['execution_time'] / 60:.1f} minutes"
            )
            print(f"📊 {results['n_populations']} populations analyzed")
            print(f"📁 Results saved: {results['output_file']}")
        else:
            print(f"\n❌ FAILED: {results['error']}")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ SETUP FAILED: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
