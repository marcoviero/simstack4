"""
Command line interface for Simstack4
"""

import argparse
import sys
from pathlib import Path
import time

from . import __version__
from .config import load_config
from .wrapper import SimstackWrapper
from .utils import setup_logging, validate_environment, print_system_info
from .exceptions.simstack_exceptions import SimstackError


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Simstack4: Simultaneous stacking for astrophysical sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with TOML config file
  simstack4 config/uvista_example.toml

  # Run with verbose logging
  simstack4 config/uvista_example.toml --verbose

  # Check system and environment
  simstack4 --check-system

  # Convert old INI config to TOML
  simstack4 --convert-config old_config.ini new_config.toml
        """
    )

    parser.add_argument(
        "config_file",
        nargs="?",
        help="Path to TOML configuration file"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Simstack4 {__version__}"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--log-file",
        help="Write logs to specified file"
    )

    parser.add_argument(
        "--check-system",
        action="store_true",
        help="Check system requirements and environment"
    )

    parser.add_argument(
        "--convert-config",
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        help="Convert old INI config to new TOML format"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running stacking"
    )

    return parser


def check_system() -> None:
    """Check system requirements and environment"""
    print("Checking Simstack4 system requirements...\n")

    # Print system info
    print_system_info()

    # Check environment variables
    try:
        env_status = validate_environment()
        print("\n=== Environment Validation ===")
        for var, status in env_status.items():
            if status["exists"]:
                print(f"✓ {var}: {status['value']}")
            else:
                print(f"✗ {var}: NOT SET")
    except Exception as e:
        print(f"✗ Environment validation failed: {e}")
        return

    print("\n✓ System check completed!")


def convert_ini_to_toml(ini_path: str, toml_path: str) -> None:
    """Convert old INI configuration to new TOML format"""
    # This would implement conversion from simstack3 INI format
    # For now, just a placeholder
    print(f"Converting {ini_path} to {toml_path}...")
    print("INI to TOML conversion not yet implemented.")
    print("Please manually create TOML config using the example format.")


def main() -> int:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.debug else ("INFO" if args.verbose else "WARNING")
    logger = setup_logging(level=log_level, log_file=args.log_file)

    try:
        # Handle special commands
        if args.check_system:
            check_system()
            return 0

        if args.convert_config:
            convert_ini_to_toml(args.convert_config[0], args.convert_config[1])
            return 0

        # Require config file for normal operation
        if not args.config_file:
            parser.error("Config file required (or use --check-system)")

        config_path = Path(args.config_file)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1

        logger.info(f"Starting Simstack4 with config: {config_path}")
        start_time = time.time()

        # Load and validate configuration
        logger.info("Loading configuration...")
        config = load_config(config_path)
        logger.info("Configuration loaded successfully")

        if args.dry_run:
            logger.info("Dry run mode - configuration is valid")
            return 0

        # Run stacking
        logger.info("Initializing SimstackWrapper...")
        simstack = SimstackWrapper(
            config=config,
            read_maps=True,
            read_catalog=True,
            stack_automatically=True
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Stacking completed successfully in {elapsed_time:.1f}s")

        # Basic results summary
        if hasattr(simstack, 'results_dict') and simstack.results_dict:
            n_bands = len(simstack.results_dict.get('band_results_dict', {}))
            logger.info(f"Results saved for {n_bands} bands")

        return 0

    except SimstackError as e:
        logger.error(f"Simstack4 error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())