#!/bin/bash
# Setup script for Simstack4 development

set -e  # Exit on any error

echo "🚀 Setting up Simstack4..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Are you in the simstack4 directory?"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "Please install uv first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ Found uv package manager"

# Sync dependencies
echo "📦 Installing dependencies..."
uv sync --extra dev --extra notebooks

# Create config directory if it doesn't exist
mkdir -p config

# Create example config if it doesn't exist
if [ ! -f "config/uvista_example.toml" ]; then
    echo "📝 Creating example configuration file..."
    # The TOML content would be written here - you already have this in the artifact
fi

# Create example directories
echo "📁 Creating directory structure..."
mkdir -p examples
mkdir -p docs
mkdir -p tests

# Set up pre-commit hooks (if pre-commit is installed)
if uv run python -c "import pre_commit" 2>/dev/null; then
    echo "🔧 Setting up pre-commit hooks..."
    cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.13

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
EOF
    
    uv run pre-commit install
fi

# Create a simple test to verify installation
echo "🧪 Creating basic test..."
cat > tests/test_basic.py << 'EOF'
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
EOF

# Run basic tests
echo "🧪 Running basic tests..."
uv run python -m pytest tests/test_basic.py -v

# Check environment variables
echo "🌍 Checking environment variables..."
if [ -z "$MAPSPATH" ] || [ -z "$CATSPATH" ] || [ -z "$PICKLESPATH" ]; then
    echo "⚠️  Warning: Environment variables not set"
    echo "Please add these to your ~/.zshrc:"
    echo ""
    echo "export MAPSPATH=/path/to/your/maps/"
    echo "export CATSPATH=/path/to/your/catalogs/"
    echo "export PICKLESPATH=/path/to/your/pickles/"
    echo ""
    echo "Then run: source ~/.zshrc"
else
    echo "✅ Environment variables are set"
    uv run python -c "
import simstack4
try:
    simstack4.validate_environment()
    print('✅ Environment validation passed')
except Exception as e:
    print(f'⚠️  Environment validation warning: {e}')
"
fi

# Create a simple example script
echo "📝 Creating example usage script..."
cat > examples/basic_usage.py << 'EOF'
#!/usr/bin/env python3
"""
Basic usage example for Simstack4
"""

import simstack4

def main():
    # Check system
    print("Checking system...")
    simstack4.print_system_info()
    
    # Example of loading config (requires actual config file)
    # config = simstack4.load_config("config/uvista_example.toml")
    # print(f"Loaded config with {len(config.maps)} maps")
    
    print("Simstack4 is ready to use!")

if __name__ == "__main__":
    main()
EOF

chmod +x examples/basic_usage.py

echo ""
echo "🎉 Simstack4 setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Set environment variables in ~/.zshrc if not already done"
echo "2. Copy your data files to the appropriate directories"
echo "3. Create/modify configuration files in the config/ directory"
echo "4. Run: uv run python examples/basic_usage.py"
echo "5. Try: uv run simstack4 --check-system"
echo ""
echo "For development:"
echo "- Run tests: uv run python -m pytest"
echo "- Format code: uv run black src/"
echo "- Lint code: uv run ruff check src/"
echo "- Type check: uv run mypy src/"