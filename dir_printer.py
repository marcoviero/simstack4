#!/usr/bin/env python3
"""
Print directory structure for simstack4 project
Run this in your simstack4 root directory
"""

import os
from pathlib import Path


def print_tree(directory, prefix="", max_depth=None, current_depth=0):
    """Print directory tree structure"""
    if max_depth is not None and current_depth >= max_depth:
        return

    directory = Path(directory)
    entries = sorted([p for p in directory.iterdir() if not p.name.startswith('.')])

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{entry.name}")

        if entry.is_dir() and not entry.name.startswith('.'):
            extension = "    " if is_last else "│   "
            print_tree(entry, prefix + extension, max_depth, current_depth + 1)


def get_file_info(filepath):
    """Get basic file information"""
    try:
        stat = filepath.stat()
        size = stat.st_size
        if size < 1024:
            size_str = f"{size}B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f}KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f}MB"
        return size_str
    except:
        return "?"


def detailed_file_listing():
    """Show detailed file listing for Python files"""
    print("\n" + "=" * 60)
    print("DETAILED FILE LISTING")
    print("=" * 60)

    for root, dirs, files in os.walk('.'):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        if files:
            rel_root = os.path.relpath(root)
            if rel_root != '.':
                print(f"\n📁 {rel_root}/")

            for file in sorted(files):
                if not file.startswith('.'):
                    filepath = Path(root) / file
                    size = get_file_info(filepath)

                    # Indicate file types with icons
                    if file.endswith('.py'):
                        icon = "🐍"
                    elif file.endswith(('.ini', '.cfg', '.conf')):
                        icon = "⚙️"
                    elif file.endswith(('.md', '.txt', '.rst')):
                        icon = "📄"
                    elif file.endswith(('.yml', '.yaml')):
                        icon = "📋"
                    elif file.endswith('.toml'):
                        icon = "📦"
                    elif file.endswith(('.json')):
                        icon = "📊"
                    else:
                        icon = "📎"

                    print(f"  {icon} {file} ({size})")


def show_python_files_content_summary():
    """Show first few lines of each Python file"""
    print("\n" + "=" * 60)
    print("PYTHON FILES CONTENT SUMMARY")
    print("=" * 60)

    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in sorted(files):
            if file.endswith('.py') and not file.startswith('.'):
                filepath = Path(root) / file
                rel_path = filepath.relative_to('.')

                print(f"\n🐍 {rel_path}")
                print("-" * len(str(rel_path)))

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[:10]  # First 10 lines
                        for i, line in enumerate(lines, 1):
                            print(f"{i:2d}: {line.rstrip()}")
                        if len(f.readlines()) > 10:
                            print(f"    ... ({len(lines)} more lines)")
                except Exception as e:
                    print(f"    ❌ Error reading file: {e}")


if __name__ == "__main__":
    print("SIMSTACK4 PROJECT STRUCTURE")
    print("=" * 60)

    current_dir = Path.cwd()
    print(f"📂 {current_dir.name}/")
    print_tree(".", max_depth=4)

    detailed_file_listing()

    print("\n" + "=" * 60)
    print("UV PACKAGE MANAGER FILES")
    print("=" * 60)

    uv_files = ['pyproject.toml', 'uv.lock', '.python-version']
    for file in uv_files:
        if Path(file).exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")

    show_python_files_content_summary()

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. Share the output above")
    print("2. Paste content of incomplete files")
    print("3. We'll work on the population loop improvements")
    print("4. Set up uv package management properly")
    print("=" * 60)