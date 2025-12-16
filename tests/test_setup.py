"""Test to verify project setup is working correctly."""

import sys
from pathlib import Path


def test_project_structure():
    """Test that all required directories exist."""
    project_root = Path(__file__).parent.parent
    
    # Check main directories
    assert (project_root / "src").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "docs").exists()
    
    # Check src subdirectories
    assert (project_root / "src" / "scraper").exists()
    assert (project_root / "src" / "processing").exists()
    assert (project_root / "src" / "recommendation").exists()
    assert (project_root / "src" / "api").exists()
    assert (project_root / "src" / "evaluation").exists()
    
    # Check test subdirectories
    assert (project_root / "tests" / "unit").exists()
    assert (project_root / "tests" / "integration").exists()


def test_configuration_files():
    """Test that all configuration files exist."""
    project_root = Path(__file__).parent.parent
    
    # Check configuration files
    assert (project_root / "requirements.txt").exists()
    assert (project_root / "requirements-dev.txt").exists()
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "Dockerfile").exists()
    assert (project_root / "docker-compose.yml").exists()
    assert (project_root / ".gitignore").exists()
    assert (project_root / ".env.example").exists()
    assert (project_root / "README.md").exists()


def test_python_path():
    """Test that src directory is importable."""
    project_root = Path(__file__).parent.parent
    src_path = str(project_root / "src")
    
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Try importing the main modules
    try:
        import scraper
        import processing
        import recommendation
        import api
        import evaluation
        assert True
    except ImportError as e:
        assert False, f"Failed to import modules: {e}"


if __name__ == "__main__":
    test_project_structure()
    test_configuration_files()
    test_python_path()
    print("All setup tests passed!")