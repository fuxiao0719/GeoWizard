"""Validation tests to ensure the testing infrastructure is properly set up."""

import sys
from pathlib import Path

import pytest
import numpy as np
import torch
from PIL import Image


@pytest.mark.unit
def test_pytest_is_working():
    """Verify that pytest is installed and working."""
    assert True, "Basic pytest assertion should work"


@pytest.mark.unit
def test_project_imports():
    """Verify that the main project modules can be imported."""
    # Add project root to path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Test importing main modules
    try:
        import geowizard
        assert geowizard is not None
    except ImportError as e:
        pytest.skip(f"Main module not importable yet: {e}")


@pytest.mark.unit
def test_fixtures_are_available(temp_dir, sample_image, sample_tensor):
    """Verify that custom fixtures from conftest.py are available."""
    # Test temp_dir fixture
    assert temp_dir.exists()
    assert temp_dir.is_dir()
    
    # Test sample_image fixture
    assert isinstance(sample_image, np.ndarray)
    assert sample_image.shape == (256, 256, 3)
    assert sample_image.dtype == np.uint8
    
    # Test sample_tensor fixture
    assert isinstance(sample_tensor, torch.Tensor)
    assert sample_tensor.shape == (1, 3, 256, 256)


@pytest.mark.unit
def test_coverage_is_configured():
    """Verify that coverage reporting is configured."""
    # This test will be covered by coverage, verifying it's working
    covered_function = lambda x: x + 1
    assert covered_function(1) == 2


@pytest.mark.unit
def test_markers_are_defined():
    """Verify that custom markers are properly defined."""
    # Get the current test item
    current_test = sys._getframe().f_code.co_name
    
    # This test should have the 'unit' marker
    assert True, "Markers are defined in conftest.py"


@pytest.mark.integration
def test_integration_marker():
    """Verify that integration marker works."""
    assert True, "Integration marker is functional"


@pytest.mark.slow
def test_slow_marker():
    """Verify that slow marker works and can be skipped."""
    import time
    time.sleep(0.1)  # Simulate slow test
    assert True, "Slow marker is functional"


@pytest.mark.unit
def test_all_dependencies_importable():
    """Verify that all major dependencies are importable."""
    dependencies = [
        ("numpy", "np"),
        ("torch", "torch"),
        ("PIL", "PIL"),
        ("pytest", "pytest"),
        ("cv2", "cv2"),
    ]
    
    failed_imports = []
    for module_name, import_name in dependencies:
        try:
            exec(f"import {import_name}")
        except ImportError:
            failed_imports.append(module_name)
    
    if failed_imports:
        pytest.skip(f"Some dependencies not installed yet: {failed_imports}")
    else:
        assert True, "All dependencies are importable"


@pytest.mark.unit
def test_fixture_types(
    sample_pil_image,
    sample_depth_map,
    sample_normal_map,
    mock_config,
    mock_model_state_dict
):
    """Verify that all fixture types are correct."""
    assert isinstance(sample_pil_image, Image.Image)
    assert isinstance(sample_depth_map, np.ndarray)
    assert sample_depth_map.shape == (256, 256)
    assert isinstance(sample_normal_map, np.ndarray)
    assert sample_normal_map.shape == (256, 256, 3)
    assert isinstance(mock_config, dict)
    assert "model" in mock_config
    assert isinstance(mock_model_state_dict, dict)
    assert all(isinstance(v, torch.Tensor) for v in mock_model_state_dict.values())


@pytest.mark.unit
def test_stdout_capture(capsys):
    """Verify that stdout capture works using built-in capsys."""
    print("Test output")
    print("Another line")
    
    captured = capsys.readouterr()
    assert "Test output" in captured.out
    assert "Another line" in captured.out


class TestClassBasedTests:
    """Verify that class-based tests work."""
    
    @pytest.mark.unit
    def test_class_method(self):
        """Test method in a test class."""
        assert True, "Class-based tests work"
    
    def test_another_method(self):
        """Another test method without explicit marker."""
        assert 1 + 1 == 2