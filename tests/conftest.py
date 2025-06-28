"""Shared pytest fixtures and configuration for all tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch
import numpy as np
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample image array for testing."""
    # Create a simple 256x256 RGB image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Create a sample PIL Image for testing."""
    # Create a simple 256x256 RGB image
    image_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(image_array)


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Create a sample torch tensor for testing."""
    # Create a 4D tensor (batch_size=1, channels=3, height=256, width=256)
    tensor = torch.randn(1, 3, 256, 256)
    return tensor


@pytest.fixture
def sample_depth_map() -> np.ndarray:
    """Create a sample depth map for testing."""
    # Create a 256x256 depth map with values between 0 and 10
    depth_map = np.random.uniform(0, 10, (256, 256)).astype(np.float32)
    return depth_map


@pytest.fixture
def sample_normal_map() -> np.ndarray:
    """Create a sample surface normal map for testing."""
    # Create a 256x256x3 normal map (normalized vectors)
    normal_map = np.random.randn(256, 256, 3).astype(np.float32)
    # Normalize vectors
    norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map = normal_map / (norm + 1e-8)
    return normal_map


@pytest.fixture
def mock_config() -> dict:
    """Create a mock configuration dictionary for testing."""
    return {
        "model": {
            "name": "test_model",
            "checkpoint": "path/to/checkpoint.pth",
            "device": "cpu",
        },
        "inference": {
            "batch_size": 1,
            "image_size": 256,
            "num_inference_steps": 50,
        },
        "paths": {
            "output_dir": "test_outputs",
            "cache_dir": ".cache",
        }
    }


@pytest.fixture
def mock_model_state_dict() -> dict:
    """Create a mock model state dictionary for testing."""
    return {
        "encoder.weight": torch.randn(64, 3, 7, 7),
        "encoder.bias": torch.randn(64),
        "decoder.weight": torch.randn(3, 64, 7, 7),
        "decoder.bias": torch.randn(3),
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    # Disable GPU for tests
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # Set deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    # Disable gradio analytics
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")




# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Skip slow tests by default
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle custom options."""
    if config.getoption("--runslow"):
        # Do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)