
"""Unit Test Config Parser"""

from pathlib import Path
from typing import Dict

import pytest
from tsff.common.config_manager import ConfigManager


@pytest.fixture
def mock_config() -> Dict:
    """
    Mock config data
    """
    config = {
        "dataset": {
            "db_name": "database",
            "table_name": "table"
        },
        "dataset_schema": {
            "required_columns": ["time", "target", "grain", "feature"],
            "target_colname": "target",
            "time_colname": "time",
            "grain_colnames": ["grain"]
        }
    }
    return config


@pytest.fixture
def mock_path_to_config() -> str:
    """
    Mock path to configuration dictionary
    """
    return f"{Path(__file__).parent}/mock_config.json"


def test_load(mock_config: Dict, mock_path_to_config: str):
    """
    Test the class ConfigManager output through assertion.

    Args:
        mock_config (Dict): Mock configuration dictionary
        mock_path_to_config (str): Path to configuration file
    """
    manager = ConfigManager()
    config = manager.load(mock_path_to_config)
    assert config == mock_config
