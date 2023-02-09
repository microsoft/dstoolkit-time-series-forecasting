"""
This module contains the ConfigManager class, which parses the JSON config file used by the
configuration file creation and model training & evaluation notebooks.
"""
import datetime as dt
import json
from typing import Dict, List, Optional
from copy import deepcopy


class ConfigManager:
    """Manage config dictionary and file."""
    
    def __init__(self, path_to_config: Optional[str] = None):
        """
        Constructor for ConfigManager class
        
        Args:
            path_to_config (Optional[str]): Path to a json configuration file.
        """
        if path_to_config:
            self.config = self.load(path_to_config)

    @staticmethod
    def load(path_to_config: str):
        """
        Loads a config into a dictionary from a json file.

        Args:
            path_to_config (str): Path to the json configuration file.
 
        Returns:
            config (Dict): Loaded configuration dictionary
        """
        with open(path_to_config, "r") as f:
            config = json.load(f)
        return config

    def modify(self,
               path_to_section: List[str],
               new_section_values: Dict,
               in_place: bool = False):
        """
        Method to modify parts of a configuration file and return the modified full config

        Args:
            path_to_section (List): List of strings specifying the path to
                                       navigate through nested dictionaries
            new_section_values (Dict): Update the subsection with these keys and values
            in_place (bool): If config attribute should be modified in place (default: True)

        """
        # Do a deepcopy of the config so that you don't modify original config
        if not in_place:
            config = deepcopy(self.config)
        else:
            config = self.config

        # Initialize section of the config
        section_of_config = config
        # Navigate through the configuration dictionary by following the path
        for key in path_to_section:
            section_of_config = section_of_config[key]
        # Update the subsection of the config
        for key, value in new_section_values.items():
            if key in section_of_config and section_of_config[key] != value:
                section_of_config[key] = value
        return config

    @staticmethod
    def save(config: Dict, path_to_save: str):
        """
        Writes the config dictionary to the specified path.

        Args:
            config (Dict): Configuration dictionary to save
            path_to_save (str): Path to the destination file.
        """
        with open(path_to_save, "w") as f:
            f.write(json.dumps(config, indent=4))

    def set(self, config: Dict):
        """
        Sets a config dictionary to the internal config attribute

        Args:
            config (Dict): Configuration dictionary to set
        """
        self.config = config
        
    def get(self):
        """
        Gets the config dictionary stored internally
        
        Returns:
            config (Dict): Configuration dictionary
        """
        return self.config

    @staticmethod
    def generate_filepath(path_to_folder: str, 
                          filename_prefix: str = "config", 
                          fmt: str = "json", 
                          random: bool = True) -> str:
        """
        Generates a destination filepath

        Args:
            path_to_folder (str): A path to the folder where file will live
            filename_prefix (str): A prefix for the generated file name.
            fmt (str): format for file (default: json)
            random (bool): Whether to append current timestamp to the filename.

        Returns:
            filepath (str): Generated filepath.

        Raises:
            ValueError: When no prefix is provided.
        """
        if filename_prefix is None or filename_prefix == "":
            raise ValueError("Parameter filename_prefix is None or empty.")
        timestamp = dt.datetime.now().strftime("_%Y%m%d%H%M%S") if random else ""
        return f"{path_to_folder}/{filename_prefix}{timestamp}.{fmt}"
