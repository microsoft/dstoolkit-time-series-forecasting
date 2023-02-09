# Common

Module with common utilities used across the whole package, not specific to a single step of the pipeline. Specifically:

- Management of configuration file
- Management of data files

## config_parser.py

Class to manage the configuration file.

Class attribute store the path to the configuration file and a Python dictionary with its content. Includes methods to:

- Parse a JSON file into a python dictionary
- Get and set the value of a field in the configuration file

## config_file.py

Contains methods that can be used to create and validate the config file. It includes:

- Function to generate the name of the config file
- Function to save the config dictionary as a JSON to DBFS
- Function to verify that a created config file loads properly

## dataloader.py

Class to manage data files.

It contains methods to:

- Check whether a file exists
- Read/write files from/to a variety of formats and locations
