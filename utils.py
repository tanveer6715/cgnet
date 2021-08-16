import yaml
import os 

from functools import wraps
from time import process_time

def load_config(config_path):
    
    """Load configuration file in YAML format 
    Args : 
        config_path (string) : path to configuration file 

    Returns : 
        config (dict) : configuration in dictionary format
    
    """
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config

