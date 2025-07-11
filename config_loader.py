import yaml
import numpy as np
import re

def parse_value(value):
    """
    Recursively parse values to handle special strings.
    """
    if isinstance(value, str):
        # Handle simple fractions
        match = re.match(r'([-+]?[\d\.]+)\s*\/\s*([\d\.]+)', value)
        if match:
            num = float(match.group(1))
            den = float(match.group(2))
            return num / den
            
        try:
            return float(value)
        except ValueError:
            return value
    elif isinstance(value, dict):
        return {k: parse_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [parse_value(v) for v in value]
    else:
        return value

def load_config(config_path):
    """
    Loads a YAML configuration file and parses special string values.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: The configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return parse_value(config)

if __name__ == '__main__':
    # Example usage:
    config = load_config('configs/v1_config.yml')
    import pprint
    pprint.pprint(config)
    print("\nParsed value for ang_dist_range_a2t:", config['target']['ang_dist_range_a2t'])
    print("Parsed value for action_range_scale:", config['action_range_scale'])
