import os

import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")


def load_config():
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
    return config


CONFIG = load_config()
