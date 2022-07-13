import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def load_hp(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.load(file.read())  # has tuple
        except yaml.YAMLError as exc:
            print(exc)
    return config