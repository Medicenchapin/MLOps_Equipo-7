import yaml

def dataload(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))
    raw_data_path = config['data_load']['raw_data_path']